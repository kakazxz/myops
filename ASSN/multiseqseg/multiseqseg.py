
import os
import time
from glob import glob
import tensorflow as tf
from random import shuffle
from sitkImageIO.itkdatawriter import sitk_write_labs,sitk_write_images,sitk_write_lab,sitk_write_image,sitk_write_multi_lab
from multiseqseg.ops import *
from multiseqseg.utils import *
from multiseqseg.load_data import load_data
from model.base_model import BaseModel
from multiseqseg.sampler import Sampler
from dirutil.helper import get_name_wo_suffix
from multiseqseg.dice_loss import soft_dice_loss
import  SimpleITK as sitk
from multiseqseg.challenge_sampler import ChallengeMyoSampler
from cascade_multiseqseg.tool import reindex_for_myo_scar_edema
class Multiseqseg(BaseModel):
    def __init__(self, sess, config, train=True):
        self.name='segment'
        self.sess = sess
        self.args=config
        self.batch_size = config.batch_size
        self.is_train=True
        self.train_sampler=Sampler(config, True)
        self.valid_sampler=Sampler(config, False)
        self.csampler = ChallengeMyoSampler(self.args)
        self.gf_dim = 32
        self.num_res_blocks =5
        self.build_model(config=config, train=train)

    def get_cost(self):
        return tf.reduce_mean(soft_dice_loss(self.gt,self.pre_mask,axis=[1,2]))
        #return tf.reduce_mean(soft_dice_loss(self.gt,self.pre_mask,axis=[1,2]))+tf.reduce_mean(tf.reduce_mean(weighted_2Dbinary_cross_entropy(self.gt, self.pre_mask), axis=[1, 2, 3]))
        # return tf.reduce_mean(tf.reduce_mean(weighted_2Dbinary_cross_entropy(self.gt, self.pre_mask), axis=[1, 2, 3]))

    def build_model(self, config, train):
        self.C0 = tf.placeholder(
            tf.float32, [self.batch_size, config.fine_size, config.fine_size,1 ],
            name='c0')
        self.DE = tf.placeholder(
            tf.float32, [self.batch_size, config.fine_size, config.fine_size,1],
            name='DE')
        self.T2 = tf.placeholder(
            tf.float32, [self.batch_size, config.fine_size, config.fine_size, 1],
            name='T2')
        self.gt = tf.placeholder(
            tf.float32, [self.batch_size, config.fine_size, config.fine_size, self.args.c_dim],
            name='gt')

        self.pre_mask = self.seg(self.C0, self.DE, self.T2, train=train)
        self.PreMask_summary = tf.summary.image("G", self.pre_mask)
        # self.g_loss = tf.reduce_mean((self.pre_mask - self.gt)**2) # after tonemapping
        self.g_loss =self.get_cost()
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=5)
        self.merge_summary = tf.summary.merge([self.PreMask_summary, self.g_loss_sum])

    def sample_network(self,itr):
        gt, c0, de, t2 = self.train_sampler.next_sample()
        output_dir=self.args.sample_dir

        feed_dict = {self.DE: de, self.C0: c0, self.T2: t2, self.gt: gt}

        pre_mask,errG = self.sess.run([self.pre_mask,self.g_loss], feed_dict=feed_dict)

        sitk_write_images(c0, dir=output_dir, name='%d_c0' % (itr))
        sitk_write_images(de, dir=output_dir, name='%d_de' % (itr))
        sitk_write_images(t2, dir=output_dir, name='%d_t2' % (itr))
        sitk_write_labs(gt, dir=output_dir, name='%d_Y' % (itr))
        sitk_write_images(pre_mask, dir=output_dir, name='%d_pre_Y' % (itr))
        sitk_write_labs(pre_mask, dir=output_dir, name='%d_pre_binary_Y' % (itr))

    def train(self, config):
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        init=tf.global_variables_initializer()
        self.sess.run(init)
        self.writer = tf.summary.FileWriter(config.log_dir, self.sess.graph)
        start_time = time.time()
        if self.load(config.checkpoint_dir):
            print("An existing model was found in the checkpoint directory.")
        else:
            print("An existing model was not found in the checkpoint directory. Initializing a new one.")
        
        for itr in range(config.iteration):
            gt,c0,de,t2=self.train_sampler.next_sample()

            feed_dict={self.DE:de,self.C0:c0,self.T2:t2,self.gt:gt}

            _,errG = self.sess.run([g_optim, self.g_loss],feed_dict=feed_dict)
            # self.writer.add_summary(summary_str, counter)

            print("Epoch: [%2d]  time: %4.4f, g_loss: %.8f"  % (itr,  time.time() - start_time, errG))
            if np.mod(itr, self.args.sample_freq) == 1:
                self.sample_network(itr)
            if np.mod(itr, self.args.save_freq) == 1:
                self.save(config.checkpoint_dir, itr)


    def test(self):
        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i in range(self.csampler.num):
            p_c0s,p_des,p_t2s = self.csampler.prepare_sample_path(1)
            c0s,des,t2s=self.csampler.get_batch_data(p_c0s, p_des, p_t2s, True)
            feed_dict = {self.DE: des, self.C0: c0s, self.T2: t2s}

            pre_mask = self.sess.run(self.pre_mask, feed_dict=feed_dict)

            ref=sitk.ReadImage(p_c0s[0])

            # sitk_write_image(c0s[0,:,:,0], parameter_img=ref,dir=self.args.test_dir+"/img/", name=get_name_wo_suffix(p_c0s[0]))
            # sitk_write_image(des[0,:,:,0], parameter_img=ref,dir=self.args.test_dir+"/img/", name=get_name_wo_suffix(p_des[0]))
            # sitk_write_image(t2s[0,:,:,0], parameter_img=ref,dir=self.args.test_dir+"/img/", name=get_name_wo_suffix(p_t2s[0]))
            # sitk_write_multi_lab(binary,parameter_img=ref, dir=self.args.test_dir, name=str(i)+"_"+get_name_wo_suffix(p_c0s[0]).replace('C0','gd'))
            # sitk_write_image(pre_mask[0],dir=self.args.test_dir, name=str(i)+"_"+get_name_wo_suffix(p_gts[0])+'_predict')
            binary=np.argmax(pre_mask[0],axis=-1)
            sitk_write_multi_lab((binary), parameter_img=ref,dir=self.args.test_dir, name=get_name_wo_suffix(p_c0s[0]).replace('C0','gd'))

    def seg(self, c0, DE, T2, train, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            s_h, s_w = self.args.fine_size, self.args.fine_size
            s2_h, s4_h, s8_h, s16_h, s2_w, s4_w, s8_w, s16_w = \
                int(s_h / 2), int(s_h / 4), int(s_h / 8), int(s_h / 16), int(s_w / 2), int(s_w / 4), int(s_w / 8), int(
                    s_w / 16)

            def residule_block(x, dim, ks=3, s=1, train=True, name='res'):
                p = int((ks - 1) / 2)
                y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = batch_norm(conv2d(y, dim, k_h=ks, k_w=ks, d_h=s, d_w=s, padding='VALID', name=name + '_c1'),
                               train=train, name=name + '_bn1')
                y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = batch_norm(conv2d(y, dim, k_h=ks, k_w=ks, d_h=s, d_w=s, padding='VALID', name=name + '_c2'),
                               train=train, name=name + '_bn2')
                return y + x

            image1 = c0
            image2 = DE
            image3 = T2
            with tf.variable_scope("encoder1"):
                # image is (256 x 256 x 1)
                e1_1 = conv2d(image1, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
                e1_2 = batch_norm(conv2d(lrelu(e1_1), self.gf_dim * 2, name='g_e2_conv'), train=train, name='g_e2_bn')

            with tf.variable_scope("encoder2"):
                # image is (256 x 256 x 1)
                e2_1 = conv2d(image2, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
                e2_2 = batch_norm(conv2d(lrelu(e2_1), self.gf_dim * 2, name='g_e2_conv'), train=train, name='g_e2_bn')

            with tf.variable_scope("encoder3"):
                # image is (256 x 256 x 1)
                e3_1 = conv2d(image3, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
                e3_2 = batch_norm(conv2d(lrelu(e3_1), self.gf_dim * 2, name='g_e2_conv'), train=train, name='g_e2_bn')

            with tf.variable_scope('merger'):
                e_2 = tf.concat([e1_2, e2_2, e3_2], 3)
                # e2 is (64 x 64 x self.gf_dim*2*3)
                e_3 = batch_norm(conv2d(lrelu(e_2), self.gf_dim * 4, name='g_e3_conv'), train=train, name='g_e3_bn')
                # e3 is (32 x 32 x self.gf_dim*4)

                res_layer = e_3
                for i in range(self.num_res_blocks):
                    # res_layer = batch_norm(conv2d(lrelu(res_layer), self.gf_dim*4, k_h=3, k_w=3, d_h=1, d_w=1,
                    #                              name='g_e5_conv_%d' %(i+1)), train=train, name='g_e5_bn_%d' %(i+1))
                    res_layer = residule_block(tf.nn.relu(res_layer), self.gf_dim * 4, ks=3, train=train,
                                               name='g_r%d' % (i + 1))

            with tf.variable_scope("decoder"):
                d0 = tf.concat([res_layer, e_3], 3)
                # d0 is (32 x 32 x self.gf_dim*4*2)
                d1 = batch_norm(conv2d_transpose(tf.nn.relu(d0),
                                                 [self.batch_size, s4_h, s4_w, self.gf_dim * 2], name='g_d1'),
                                train=train, name='g_d1_bn')
                d1 = tf.concat([d1, e1_2, e2_2, e3_2], 3)
                # d1 is (64 x 64 x self.gf_dim*2*4)
                d2 = batch_norm(conv2d_transpose(tf.nn.relu(d1),
                                                 [self.batch_size, s2_h, s2_w, self.gf_dim], name='g_d2'), train=train,
                                name='g_d2_bn')
                d2 = tf.concat([d2, e1_1, e2_1, e3_1], 3)
                # d2 is (128 x 128 x self.gf_dim*1*4)
                d3 = batch_norm(conv2d_transpose(tf.nn.relu(d2),
                                                 [self.batch_size, s_h, s_w, self.gf_dim], name='g_d3'), train=train,
                                name='g_d3_bn')
                # d3 is (256 x 256 x self.gf_dim)
                out = conv2d(tf.nn.relu(d3), self.args.c_dim, d_h=1, d_w=1, name='g_d_out_conv')
                # return tf.nn.tanh(out)
                # return tf.nn.sigmoid(out)
                return tf.nn.softmax(out)

