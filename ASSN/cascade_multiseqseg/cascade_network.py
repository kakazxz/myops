import tensorflow as tf
import SimpleITK as sitk
import time
import numpy as np
from sitkImageIO.itkdatawriter import sitk_write_multi_lab
from dirutil.helper import get_name_wo_suffix
from multiseqseg.multiseqseg import Multiseqseg
from multiseqseg.dice_loss import soft_dice_loss
from cascade_multiseqseg.tool import reindex_for_myo_scar_edema
from autoencoder.autoencoder import AECNN
from sitkImageIO.itkdatawriter import sitk_write_image,sitk_write_lab
from cascade_multiseqseg.tool import l2_loss
from evaluate.metric import calculate_binary_dice
from medpy.metric import hd
from cascade_multiseqseg.cascade_sampler import CascadeMyoSampler,CascadeMyoPathologySampler,CascasedChallengeSample,CascasedValidSample

class ACMyoMultiSeq(Multiseqseg):
    def __init__(self,sess,config,train):
        super(ACMyoMultiSeq,self).__init__(sess,config,train)

        self.train_sampler=CascadeMyoSampler(self.args, True)
        self.valid_sampler=CascadeMyoSampler(self.args, False)

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
        self.cnnae=AECNN('AECNN',self.args,train)
        self.pre_mask = self.seg(self.C0, self.DE, self.T2, train=train)

        with tf.name_scope('AE_1'):
            self.ae_h1,self.ae_out1=self.cnnae(self.gt)
        with tf.name_scope('AE_2'):
            self.ae_h2,self.ae_out2_rebuild=self.cnnae(self.pre_mask)


        self.g_loss =self.get_cost()

        self.binary_pre_mask=tf.expand_dims(tf.cast(tf.argmax(self.pre_mask,axis=-1),dtype=tf.uint8),-1)
        tf.summary.image("pre_mask", self.binary_pre_mask*255,max_outputs=9)

        self.rebuild_mask= tf.expand_dims(tf.cast(tf.argmax(self.ae_out2_rebuild, axis=-1), dtype=tf.uint8), -1)
        tf.summary.image("rebuild", self.rebuild_mask*255,max_outputs=9)

        tf.summary.scalar("g_loss", self.g_loss)
        tf.summary.scalar("ce_loss", self.ce_loss)
        tf.summary.scalar("ac_loss", self.ac_loss)
        tf.summary.scalar("dice_loss", self.dice_loss)

        #self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        #注意只能对这些变量进行优化
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(var_list=self.g_vars,max_to_keep=5)
        self.merge_summary = tf.summary.merge_all()

    def get_cost(self):
        self.dice_loss=self.args.dice_reg*tf.reduce_mean(soft_dice_loss(self.gt, self.pre_mask, axis=[1, 2]))
        self.ce_loss=self.args.ce_reg*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.pre_mask, logits=self.gt))
        # self.ac_loss=self.args.ac_reg*l2_loss(self.ae_h1,self.ae_h2)
        self.ac_loss=self.args.ac_reg*l2_loss(self.ae_out1,self.ae_out2_rebuild)
        return self.dice_loss+self.ce_loss+self.ac_loss

    def sample_network(self, itr):
        gt, c0, de, t2 = self.valid_sampler.next_sample()
        output_dir = self.args.sample_dir
        feed_dict = {self.DE: de, self.C0: c0, self.T2: t2, self.gt: gt}
        binary_pre_mask, errG ,rebuild= self.sess.run([self.binary_pre_mask, self.g_loss,self.ae_out2_rebuild], feed_dict=feed_dict)
        sitk_write_image(c0[0], dir=output_dir, name='%d_c0' % (itr))
        sitk_write_image(de[0], dir=output_dir, name='%d_de' % (itr))
        sitk_write_image(t2[0], dir=output_dir, name='%d_t2' % (itr))
        sitk_write_lab(np.expand_dims(np.argmax(gt[0],-1),-1), dir=output_dir, name='%d_Y' % (itr))
        sitk_write_image(binary_pre_mask[0], dir=output_dir, name='%d_pre_Y' % (itr))
        sitk_write_lab(binary_pre_mask[0], dir=output_dir, name='%d_pre_binary_Y' % (itr))
        sitk_write_lab(rebuild[0], dir=output_dir, name='%d_rebuild_Y' % (itr))

    def train(self, config):
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss,var_list=self.g_vars)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.cnnae.restore(self.sess,self.args.aecnn_model)
        self.writer = tf.summary.FileWriter(config.log_dir, self.sess.graph)
        start_time = time.time()
        if self.load(config.checkpoint_dir):
            print("An existing model was found in the checkpoint directory.")
        else:
            print("An existing model was not found in the checkpoint directory. Initializing a new one.")

        for itr in range(config.iteration):
            gt, c0, de, t2 = self.train_sampler.next_sample()

            feed_dict = {self.DE: de, self.C0: c0, self.T2: t2, self.gt: gt}

            _, errG,summary = self.sess.run([g_optim, self.g_loss,self.merge_summary], feed_dict=feed_dict)
            self.writer.add_summary(summary, itr)

            print("Epoch: [%2d]  time: %4.4f, g_loss: %.8f" % (itr, time.time() - start_time, errG))
            if np.mod(itr, self.args.sample_freq) == 1:
                self.sample_network(itr)
            if np.mod(itr, self.args.save_freq) == 1:
                self.save(config.checkpoint_dir, itr)

    def test(self):
        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.cnnae.restore(self.sess,self.args.aecnn_model)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i in range(self.csampler.num):
            p_c0s,p_des,p_t2s = self.csampler.prepare_sample_path(1)
            c0s,des,t2s=self.csampler.get_batch_data(p_c0s, p_des, p_t2s, True)
            feed_dict = {self.DE: des, self.C0: c0s, self.T2: t2s}

            pre_mask,rebuild_mask = self.sess.run([self.pre_mask,self.rebuild_mask], feed_dict=feed_dict)

            ref=sitk.ReadImage(p_c0s[0])

            # sitk_write_image(c0s[0,:,:,0], parameter_img=ref,dir=self.args.test_dir+"/img/", name=get_name_wo_suffix(p_c0s[0]))
            # sitk_write_image(des[0,:,:,0], parameter_img=ref,dir=self.args.test_dir+"/img/", name=get_name_wo_suffix(p_des[0]))
            # sitk_write_image(t2s[0,:,:,0], parameter_img=ref,dir=self.args.test_dir+"/img/", name=get_name_wo_suffix(p_t2s[0]))
            # sitk_write_multi_lab(binary,parameter_img=ref, dir=self.args.test_dir, name=str(i)+"_"+get_name_wo_suffix(p_c0s[0]).replace('C0','gd'))
            # sitk_write_image(pre_mask[0],dir=self.args.test_dir, name=str(i)+"_"+get_name_wo_suffix(p_gts[0])+'_predict')
            # binary=np.argmax(rebuild_mask[0],axis=-1)
            # sitk_write_multi_lab((binary), parameter_img=ref,dir=self.args.test_dir, name=get_name_wo_suffix(p_c0s[0]).replace('C0','gd'))
            sitk_write_lab((np.squeeze(rebuild_mask)), parameter_img=ref,dir=self.args.test_dir, name=get_name_wo_suffix(p_c0s[0]).replace('C0','gd'))


    def valid(self):
        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.cnnae.restore(self.sess,self.args.aecnn_model)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        dice=[]
        hs=[]
        for i in range(self.csampler.num):
            p_gts,p_c0s,p_des,p_t2s = self.valid_sampler.prepare_sample_path(1)
            gts,c0s,des,t2s=self.valid_sampler.get_batch_data(p_gts,p_c0s, p_des, p_t2s, True)
            feed_dict = {self.DE: des, self.C0: c0s, self.T2: t2s}
            pre_mask,rebuild_mask = self.sess.run([self.pre_mask,self.rebuild_mask], feed_dict=feed_dict)
            ref=sitk.ReadImage(p_c0s[0])
            binary=np.argmax(pre_mask[0],axis=-1)

            dice.append(calculate_binary_dice(np.squeeze(np.argmax(gts[0], axis=-1)), (np.squeeze(rebuild_mask) / 255).astype(np.int16)))
            hs.append(hd(np.squeeze(np.argmax(gts[0], axis=-1)), (np.squeeze(rebuild_mask) / 255).astype(np.int16),ref.GetSpacing()))
            # sitk_write_multi_lab((binary), parameter_img=ref,dir=self.args.valid_dir, name=get_name_wo_suffix(p_c0s[0]).replace('C0','gd'))
            sitk_write_lab((np.squeeze(rebuild_mask)), parameter_img=ref,dir=self.args.valid_dir, name=get_name_wo_suffix(p_c0s[0]).replace('C0','gd'))

        print(np.mean(dice))
        print(np.std(dice))

        print(np.mean(hs))
        print(hs)
        print(np.std(hs))

