import time

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from medpy.metric import hd

from autoencoder.autoencoder import AECNN
from cascade_multiseqseg.cascade_network import ACMyoMultiSeq
from cascade_multiseqseg.tool import l2_loss
from dirutil.helper import get_name_wo_suffix
from evaluate.metric import calculate_binary_dice
from multiseqseg.dice_loss import soft_dice_loss
from sitkImageIO.itkdatawriter import sitk_write_lab

'''
the end2end version
'''
class Reconstruction(ACMyoMultiSeq):

    def batch_norm(self,x, momentum=0.9, epsilon=1e-5, train=True, name="batch_norm"):
        # return batch_instance_norm(x, name)
        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon, scale=True, is_training=train, scope=name)

    def __conv_bn_relu(self,x, dim, ks=3, s=1, train=True, name='res'):
        y = self.batch_norm(tf.layers.conv2d(x, dim, ks, s, padding='same', use_bias=False, name=name + '_c1'),train=train, name=name + '_bn1')
        y = tf.nn.relu(y)
        return y

    def __deconv_bn_relu(self,x, dim, ks=3, s=1, train=True, name='res'):
        y = self.batch_norm(tf.layers.conv2d_transpose(x, dim,ks,s ,padding='same', use_bias=False,name=name + '_c1'),
                       train=train, name=name + '_bn1')
        y = tf.nn.relu(y)
        return y
    def cnnae(self,name, x):
        with tf.variable_scope(name):
            dim = [4, 8, 16, 32]
            x=tf.layers.conv2d(x,4,3,padding='SAME',name='conv_in')
            x=self.__conv_bn_relu(x,dim[0],3,2,self.is_train,'encoder1')
            x=self.__conv_bn_relu(x,dim[1],3,2,self.is_train,'encoder2')
            x=self.__conv_bn_relu(x,dim[2],3,2,self.is_train,'encoder3')
            x=self.__conv_bn_relu(x,dim[3],3,2,self.is_train,'encoder4')
            x=tf.layers.flatten(x)
            code=tf.layers.dense(x,16)
            #decod
            x=tf.layers.dense(code,16*8*8)
            x=tf.reshape(x,[-1,8,8,16])
            x=self.__deconv_bn_relu(x,dim[2],3,2,self.is_train,'decoder1')
            x=self.__deconv_bn_relu(x,dim[1],3,2,self.is_train,'decoder2')
            x=self.__deconv_bn_relu(x,dim[0],3,2,self.is_train,'decoder3')
            x=self.__deconv_bn_relu(x,4,3,2,self.is_train,'deencoder4')
            x=tf.layers.conv2d(x,self.args.c_dim,3,padding='SAME',name='conv_out')
            output=tf.nn.softmax(x)
        return code,output

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
        # self.cnnae=AECNN('AECNN',self.args,train)
        self.pre_mask = self.seg(self.C0, self.DE, self.T2, train=train)

        # with tf.name_scope('AE_1'):
        #     self.ae_h1,self.ae_out1=self.cnnae(self.gt)
        self.ae_h2,self.ae_out2_rebuild=self.cnnae('AECNN',self.pre_mask)


        self.build_cost()

        self.binary_pre_mask=tf.expand_dims(tf.cast(tf.argmax(self.pre_mask,axis=-1),dtype=tf.uint8),-1)
        tf.summary.image("pre_mask", self.binary_pre_mask*255,max_outputs=9)

        self.rebuild_mask= tf.expand_dims(tf.cast(tf.argmax(self.ae_out2_rebuild, axis=-1), dtype=tf.uint8), -1)
        tf.summary.image("rebuild", self.rebuild_mask*255,max_outputs=9)

        tf.summary.scalar("g_loss", self.g_loss)
        tf.summary.scalar("ce_loss", self.ce_loss)
        tf.summary.scalar("dice_loss", self.dice_loss)
        tf.summary.scalar("r_loss", self.r_loss)

        #self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        #注意只能对这些变量进行优化
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        #rebuild DAE
        self.r_vars=[var for var in t_vars if 'AECNN' in var.name]

        self.saver = tf.train.Saver(max_to_keep=5)
        self.merge_summary = tf.summary.merge_all()

    def build_cost(self):
        self.dice_loss=self.args.dice_reg*tf.reduce_mean(soft_dice_loss(self.gt, self.pre_mask, axis=[1, 2]))
        self.ce_loss=self.args.ce_reg*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.pre_mask, logits=self.gt))
        self.g_loss=self.dice_loss+self.ce_loss
        # self.ac_loss=self.args.ac_reg*l2_loss(self.ae_h1,self.ae_h2)
        self.r_loss=l2_loss(self.gt,self.ae_out2_rebuild)

    def train(self, config):
        self.g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss,var_list=self.g_vars)
        self.r_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.r_loss,var_list=self.r_vars)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # self.cnnae.restore(self.sess,self.args.aecnn_model)
        self.writer = tf.summary.FileWriter(config.log_dir, self.sess.graph)
        start_time = time.time()
        if self.load(config.checkpoint_dir):
            print("An existing model was found in the checkpoint directory.")
        else:
            print("An existing model was not found in the checkpoint directory. Initializing a new one.")

        for itr in range(10000):
            gt, c0, de, t2 = self.train_sampler.next_sample()

            feed_dict = {self.DE: de, self.C0: c0, self.T2: t2, self.gt: gt}

            errG,summary = self.optimiz_one_step('G',feed_dict,itr)
            self.writer.add_summary(summary, itr)

            print("Epoch: [%2d]  time: %4.4f, %s loss: %.8f" % (itr, time.time() - start_time, "G",errG))
            if np.mod(itr, self.args.sample_freq) == 1:
                self.sample_network(itr)
            if np.mod(itr, self.args.save_freq) == 1:
                self.save(config.checkpoint_dir, itr)

        self.trainable_part="G"
        for itr in range(10000,config.iteration):
            gt, c0, de, t2 = self.train_sampler.next_sample()

            feed_dict = {self.DE: de, self.C0: c0, self.T2: t2, self.gt: gt}

            # _, errG, summary = self.sess.run([g_optim, self.g_loss, self.merge_summary], feed_dict=feed_dict)
            # self.writer.add_summary(summary, itr)
            err,summary=self.optimiz_one_step('R',feed_dict,itr)
            self.writer.add_summary(summary, itr)

            print("Epoch: [%2d]  time: %4.4f, %s _loss: %.8f" % (itr, time.time() - start_time,'R', err))
            if np.mod(itr, self.args.sample_freq) == 1:
                self.sample_network(itr)
            if np.mod(itr, self.args.save_freq) == 1:
                self.save(config.checkpoint_dir, itr)

    def optimiz_one_step(self,part,feed_dict,itr):
        if part=='R':
            _, err, summary = self.sess.run([self.r_optim, self.r_loss, self.merge_summary], feed_dict=feed_dict)

        elif part=="G":
            _, err, summary = self.sess.run([self.g_optim, self.g_loss, self.merge_summary], feed_dict=feed_dict)

        return err,summary

    def valid(self):
        self.is_train = False
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load(self.args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        dice=[]
        hs=[]
        for i in range(self.valid_sampler.num):
            p_gts,p_c0s,p_des,p_t2s = self.valid_sampler.prepare_sample_path(1)
            gts,c0s,des,t2s=self.valid_sampler.get_batch_data(p_gts,p_c0s, p_des, p_t2s, True)
            feed_dict = {self.C0: c0s, self.DE: des,  self.T2: t2s}
            pre_mask,rebuild_mask,rebuild_out = self.sess.run([self.pre_mask,self.rebuild_mask,self.ae_out2_rebuild], feed_dict=feed_dict)
            ref=sitk.ReadImage(p_c0s[0])
            binary=np.argmax(pre_mask[0],axis=-1)

            dice.append(calculate_binary_dice(np.squeeze(np.argmax(gts[0], axis=-1)), (np.squeeze(np.argmax(rebuild_out, -1))).astype(np.int16)))
            hs.append(hd(np.squeeze(np.argmax(gts[0], axis=-1)), (np.squeeze(rebuild_mask)).astype(np.int16),ref.GetSpacing()))
            # sitk_write_multi_lab((binary), parameter_img=ref,dir=self.args.valid_dir, name=get_name_wo_suffix(p_c0s[0]).replace('C0','gd'))
            sitk_write_lab((np.squeeze(rebuild_mask)), parameter_img=ref,dir=self.args.valid_dir, name=get_name_wo_suffix(p_c0s[0]).replace('C0','recons'))
            sitk_write_lab((np.squeeze(binary)), parameter_img=ref,dir=self.args.valid_dir, name=get_name_wo_suffix(p_c0s[0]).replace('C0','pre_seg'))

        print(np.mean(dice))
        print(np.std(dice))

        print(np.mean(hs))
        print(hs)
        print(np.std(hs))