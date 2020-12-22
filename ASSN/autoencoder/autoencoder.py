from model.base_model import BaseModel
from cascade_multiseqseg.cascade_sampler import CascadeMyoSampler
from multiseqseg.ops import *
from multiseqseg.utils import *
from sitkImageIO.itkdatawriter import sitk_write_lab,sitk_write_image
import time
import os
import tensorflow as tf
from multiseqseg.dice_loss import soft_dice_loss

from dirutil.helper import get_name_wo_suffix
import SimpleITK as sitk
def swap_neighbor_labels_with_prob(batch_hard_segm, swap_prob=0.1):
    """Returns noised versions of hard segmentations, swapping the label
    of each pixel with the label of its left neighbor with a probability
    of swap_prob."""
    batch_size, height, width = batch_hard_segm.shape[:-1]
    corrupted = np.copy(batch_hard_segm)

    for i in range(batch_size):
        swap_map = np.random.choice(
            range(2), size=(height, width // 2), p=[1 - swap_prob, swap_prob])

        h_idx, w_idx = np.where(swap_map == 1)
        w_idx = 2 * w_idx + 1

        x_r_vals = corrupted[i, h_idx, w_idx, 0]
        x_l_vals = corrupted[i, h_idx, w_idx - 1, 0]

        corrupted[i, h_idx, w_idx, 0] = x_l_vals
        corrupted[i, h_idx, w_idx - 1, 0] = x_r_vals

    return corrupted

class AECNN(object):
    def __init__(self, name, args,is_train):
        self.name = 'AECNN'
        self.args=args
        self.is_train = is_train
        self.reuse = None
    def batch_norm(self,x, momentum=0.9, epsilon=1e-5, train=True, name="batch_norm"):
        # return batch_instance_norm(x, name)
        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon, scale=True, is_training=train, scope=name)

    def __conv_bn_relu(self,x, dim, ks=3, s=1, train=True, name='res'):
        y = self.batch_norm(tf.layers.conv2d(x, dim, ks, s, padding='same', use_bias=False, name=name + '_c1'),
                       train=train, name=name + '_bn1')
        y = tf.nn.relu(y)
        return y

    def __deconv_bn_relu(self,x, dim, ks=3, s=1, train=True, name='res'):
        y = self.batch_norm(tf.layers.conv2d_transpose(x, dim,ks,s ,padding='same', use_bias=False,name=name + '_c1'),
                       train=train, name=name + '_bn1')
        y = tf.nn.relu(y)
        return y
    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse):
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
            # output=tf.nn.softmax(x)
            output=x
        if self.reuse is None:
            self.var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.saver = tf.train.Saver(
                var_list=self.var_list, max_to_keep=10)
            self.reuse = True
        return code,output

    def save(self, sess, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, os.path.basename(self.args.dataset_dir))
        self.saver.save(sess, checkpoint_path, global_step=step)

    def restore(self, sess,checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("resotre:"+os.path.join(checkpoint_dir, ckpt_name))
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


class ShapeAutoEncode(BaseModel):
    def __init__(self, sess, config, train=True):
        self.sess = sess
        self.args = config
        self.batch_size = config.batch_size
        self.is_train = train
        self.reuse=None
        self.train_sampler = CascadeMyoSampler(config, True)
        self.valid_sampler = CascadeMyoSampler(config, False)
        self.__build_model(config=config, train=train)
        self.__build_cost()

    def __build_model(self,config,train):
        self.binary_shape = tf.placeholder(tf.float32, [self.batch_size, config.fine_size, config.fine_size, self.args.c_dim],name='shape')
        self.aecnn=AECNN('AECNN',self.args,self.is_train)
        self.h,self.output=self.aecnn(self.binary_shape)


    def __build_cost(self):
        self.g_loss= tf.reduce_mean(soft_dice_loss(self.binary_shape, self.output, axis=[1, 2]))
        # self.g_loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.binary_shape, logits=self.output))
        # loss=self.binary_shape*tf.log(self.output)
        # self.g_loss=tf.reduce_mean(tf.reduce_sum(loss),axis=[1,2,3])
        # self.g_loss= tf.reduce_mean(tf.reduce_mean(weighted_2Dbinary_cross_entropy(self.output, self.binary_shape), axis=[1, 2, 3]))

    def train(self, config):
        # self.saver = tf.train.Saver(max_to_keep=5)

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.writer = tf.summary.FileWriter(config.log_dir, self.sess.graph)
        start_time = time.time()
        print(tf.trainable_variables())
        if self.load(config.checkpoint_dir):
            print("An existing model was found in the checkpoint directory.")
        else:
            print("An existing model was not found in the checkpoint directory. Initializing a new one.")

        for itr in range(config.iteration):
            gt, c0, de, t2 = self.train_sampler.next_sample()
            gt=swap_neighbor_labels_with_prob(gt)
            feed_dict = {self.binary_shape: gt}
            _, errG = self.sess.run([g_optim, self.g_loss], feed_dict=feed_dict)
            # self.writer.add_summary(summary_str, counter)
            print("Epoch: [%2d]  time: %4.4f, g_loss: %.8f" % (itr, time.time() - start_time, errG))
            if np.mod(itr, self.args.sample_freq) == 1:
                self.sample_network(itr)
            if np.mod(itr, self.args.save_freq) == 1:
                self.save(config.checkpoint_dir, itr)
    def save(self, checkpoint_dir, step):
        print(tf.trainable_variables())
        self.aecnn.save(self.sess,checkpoint_dir,step)

    def sample_network(self,itr):
        gt, c0, de, t2 = self.valid_sampler.next_sample()
        output_dir=self.args.sample_dir
        feed_dict = {self.binary_shape: gt}
        pre_mask,errG = self.sess.run([self.output,self.g_loss], feed_dict=feed_dict)
        sitk_write_lab(pre_mask[0], dir=output_dir, name='%d_reconstruct_binary_Y' % (itr))
        sitk_write_image(pre_mask[0], dir=output_dir, name='%d_reconstruct_Y' % (itr))
        sitk_write_lab(gt[0], dir=output_dir, name='%d_gt' % (itr))

    def valid(self):
        self.aecnn.restore(self.sess,self.args.checkpoint_dir)
        nb=self.valid_sampler.num
        for i in range(nb):

            p_gts, p_c0s, p_des, p_t2s = self.valid_sampler.prepare_sample_path(1)
            gts, c0s, des, t2s = self.valid_sampler.get_batch_data(p_gts, p_c0s, p_des, p_t2s,True)
            feed_dict = {self.binary_shape: gts}
            pre_mask, errG = self.sess.run([self.output, self.g_loss], feed_dict=feed_dict)
            ref=sitk.ReadImage(p_gts[0])
            sitk_write_lab(np.argmax(pre_mask[0],axis=-1),parameter_img=ref, dir=self.args.valid_dir, name=get_name_wo_suffix(p_gts[0]))
