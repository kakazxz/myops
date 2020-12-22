'''
this new segmentation network with autoencoder
'''
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from multiseqseg.prepare_data import prepare_data,prepare_test_data
from cascade_multiseqseg.reconstruction import Reconstruction
from dirutil.helper import mk_or_cleardir
flags = tf.app.flags
flags.DEFINE_integer("iteration", 40000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
# flags.DEFINE_integer("load_size", 250, "The size of images to be loaded [250]")
flags.DEFINE_integer("fine_size", 128, "The fine size of images [256]")
# flags.DEFINE_integer("component", 500, "seg target label")
flags.DEFINE_string("components",'200,2221,1220','seg target label')
flags.DEFINE_integer("c_dim",2, "The channal size of the images [3]")
flags.DEFINE_string("dataset_dir", "../datasets/Reconstruction", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "../outputs/Reconstruction/checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "../outputs/Reconstruction/samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("test_dir", "../outputs/Reconstruction/test", "Directory name to save the image samples [samples]")
flags.DEFINE_string("valid_dir", "../outputs/Reconstruction/valid", "Directory name to save the image samples [samples]")
flags.DEFINE_string("log_dir", "../outputs/Reconstruction/logs", "Directory name to save the logs [logs]")
flags.DEFINE_string("results_dir", "../outputs/Reconstruction/logs", "test result")
flags.DEFINE_integer("save_freq", 2000, "Save frequency [0]")
flags.DEFINE_integer("sample_freq", 2000, "Save frequency [0]")
flags.DEFINE_integer("nb_label", 5, "number of label")

flags.DEFINE_float("ce_reg", 1, "")
flags.DEFINE_float("ac_reg", 1, "")
flags.DEFINE_float("dice_reg", 0, "")
flags.DEFINE_string("aecnn_model", '../outputs/autoencoder/checkpoint/' , "number of label")

flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_string("phase", "train", "train or test")


FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.69
with tf.Session(config=config) as sess:
    if FLAGS.phase=="train":
        prepare_data(FLAGS)
        mk_or_cleardir(FLAGS.checkpoint_dir)
        mk_or_cleardir(FLAGS.sample_dir)
        mk_or_cleardir(FLAGS.results_dir)
        mk_or_cleardir(FLAGS.log_dir)
        model = Reconstruction(sess, config=FLAGS, train=True)
        model.train(FLAGS)
    elif  FLAGS.phase=="test":
        prepare_test_data(FLAGS)
        mk_or_cleardir(FLAGS.test_dir)
        model = Reconstruction(sess, config=FLAGS, train=False)
        model.test()
    else:
        mk_or_cleardir(FLAGS.valid_dir)
        model = Reconstruction(sess, config=FLAGS, train=False)
        model.valid()