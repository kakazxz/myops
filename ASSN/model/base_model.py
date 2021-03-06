import tensorflow as tf
from logger.Logger import getLoggerV3
import os
class BaseModel():
    def __init__(self,sess,args):
        self.sess=sess
        self.args=args
        self.name=self.__class__.__name__
        self.logger=getLoggerV3(self.name,self.args.log_dir)

    def save(self, checkpoint_dir, step):
        model_name = os.path.basename(self.args.dataset_dir)+".model"

        checkpoint_dir = os.path.join(checkpoint_dir, os.path.basename(self.args.dataset_dir))

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir,id=None):
        print(" [*] Reading checkpoint...")

        # model_dir = "%s_%s" % (self.args.dataset_dir, self.args.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, os.path.basename(self.args.dataset_dir))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if id is not None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)+str(id)
            else:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
