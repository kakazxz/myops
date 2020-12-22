import numpy  as np
import tensorflow as tf
def reindex_for_myo_scar_edema(array):
    new_array=np.zeros(array.shape,np.uint16)
    new_array=new_array+np.where(array==1,200,0)
    new_array=new_array+np.where(array==3,1220,0)
    new_array=new_array+np.where(array==2,2221,0)
    return new_array

def reindex_for_myo_scar_edema_ZHANGZHEN(array):
    new_array = np.zeros(array.shape, np.uint16)
    new_array = new_array + np.where(array == 1, 200, 0)
    new_array = new_array + np.where(array == 2, 1220, 0)
    new_array = new_array + np.where(array == 3, 2221, 0)
    return new_array

def l2_loss(x, y):
    return tf.reduce_mean(tf.reduce_sum(tf.square(x - y), axis=-1))
