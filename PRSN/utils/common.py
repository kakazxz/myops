import SimpleITK as sitk
import numpy as np

from scipy import ndimage

import nibabel as nib
import skimage.transform as T
import os
from skimage import exposure

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
affine=np.array([[ -1.,-0.,-0. ,-0.],[ -0. ,-1. ,-0.,239.],[  0.  , 0. ,  1.,   0.],[  0. ,  0. ,  0.  , 1.]])

def resize(scan, resizeshape, order=0):
    """采样三维矩阵，scan:nii数组；sz:xy轴的维度；z_length：z轴切片数；order：采样方式，0为最近邻采样"""
    zoom_seq = np.array([resizeshape[0], resizeshape[1], resizeshape[2]], dtype='float') / np.array(scan.shape, dtype='float')
    s = ndimage.interpolation.zoom(scan, zoom_seq, order=order, prefilter=order)

    return s

def brats_dataset(dirname, filter="flair"):
    result = []  # 含有filter的所有的文件
    for maindir, subdir, file_name_list in os.walk(dirname):
        #
        # print("1:",maindir) #当前主目录
        # print("2:",subdir) #当前主目录下的所有目录
        # print("3:",file_name_list)  #当前主目录下的所有文件

        for filename in file_name_list:
            # print(filename)
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            # ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容
            # ext = apath.split("_")[-1]
            if filter in filename:
                result.append(apath)
    return result


def load_slicer(path,mode="train"):
    f_name = path
    t1 = f_name.replace("flair", "t1")
    # print(t1)
    t1ce = f_name.replace("flair", "t1ce")
    t2 = f_name.replace("flair", "t2")
    seg = f_name.replace("flair", "seg")
    p, gdname = os.path.split(seg)
    f_nii = nib.load(f_name)
    t1_nii = nib.load(t1)
    t1ce_nii = nib.load(t1ce)
    t2_nii = nib.load(t2)

    f_img = f_nii.get_data()
    t1_img = t1_nii.get_data()
    t1ce_img = t1ce_nii.get_data()
    t2_img = t2_nii.get_data()
    if mode=="test":
        seg_img=np.ones_like(t2_img)
    else:
        seg_nii = nib.load(seg)
        seg_img = seg_nii.get_data()

    # exposure.equalize_hist()
    return f_img, t1_img,t1ce_img, t2_img, seg_img, gdname
def load_slicer2(path,mode="train"):
    f_name = path
    seg = f_name.replace("ct", "seg")
    p, gdname = os.path.split(seg)
    f_nii = nib.load(f_name)
    f_img = f_nii.get_data()
    if mode=="test":
        seg_img=np.ones_like(f_img)
    else:
        seg_nii = nib.load(seg)
        seg_img = seg_nii.get_data()

    # exposure.equalize_hist()
    return f_img, seg_img, gdname

def standardization2(data):
    indices = np.where(data > 0)
    mean = data[indices].mean()
    std = data[indices].std()
    data[indices] = (data[indices] - mean) / std
    # 其他的值保持为0
    indices = np.where(data <= 0)
    data[indices] = 0
    return data


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    data = (data - mu) / sigma
    # data[data>1]=1
    # data[data > 1] = 1
    # print(data.dtype)
    return data

def normalization(data):
    range = np.max(data) - np.min(data)
    data = (data - np.min(data)) / range
    return data



def norm_img(image): # 归一化像素值到（0，1）之间，且将溢出值取边界值
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def sitk_read_row(img_path, resize_scale=1): # 读取3D图像并resale（因为一般医学图像并不是标准的[1,1,1]scale）
    nda = sitk.ReadImage(img_path)
    nda = sitk.GetArrayFromImage(nda)  # channel first
    nda=ndimage.zoom(nda,[resize_scale,resize_scale,resize_scale],order=0) #rescale

    return nda
def convert(label):
    label=np.rint(label)
    label1=np.zeros_like(label)
    label1[label!=0]=1

    label2=np.zeros_like(label)
    label2[label!=0]=1
    label2[label == 2] = 0

    label3=np.zeros_like(label)
    label3[label == 1] = 1
    label3[label == 3] = 2

    return label1,label2,label3,label


def nib_read_row(filename,crop_size,mode="train"):
    # print(filename)
    f_img, t1_img,t1ce_img, t2_img, seg_img, gdname=load_slicer(filename,mode)
    seg_img=np.rint(seg_img).astype("float")
    seg_img[seg_img==4]=3

    shape=seg_img.shape
    if crop_size!=False:
        f_img=T.resize(f_img.astype("float"),crop_size)
        t1_img = T.resize(t1_img.astype("float"), crop_size)
        t1ce_img =T.resize(t1ce_img.astype("float"), crop_size)
        t2_img = T.resize(t2_img.astype("float"), crop_size)
        seg_img = resize(seg_img.astype("float"), crop_size)#mode='constant'
    # array_img = nib.Nifti1Image(f_img, affine)
    # nib.save(array_img, os.path.join("./test", "%s" % gdname.replace("seg", "flair")))
    # array_img = nib.Nifti1Image(seg_img, affine)
    # nib.save(array_img, os.path.join("./test", "%s" % gdname.replace("seg", "seg")))
    return f_img, t1_img,t1ce_img, t2_img, seg_img, shape[0],shape[1],shape[2],gdname


def nib_read_row2(filename,crop_size,mode="train"):
    # print(filename)
    f_img, seg_img, gdname=load_slicer2(filename,mode)
    seg_img=np.rint(seg_img).astype("float")
    seg_img[seg_img==6]=1

    shape=seg_img.shape
    f_img=T.resize(f_img.astype("float"),crop_size)

    seg_img = resize(seg_img.astype("float"), crop_size)#mode='constant'

    return f_img,  seg_img, shape[0],shape[1],shape[2],gdname

def finame(gdname):
    gdname=gdname.split("_")
    return gdname[0]+'_'+gdname[1]+'_'+gdname[2]

def make_one_hot_3d(x, n): # 对输入的volume数据x，对每个像素值进行one-hot编码
    # print(x.max())
    one_hot = np.zeros([x.shape[0], x.shape[1], x.shape[2], n]) # 创建one-hot编码后shape的zero张量
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for v in range(x.shape[2]):
                one_hot[i, j, v, int(x[i, j, v])] = 1 # 给相应类别的位置置位1，模型预测结果也应该是这个shape
    return one_hot

import random


def random_crop_2d(img, label, crop_size):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]

    if random_x_max < 0 or random_y_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)

    crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1]]

    return crop_img, crop_label


def random_crop_3d(img, label, crop_size):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_z_max = img.shape[2] - crop_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)

    crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                 z_random:z_random + crop_size[2]]

    return crop_img, crop_label

def random_crop_3d2(f_img, t1_img,t1ce_img, t2_img, label, crop_size):
    random_x_max = f_img.shape[0] - crop_size[0]
    random_y_max = f_img.shape[1] - crop_size[1]
    random_z_max = f_img.shape[2] - crop_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)

    f_img = f_img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    t1_img = t1_img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    t1ce_img = t1ce_img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    t2_img = t2_img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]

    return f_img, t1_img,t1ce_img, t2_img, label,
def random_flip_3d(img1, img2,img3, img4, mask):
    rand1 = np.random.rand()
    rand2 = np.random.rand()
    if rand1 > 0.5:
        img1 = np.flip(img1,axis=0)
        img2 = np.flip(img2,axis=0)
        img3 = np.flip(img3,axis=0)
        img4 = np.flip(img4,axis=0)
        mask = np.flip(mask,axis=0)
    if rand2 > 0.5:
        img1 = np.flip(img1,axis=1)
        img2 = np.flip(img2,axis=1)
        img3 = np.flip(img3,axis=1)
        img4 = np.flip(img4,axis=1)
        mask = np.flip(mask,axis=1)
    return img1, img2, img3, img4, mask

    # return f_img, t1_img,t1ce_img, t2_img, label,

def random_gamma(img1, img2, img3, img4, mask):
    rand1 = np.random.rand()*1.2+0.6
    img1 = exposure.adjust_gamma(img1,rand1)
    img2 = exposure.adjust_gamma(img2,rand1)
    img3 = exposure.adjust_gamma(img3,rand1)
    img4 = exposure.adjust_gamma(img4,rand1)
    # mask = np.flip(mask, axis=1)
    return img1, img2, img3, img4, mask


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr