
import glob
import numpy as np
import SimpleITK as sitk
from dirutil.helper import sort_glob
from sitkImageIO.itkdatawriter import sitk_write_image,sitk_write_lab
from preprocessor.sitkSpatialAugment import augment_img_lab,augment_multi_imgs_lab
from scipy.stats import  zscore
from keras.utils import to_categorical
class Sampler():
    def __init__(self,config,is_train):
        self.args=config
        self.is_train=is_train
        if is_train==True:
            self.imgs=sort_glob("%s/train_img/*.nii.gz"%config.dataset_dir)
            self.labs=sort_glob("%s/train_lab/*.nii.gz"%config.dataset_dir)
        else:
            self.imgs=sort_glob("%s/valid_img/*.nii.gz"%config.dataset_dir)
            self.labs=sort_glob("%s/valid_lab/*.nii.gz"%config.dataset_dir)

        self.index=0
        self.num=len(self.labs)
    def reset_sequnce_index(self):
        self.index=0
    def next_sample(self,batch_size=None,is_pair=True):
        if batch_size==None:
            gts, c0s, des, t2s = self.prepare_sample_path(self.args.batch_size)
        else:
            gts, c0s, des, t2s = self.prepare_sample_path(batch_size)
        # print(gts)
        # print(c0s)
        # print(des)
        # print(t2s)
        return self.get_batch_data(gts,c0s,des,t2s,is_pair)

    def prepare_sample_path(self,batch_size):
        gts = []
        c0s = []
        des = []
        t2s = []
        for i in range(batch_size):
            gt, c0, de, t2 = self.select_sample()
            gts.append(gt)
            c0s.append(c0)
            des.append(de)
            t2s.append(t2)
        return gts,c0s, des,t2s

    def __get_file(self,id,slice,type):
        for p in self.imgs:
            if p.find("%s_%s_%s"%(id,type,slice))>=0 :
                return p
        return None

    def augmentUnpairMultiSeq(self, c0s, des, T2s, labs, img_size=96):
        aug_c0s, _ = augment_img_lab([c0s], [labs], img_size)
        aug_des, aug_labs = augment_img_lab([des], [labs], img_size)
        aug_t2s, _ = augment_img_lab([T2s], [labs], img_size)
        return aug_c0s[0], aug_des[0], aug_t2s[0], aug_labs[0]

    def augmentpairMultiSeq(self, c0s, des, t2s, labs, img_size=96):

        aug_c0,aug_des,aug_t2, aug_labs = augment_multi_imgs_lab([c0s],[des],[t2s], [labs], img_size)

        return aug_c0[0], aug_des[0], aug_t2[0], aug_labs[0]


    def select_sample(self):
        if self.is_train==True:
            i=np.random.randint(self.num)
        else:
            i=self.index
            self.index=(self.index+1)%self.num
        gt=self.labs[i]
        id=gt.split('_')[-3]
        slice=gt.split('_')[-1].split('.')[0]
        return  gt,self.__get_file(id,slice,'C0'),self.__get_file(id,slice,'DE'),self.__get_file(id,slice,'T2')
    def normalize(self,imgc0):
        tmp=sitk.GetArrayFromImage(imgc0)
        return tmp/127.0-1
    def get_batch_data(self,gts,c0s,des,t2s,is_pair):
        gt_lab = []
        c0_img = []
        de_img = []
        t2_img = []
        for gt,c0,de,t2 in zip(gts,c0s,des,t2s):
            # print(str(index_mv)+":"+str(index_fix))
            imgc0, imgde,imgt2 = sitk.ReadImage(c0), sitk.ReadImage(de), sitk.ReadImage(t2)
            lab= sitk.ReadImage(gt)

            #当在训练的时候可以随机进行数据augmentation
            if np.random.randint(4)!=1 and self.is_train==True:
                # imgc0,imgde,imgt2,lab= self.augmentUnpairMultiSeq(imgc0, imgde, imgt2, lab, self.args.fine_size)
                if is_pair:
                    imgc0,imgde,imgt2,lab= self.augmentpairMultiSeq(imgc0, imgde, imgt2, lab, self.args.fine_size)
                else:
                    imgc0,imgde,imgt2,lab= self.augmentUnpairMultiSeq(imgc0, imgde, imgt2, lab, self.args.fine_size)

                # sitk_write_image(imgc0,dir='../tmp',name='c0')
                # sitk_write_image(imgde,dir='../tmp',name='de')
                # sitk_write_image(imgt2,dir='../tmp',name='t2')
                # sitk_write_lab(lab,dir='../tmp',name='lab')

            imgc0, imgde,imgt2 = sitk.RescaleIntensity(imgc0), sitk.RescaleIntensity(imgde), sitk.RescaleIntensity(imgt2)

            c0_img.append(np.expand_dims(self.normalize(imgc0), axis=-1))
            de_img.append(np.expand_dims(self.normalize(imgde), axis=-1))
            t2_img.append(np.expand_dims(self.normalize(imgt2), axis=-1))

            lab=sitk.GetArrayFromImage(lab)



            one_hot = self.create_label(lab)
            # gt_lab.append(np.expand_dims(one_hot,axis=-1))
            gt_lab.append(one_hot)
        gt_lab = np.array(gt_lab).astype(np.float32)
        c0_img = np.array(c0_img).astype(np.float32)
        de_img = np.array(de_img).astype(np.float32)
        t2_img = np.array(t2_img).astype(np.float32)

        return gt_lab, c0_img,de_img,t2_img

    ##把实际的标签转换成所使用的标签
    def create_label(self, lab):
        one_hot = np.zeros(lab.shape, np.uint16)
        for index, i_str in enumerate(self.args.components.split(',')):
            i = int(i_str)
            one_hot = one_hot + np.where(lab == i, index + 1, 0)
        one_hot = to_categorical(one_hot, num_classes=self.args.c_dim)
        return one_hot