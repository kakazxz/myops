
import glob
import numpy as np
import SimpleITK as sitk
from sitkImageIO.itkdatawriter import sitk_write_image,sitk_write_lab
from preprocessor.sitkSpatialAugment import augment_img_lab,augment_multi_imgs_lab
from scipy.stats import  zscore
from keras.utils import to_categorical
from dirutil.helper import sort_glob
class ChallengeMyoSampler():
    def __init__(self,config):
        self.args=config
        self.is_train=False
        self.C0_imgs=sort_glob("%s/challenge_test_img/*C0*.nii.gz" % config.dataset_dir)
        self.DE_imgs=sort_glob("%s/challenge_test_img/*DE*.nii.gz" % config.dataset_dir)
        self.T2_imgs=sort_glob("%s/challenge_test_img/*T2*.nii.gz" % config.dataset_dir)
        self.index=0
        assert len(self.DE_imgs)==len(self.T2_imgs)
        assert len(self.C0_imgs)==len(self.T2_imgs)
        self.num=len(self.DE_imgs)

    def prepare_sample_path(self,batch_size):
        c0s = []
        des = []
        t2s = []
        for i in range(batch_size):
            c0, de, t2 = self.select_sample()
            c0s.append(c0)
            des.append(de)
            t2s.append(t2)
        return c0s, des,t2s

    def select_sample(self):
        if self.is_train==True:
            i=np.random.randint(self.num)
        else:
            i=self.index
            self.index=(self.index+1)%self.num
        return self.C0_imgs[i], self.DE_imgs[i], self.T2_imgs[i]

    def normalize(self,imgc0):
        tmp=sitk.GetArrayFromImage(imgc0)
        return tmp/127.0-1
    def get_batch_data(self,c0s,des,t2s,is_pair):
        c0_img = []
        de_img = []
        t2_img = []
        for c0,de,t2 in zip(c0s,des,t2s):
            # print(str(index_mv)+":"+str(index_fix))
            imgc0, imgde,imgt2 = sitk.ReadImage(c0), sitk.ReadImage(de), sitk.ReadImage(t2)
            imgc0, imgde,imgt2 = sitk.RescaleIntensity(imgc0), sitk.RescaleIntensity(imgde), sitk.RescaleIntensity(imgt2)
            c0_img.append(np.expand_dims(self.normalize(imgc0), axis=-1))
            de_img.append(np.expand_dims(self.normalize(imgde), axis=-1))
            t2_img.append(np.expand_dims(self.normalize(imgt2), axis=-1))

        c0_img = np.array(c0_img).astype(np.float32)
        de_img = np.array(de_img).astype(np.float32)
        t2_img = np.array(t2_img).astype(np.float32)

        return c0_img,de_img,t2_img