
from keras.utils import to_categorical
from multiseqseg.sampler import Sampler
from multiseqseg.challenge_sampler import ChallengeMyoSampler
from dirutil.helper import sort_glob
import numpy as np
import SimpleITK as sitk
import os
from dirutil.helper import get_name_wo_suffix
from sitkImageIO.itkdatawriter import sitk_write_image,sitk_write_lab
from preprocessor.tools import binarize_img, normalize_mask


class CascadeMyoSampler(Sampler):
    def __init__(self,config,is_train):
        super(CascadeMyoSampler, self).__init__(config, is_train)


    def create_label(self, lab):
        one_hot = to_categorical(lab, num_classes=self.args.c_dim)
        return one_hot

class CascadeMyoPathologySampler(Sampler):
    def __init__(self,config,is_train,is_valid=False):
        super(CascadeMyoPathologySampler, self).__init__(config, is_train)
        if is_train==True:
            self.imgs=sort_glob("%s/train_img/*.nii.gz"%config.dataset_dir)
            self.labs=sort_glob("%s/train_lab/*.nii.gz"%config.dataset_dir)
        else:
            self.imgs=sort_glob("%s/valid_img/*.nii.gz"%config.dataset_dir)
            self.labs=sort_glob("%s/valid_lab/*.nii.gz"%config.dataset_dir)

        if is_valid==True:
            self.imgs = sort_glob("../datasets/cascasdePathology/pre_seg_valid_img/*.nii.gz" )
            self.labs = sort_glob("../datasets/cascasdePathology/pre_seg_valid_mask/*gd*.nii.gz")


    def get_batch_data(self,gts,c0s,des,t2s,is_pair):
        gt_lab = []
        c0_img = []
        de_img = []
        t2_img = []
        for gt,c0,de,t2 in zip(gts,c0s,des,t2s):
            # print(str(index_mv)+":"+str(index_fix))
            imgc0, imgde,imgt2 = sitk.ReadImage(c0), sitk.ReadImage(de), sitk.ReadImage(t2)
            lab= sitk.ReadImage(gt)
            ids=[int(i) for i in self.args.components.split(',')]
            #当在训练的时候可以随机进行数据augmentation

            # sitk_write_image(imgc0, dir='../tmp', name='ori_c0')
            # sitk_write_image(imgde, dir='../tmp', name='ori_de')
            # sitk_write_image(imgt2, dir='../tmp', name='ori_t2')
            # sitk_write_lab(binarize_img(lab, ids), dir='../tmp', name='lab')

            if np.random.randint(4)!=1 and self.is_train==True:
                # imgc0,imgde,imgt2,lab= self.augmentUnpairMultiSeq(imgc0, imgde, imgt2, lab, self.args.fine_size)
                if is_pair:
                    imgc0,imgde,imgt2,lab= self.augmentpairMultiSeq(imgc0, imgde, imgt2, lab, self.args.fine_size)
                else:
                    imgc0,imgde,imgt2,lab= self.augmentUnpairMultiSeq(imgc0, imgde, imgt2, lab, self.args.fine_size)

            # imgc0, imgde,imgt2 = sitk.RescaleIntensity(imgc0), sitk.RescaleIntensity(imgde), sitk.RescaleIntensity(imgt2)


            binarized_lab=binarize_img(lab, ids)
            imgc0=normalize_mask(imgc0,binarized_lab )
            imgde=normalize_mask(imgde,binarized_lab)
            imgt2=normalize_mask(imgt2, binarized_lab)

            #处理前面数据增强的导致会有边缘插值的问题。
            # sitk_write_image(imgc0, dir='../tmp', name='_c0')
            # sitk_write_image(imgde, dir='../tmp', name='_de')
            # sitk_write_image(imgt2, dir='../tmp', name='_t2')
            # sitk_write_lab(binarize_img(lab, ids), dir='../tmp', name='lab')

            c0_img.append(np.expand_dims(sitk.GetArrayFromImage(imgc0), axis=-1))
            de_img.append(np.expand_dims(sitk.GetArrayFromImage(imgde), axis=-1))
            t2_img.append(np.expand_dims(sitk.GetArrayFromImage(imgt2), axis=-1))

            lab=sitk.GetArrayFromImage(lab)
            one_hot = self.create_label(lab)
            # gt_lab.append(np.expand_dims(one_hot,axis=-1))
            gt_lab.append(one_hot)
        gt_lab = np.array(gt_lab).astype(np.float32)
        c0_img = np.array(c0_img).astype(np.float32)
        de_img = np.array(de_img).astype(np.float32)
        t2_img = np.array(t2_img).astype(np.float32)

        return gt_lab, c0_img,de_img,t2_img

class CascasedChallengeSample(ChallengeMyoSampler):
    def __init__(self,config):
        super(CascasedChallengeSample,self).__init__(config)
        self.C0_imgs=sort_glob("%s/challenge_pre_seg_test_img/*C0*.nii.gz" % config.dataset_dir)
        self.DE_imgs=sort_glob("%s/challenge_pre_seg_test_img/*DE*.nii.gz" % config.dataset_dir)
        self.T2_imgs=sort_glob("%s/challenge_pre_seg_test_img/*T2*.nii.gz" % config.dataset_dir)
        self.pre_seg=sort_glob('../outputs/cascadeMyo/test'+"/*gd*.nii.gz")
        self.myo_seg_dir=self.args.myo_seg_dir
        assert len(self.DE_imgs)==len(self.T2_imgs)
        assert len(self.C0_imgs)==len(self.T2_imgs)
        self.num=len(self.DE_imgs)

    def get_batch_data(self,c0s,des,t2s,is_pair):
        c0_img = []
        de_img = []
        t2_img = []
        for  c0,de,t2 in zip(c0s,des,t2s):
            # print(str(index_mv)+":"+str(index_fix))
            imgc0, imgde,imgt2 = sitk.ReadImage(c0), sitk.ReadImage(de), sitk.ReadImage(t2)
            lab=sitk.ReadImage(self.myo_seg_dir+"/"+os.path.basename(c0).replace("C0","gd"))
            binarized_lab = binarize_img(lab, [1])
            imgc0 = normalize_mask(imgc0, binarized_lab)
            imgde = normalize_mask(imgde, binarized_lab)
            imgt2 = normalize_mask(imgt2, binarized_lab)

            c0_img.append(np.expand_dims(sitk.GetArrayFromImage(imgc0), axis=-1))
            de_img.append(np.expand_dims(sitk.GetArrayFromImage(imgde), axis=-1))
            t2_img.append(np.expand_dims(sitk.GetArrayFromImage(imgt2), axis=-1))

        c0_img = np.array(c0_img).astype(np.float32)
        de_img = np.array(de_img).astype(np.float32)
        t2_img = np.array(t2_img).astype(np.float32)

        return c0_img,de_img,t2_img

class CascasedValidSample(CascasedChallengeSample):
    def __init__(self,config):
        super(CascasedValidSample,self).__init__(config)
        self.C0_imgs=sort_glob("%s/pre_seg_valid_img/*C0*.nii.gz" % config.dataset_dir)
        self.DE_imgs=sort_glob("%s/pre_seg_valid_img/*DE*.nii.gz" % config.dataset_dir)
        self.T2_imgs=sort_glob("%s/pre_seg_valid_img/*T2*.nii.gz" % config.dataset_dir)
        # self.myo_seg_dir = '../outputs/cascadeMyo/valid'
        assert len(self.DE_imgs)==len(self.T2_imgs)
        assert len(self.C0_imgs)==len(self.T2_imgs)
        self.num=len(self.DE_imgs)

