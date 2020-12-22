import glob
import SimpleITK as sitk
from preprocessor.tools import crop_by_bbox,get_bounding_box_by_ids,sitkResize3DV2,sitkResample3DV2
import os
from dirutil.helper import sort_glob
import numpy as np
from preprocessor.tools import binarize_img,convertArrayToImg
from sitkImageIO.itkdatawriter import sitk_write_image,write_png_image,write_png_lab
from dirutil.helper import mkdir_if_not_exist
from preprocessor.tools import reindex_label
from dirutil.helper import get_name_wo_suffix
from dirutil.helper import mk_or_cleardir
from preprocessor.sitkOPtool import recast_pixel_val
def prepare_masked_data(config):

    if os.path.exists(config.dataset_dir):
        print('data already exists')
        return
    files = sort_glob("../../dataset/myops/train25_myops_gd_convert/*.nii.gz")
    output_lab_dir="%s/train_lab"%(config.dataset_dir)
    output_img_dir="%s/train_img"%(config.dataset_dir)
    prepare_slices(config,files[:20],output_lab_dir,output_img_dir)

    output_lab_dir="%s/valid_lab"%(config.dataset_dir)
    output_img_dir="%s/valid_img"%(config.dataset_dir)
    prepare_slices(config,files[-5:],output_lab_dir,output_img_dir)

def merge_slice(config):

    for index in range(201,221):
        files = sort_glob(config.myo_seg_dir + "/*%d_gd*.nii.gz"%(index))
        volume=[]
        for f in files:
            img=sitk.ReadImage(f)
            volume.append(sitk.GetArrayFromImage(img))
        volume=np.array(volume).astype(np.uint16)

        sitk_write_image(volume,'../../dataset/myops/crop_20_label/','myops_test_%d_gd'%(index))

def prepare_slices(config,files,output_lab_dir,output_img_dir):
    mk_or_cleardir(output_lab_dir)
    mk_or_cleardir(output_img_dir)
    for p in files:
        lab = sitk.ReadImage(p)

        ids=[int(item) for item in config.components.split(',')]
        bbox=get_bounding_box_by_ids(sitk.GetArrayFromImage(lab),padding=10,ids=ids)
        ##extend bbox
        crop_lab=crop_by_bbox(lab,bbox)
        crop_lab=sitkResize3DV2(crop_lab,[config.fine_size,config.fine_size,crop_lab.GetSize()[-1]],sitk.sitkNearestNeighbor)
        #
        binary_crop_lab = binarize_img(crop_lab, ids)
        for i in range(crop_lab.GetSize()[-1]):
            sitk_write_image(crop_lab[:,:,i],dir=output_lab_dir,name="%s_%d"%(os.path.basename(p).split('.')[0],i))
            img_file=sort_glob("../../dataset/myops/train25_convert/*%s*.nii.gz"%(os.path.basename(p).split("_")[2]))
            for j in img_file:
                img = sitk.ReadImage(j)
                # img = sitkResample3DV2(img, sitk.sitkLinear, [1, 1, 1])
                crop_img=crop_by_bbox(img,bbox)
                crop_img = sitkResize3DV2(crop_img, [config.fine_size, config.fine_size,crop_img.GetSize()[-1]], sitk.sitkLinear)
                sitk_write_image(crop_img[:,:,i]*recast_pixel_val(crop_img[:,:,i],binary_crop_lab[:,:,i]), dir=output_img_dir, name="%s_%d"%(os.path.basename(j).split('.')[0],i))

def prepare_masked_test_data(config):
    # merge_slice(config)

    mask = sort_glob('../outputs/ACMyo/test'+"/*gd*.nii.gz")
    tobe_masked_img='../datasets/cascadeMyo/challenge_test_img'
    output_img_dir = "%s/challenge_pre_seg_test_img" % (config.dataset_dir)

    for lab in mask:
        sitk_lab=sitk.ReadImage(lab)
        name=get_name_wo_suffix(lab)

        id,slice=name.split('_')[2],name.split('_')[4]

        for type in ['C0','DE','T2']:
            img_name='myops_test_%s_%s_%s.nii.gz'%(id,type,slice)
            sitk_img=sitk.ReadImage(tobe_masked_img+"/"+img_name)
            sitk_write_image(sitk_img * recast_pixel_val(sitk_img,sitk_lab),dir=output_img_dir, name=get_name_wo_suffix(img_name))

def prepare_masked_valid_data(config):
    # merge_slice(config)
    masks = sort_glob('../outputs/ACMyo/valid'+"/*gd*.nii.gz")
    tobe_masked_img='../datasets/cascadeMyo/valid_img'
    tobe_masked_lab='../datasets/cascadeMyo/valid_lab'
    output_img_dir = "%s/pre_seg_valid_img" % (config.dataset_dir)
    output_lab_dir = "%s/pre_seg_valid_lab" % (config.dataset_dir)

    for mask in masks:
        sitk_mask=sitk.ReadImage(mask)
        name=get_name_wo_suffix(mask)

        id,slice=name.split('_')[2],name.split('_')[4]
        lab_name='myops_training_%s_gd_%s.nii.gz'%(id,slice)
        sitk_lab = sitk.ReadImage(tobe_masked_lab + "/" + lab_name)
        sitk_write_image(sitk_mask, dir=output_lab_dir,name=get_name_wo_suffix(mask))

        for type in ['C0','DE','T2']:
            img_name='myops_training_%s_%s_%s.nii.gz'%(id,type,slice)
            sitk_img=sitk.ReadImage(tobe_masked_img+"/"+img_name)
            sitk_write_image(sitk_img * recast_pixel_val(sitk_img,sitk_mask),dir=output_img_dir, name=get_name_wo_suffix(img_name))















