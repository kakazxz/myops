import glob
import SimpleITK as sitk
from preprocessor.tools import crop_by_bbox,get_bounding_box_by_ids,sitkResize3DV2,sitkResample3DV2
import os
from dirutil.helper import sort_glob
import numpy as np
from preprocessor.tools import binarize_img
from sitkImageIO.itkdatawriter import sitk_write_image,write_png_image,write_png_lab
from dirutil.helper import mkdir_if_not_exist
from preprocessor.tools import reindex_label
from dirutil.helper import mk_or_cleardir

def prepare_data(config):

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

def prepare_test_data(config):

    files = sort_glob("../../dataset/myops/crop_20/*.nii.gz")
    output_img_dir = "%s/challenge_test_img" % (config.dataset_dir)
    prepare_test_slice(config,files,output_img_dir)

def prepare_slices(config,files,output_lab_dir,output_img_dir):
    mk_or_cleardir(output_lab_dir)
    mk_or_cleardir(output_img_dir)
    for p in files:
        lab = sitk.ReadImage(p)
        #先转化成统一space,保证crop的大小一致
        # lab=sitkResample3DV2(lab,sitk.sitkNearestNeighbor,[1,1,1])
        ids=[int(item) for item in config.components.split(',')]
        bbox=get_bounding_box_by_ids(lab,padding=10,ids=ids)
        ##extend bbox
        crop_lab=crop_by_bbox(lab,bbox)
        crop_lab=binarize_img(crop_lab,ids)
        crop_lab=sitkResize3DV2(crop_lab,[config.fine_size,config.fine_size,crop_lab.GetSize()[-1]],sitk.sitkNearestNeighbor)
        for i in range(crop_lab.GetSize()[-1]):
            sitk_write_image(crop_lab[:,:,i],dir=output_lab_dir,name="%s_%d"%(os.path.basename(p).split('.')[0],i))
            img_file=glob.glob("../../dataset/myops/train25_convert/*%s*.nii.gz"%(os.path.basename(p).split("_")[2]))
            img_file.sort()
            for j in img_file:
                img = sitk.ReadImage(j)
                # img = sitkResample3DV2(img, sitk.sitkLinear, [1, 1, 1])
                crop_img=crop_by_bbox(img,bbox)

                crop_img = sitkResize3DV2(crop_img, [config.fine_size, config.fine_size,crop_img.GetSize()[-1]], sitk.sitkLinear)
                sitk_write_image(crop_img[:,:,i], dir=output_img_dir, name="%s_%d"%(os.path.basename(j).split('.')[0],i))

def prepare_test_slice(config,files,output_img_dir):
    mk_or_cleardir(output_img_dir)
    for p in files:
        img = sitk.ReadImage(p)
        rez_img = sitkResize3DV2(img, [config.fine_size,config.fine_size, img.GetSize()[-1]], sitk.sitkLinear)
        for i in range(rez_img.GetSize()[-1]):
            sitk_write_image(rez_img[:, :, i], dir=output_img_dir,
                             name="%s_%d" % (os.path.basename(p).split('.')[0], i))

# def slice_png_data(args,modality='DE',tag='A',):
#     '''
#     :return:
#     '''
#     files = glob.glob("../../dataset/myops/train25_myops_gd_convert/*.nii.gz")
#     files.sort()
#     train_img_dir="%s/train%s/"%(args.dataset,tag)
#     train_lab_dir="%s/train%s_lab/"%(args.dataset,tag)
#     test_img_dir="%s/test%s/"%(args.dataset,tag)
#     test_lab_dir="%s/test%s_lab/"%(args.dataset,tag)
#
#     method_name(files[:20], modality, train_img_dir, train_lab_dir)
#     method_name(files[-5:], modality, test_img_dir, test_lab_dir)


# def method_name(files, modality, img_dir, lab_dir):
#     for i in files:
#         lab = sitk.ReadImage(i)
#         # 先转化成统一space,保证crop的大小一致
#         lab = sitkResample3DV2(lab, sitk.sitkNearestNeighbor, [1, 1, 1])
#         bbox = get_bounding_boxV2(sitk.GetArrayFromImage(lab), padding=10)
#         ##extend bbox
#         crop_lab = crop_by_bbox(lab, bbox)
#         crop_lab = sitkResize3DV2(crop_lab, [256, 256, crop_lab.GetSize()[-1]], sitk.sitkNearestNeighbor)
#
#         write_png_lab(crop_lab[:, :, crop_lab.GetSize()[-1] // 2], dir=lab_dir, name=os.path.basename(i))
#         img_file = glob.glob(
#             "../../dataset/myops/train25_convert/*%s*%s.nii.gz" % (os.path.basename(i).split("_")[2], modality))
#         for j in img_file:
#             img = sitk.ReadImage(j)
#             img = sitkResample3DV2(img, sitk.sitkLinear, [1, 1, 1])
#             crop_img = crop_by_bbox(img, bbox)
#             crop_img = sitkResize3DV2(crop_img, [256, 256, crop_img.GetSize()[-1]], sitk.sitkLinear)
#             crop_img = sitk.RescaleIntensity(crop_img, 0, 255)
#             write_png_image(crop_img[:, :, crop_lab.GetSize()[-1] // 2], dir=img_dir, name=os.path.basename(j))
