#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/11 0011 21:58
# @Author  : Kaka
# @File    : restore.py
# @Software: PyCharm
import os
import nibabel as nib
import numpy as np
import skimage.transform as t
def filter_data(dirname, filter="gd"):
    result = []  # 含有filter的所有的文件
    for maindir, subdir, file_name_list in os.walk(dirname):

        # print("1:",maindir) #当前主目录
        # print("2:",subdir) #当前主目录下的所有目录
        # print("3:",file_name_list)  #当前主目录下的所有文件

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            # ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容
            ext = apath.split("_")
            if filter in ext:
                result.append(apath)
    return result

def restore(origin,crop,slicer,savepath,st=201,ed=220):
    """
    姜切片还原为原始数据的尺寸
    :param origin: 原始体数据路径
    :param crop: 裁剪后的图像路径
    :param slicer: 切片路径
    :param savepath:
    :return:
    """
    if not (os.path.exists(savepath)):
        os.makedirs(savepath)
    for i in range(st,ed+1):
        slist=filter_data(slicer,str(i))
        if len(slist) !=0:
            n=os.path.split(slist[0])
            file=n[-1].replace("_0","").replace("gd","C0")
            cropfile=os.path.join(crop,file)
            originfile=os.path.join(origin,file)
            savefile=os.path.join(savepath,file.replace("C0","gd"))
            cropnii=nib.load(cropfile)
            cropimg=cropnii.get_data()
            originnii=nib.load(originfile)
            originimg=originnii.get_data()
            assert len(slist) == cropimg.shape[-1],'切片数量：%d，维度%d不一致'%(len(slist),cropimg.shape[0])
            crop_mask=np.zeros_like(cropimg)
            crop_s_shape=(crop_mask.shape[0],crop_mask.shape[1])
            for j in range(len(slist)):
                slicer_img=nib.load(slist[j]).get_data().astype("float")
                slicer_img=t.resize(slicer_img,crop_s_shape,order=0).astype("uint16")
                print(np.max(slicer_img))
                crop_mask[:,:,j]=slicer_img
            # print(originnii.header)
            x=abs(originnii.header["qoffset_x"]/originnii.header["pixdim"][1]-cropnii.header["qoffset_x"]/cropnii.header["pixdim"][1])
            y = abs(originnii.header["qoffset_y"]/originnii.header["pixdim"][2] - cropnii.header["qoffset_y"]/cropnii.header["pixdim"][2])
            # print(x,y,x+crop_mask.shape[0],y+crop_mask.shape[1])
            sx=int(x);ex=int(x+crop_mask.shape[0]);
            sy = int(y);ey = int(y + crop_mask.shape[1]);
            origin_mask=np.zeros_like(originimg)
            origin_mask[sx:ex,sy:ey,:]=crop_mask
            array_img = nib.Nifti1Image(origin_mask, originnii.affine)
            nib.save(array_img, savefile)


# list=filter_data("./restore/slicer","gd")
# print(list)

