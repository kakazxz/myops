import SimpleITK as sitk
from dirutil.helper import sort_glob
import os
import numpy as np
from medpy.metric.binary import  dc,hd,asd
from tool.mask import create_mask
from cascade_multiseqseg.tool import reindex_for_myo_scar_edema_ZHANGZHEN
def evaluate():
    pred_images=sort_glob("../outputs/current_result/pred/*.nii.gz")
    gt_images=sort_glob("../outputs/current_result/gd/*.nii.gz")
    preds=[]
    gt=[]
    indexes=[2221]
    result = cal_dice(gt_images, indexes, pred_images)
    print("mean:"+str(result.mean()))
    print("std:"+str(result.std()))
    print("==================")
    indexes=[2221,1220]
    result = cal_dice(gt_images, indexes, pred_images)
    print("mean:"+str(result.mean()))
    print("std:"+str(result.std()))


def evaluateV2():
    gt_images=sort_glob("../datasets/cascasdePathology/valid_lab/*.nii.gz")
    pred_images=sort_glob("../outputs/cascadeMyoPathology/valid/*.nii.gz")

    indexes=[2221]
    result = cal_diceV2(gt_images, indexes, pred_images)
    print("mean:"+str(result.mean()))
    print("std:"+str(result.std()))
    print("==================")
    indexes=[2221,1220]
    result = cal_diceV2(gt_images, indexes, pred_images)
    print("mean:"+str(result.mean()))
    print("std:"+str(result.std()))



def cal_diceV2(gt_images, indexes, pred_images):
    result = []
    for p_pre, p_gt in zip(pred_images, gt_images):
        img = sitk.ReadImage(p_pre)
        img = sitk.GetArrayFromImage(img)
        img = create_mask(img, indexes)

        gt_img = sitk.ReadImage(p_gt)
        gt_img = sitk.GetArrayFromImage(gt_img)
        gt_img = create_mask(gt_img, indexes)

        result.append(dc(img, gt_img))
        print(dc(img, gt_img))
    result = np.array(result, dtype=np.float32)
    return result

def cal_dice(gt_images, indexes, pred_images):
    result = []
    for p_pre, p_gt in zip(pred_images, gt_images):
        img = sitk.ReadImage(p_pre)
        img = sitk.GetArrayFromImage(img)
        img = reindex_for_myo_scar_edema_ZHANGZHEN(img)

        img = create_mask(img, indexes)

        gt_img = sitk.ReadImage(p_gt)
        gt_img = sitk.GetArrayFromImage(gt_img)

        gt_img = create_mask(gt_img, indexes)

        result.append(dc(img, gt_img))
        print(dc(img, gt_img))
    result = np.array(result, dtype=np.float32)
    return result