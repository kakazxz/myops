from dataset.MDataset import MultiModalityData_load
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
from models.prsn import PRSN4 as PRSN
from utils import logger, init_util, metrics,common
import numpy as np
import os
import nibabel as nib
from skimage import transform,exposure
def finame(gdname):
    gdname=gdname[0].split("_")
    return gdname[0]+'_'+gdname[1]+'_'+gdname[2]

def inference(model, test_loader,savepath):
    print("Evaluation of Testset Starting...")
    model.eval()
    val_loss = 0
    val_dice0 = 0
    val_dice1 = 0
    val_dice2 = 0
    val_dice3 = 0
    affine=np.array([[ -1.,-0.,-0. ,-0.],[ -0. ,-1. ,-0.,239.],[  0.  , 0. ,  1.,   0.],[  0. ,  0. ,  0.  , 1.]])
    with torch.no_grad():
        for img_C0, img_DE, img_T2, img_mask,x,y, preadname, gdname in tqdm(test_loader):
            img_C0, img_DE, img_T2, img_mask = img_C0.float().to(device), img_DE.float().to(device), img_T2.float().to(device), img_mask.float().to(device)
            output,output1,output2,output3 = model( img_C0, img_DE, img_T2)
            # print(output.shape)
            loss = metrics.DiceMeanLoss()(output, img_mask)
            dice0 = metrics.dice(output, img_mask, 0)
            dice1 = metrics.dice(output, img_mask, 1)
            dice2 = metrics.dice(output, img_mask, 2)
            dice3 = metrics.dice(output, img_mask, 3)

            val_loss += float(loss)
            val_dice0 += float(dice0)
            val_dice1 += float(dice1)
            val_dice2 += float(dice2)
            val_dice3 += float(dice3)
            for x1,y1,output1,gdname1 in zip(x,y,output,gdname):
                # print(x1,y1)
                output1 = output1.squeeze().cpu().detach().numpy()
                output1 = np.argmax(output1, axis=0).astype("float")
                output1=transform.resize(output1, [x1,y1], order=0)
                savedir = os.path.join(savepath, finame(gdname))
                if not (os.path.exists(savedir)):
                    os.makedirs(savedir)
                array_img = nib.Nifti1Image(output1,affine)
                nib.save(array_img, os.path.join(savedir, "%s" % gdname))

    val_loss /= len(test_loader)
    val_dice0 /= len(test_loader)
    val_dice1 /= len(test_loader)
    val_dice2 /= len(test_loader)
    val_dice3 /= len(test_loader)

    print('\nTest set: Average loss: {:.6f}, dice0: {:.6f}\tdice1: {:.6f}\tdice2: {:.6f}\tdice3: {:.6f}\t\n'.format(
        val_loss, val_dice0, val_dice1, val_dice2,val_dice3))

if __name__ == '__main__':
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    test_set = MultiModalityData_load(args.test_path,train=False, test=True,valid=False,)
    test_loader = DataLoader(dataset=test_set,batch_size=args.batch_size,num_workers=1, shuffle=False)
    # model info
    # model = UNet(1, [32, 48, 64, 96, 128], 3, net_mode='3d',conv_block=RecombinationBlock).to(device)
    # model = PRSN(1, 4, 32).to(device)
    # model.load_state_dict(torch.load('./output/{}/state4.pkl'.format(args.save)))
    model = torch.load('./output/{}/state{}.pkl'.format(args.save,args.stnum))#args.save))
    inference(model, test_loader,args.save_path)