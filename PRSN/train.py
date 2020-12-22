from dataset.MDataset import MultiModalityData_load
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

import config
from models.prsn import PRSN4 as PRSN
from utils import init_util, metrics,common,logger


def val(model, val_loader, epoch, logger):
    model.eval()
    val_loss = 0
    val_dice0 = 0
    val_dice1 = 0
    val_dice2 = 0
    val_dice3 = 0
    with torch.no_grad():
        for C0, DE,T2,mask1,mask2,mask3,mask in val_loader:
            C0, DE, T2, mask1, mask2, mask3, mask = C0.float().to(device), DE.float().to(device), T2.float().to(device), mask1.float().to(device), mask2.float().to(device), mask3.float().to(device), mask.float().to(device)

            output, output1, output2,output3 = model(C0, DE,T2)
            loss = metrics.DiceMeanLoss()(output, mask)
            dice0 = metrics.dice(output, mask, 0)
            dice1 = metrics.dice(output, mask, 1)
            dice2 = metrics.dice(output, mask, 2)
            dice3 = metrics.dice(output, mask, 3)
            val_loss += float(loss)
            val_dice0 += float(dice0)
            val_dice1 += float(dice1)
            val_dice2 += float(dice2)
            val_dice3 += float(dice3)

    val_loss /= len(val_loader)
    val_dice0 /= len(val_loader)
    val_dice1 /= len(val_loader)
    val_dice2 /= len(val_loader)
    val_dice3 /= len(val_loader)
    logger.scalar_summary('valid/val_loss', val_loss, epoch)
    logger.scalar_summary('valid/val_dice0', val_dice0, epoch)
    logger.scalar_summary('valid/val_dice1', val_dice1, epoch)
    logger.scalar_summary('valid/val_dice2', val_dice2, epoch)
    logger.scalar_summary('valid/val_dice3', val_dice3, epoch)
    print('\nVal set: Average loss: {:.6f}, dice0: {:.6f}\tdice1: {:.6f}\tdice2: {:.6f}\tdice3: {:.6f}\t\n'.format(
        val_loss, val_dice0, val_dice1, val_dice2,val_dice3))


def train(model, train_loader, optimizer, epoch, logger):
    model.train()
    train_loss = 0
    train_dice0 = 0
    train_dice1 = 0
    train_dice2 = 0
    train_dice3 = 0
    for batch_idx, (C0, DE,T2,mask1,mask2,mask3,mask) in enumerate(train_loader):
        # data, target = data.float(), target.float()
        C0, DE,T2,mask1,mask2,mask3,mask = C0.float().to(device), DE.float().to(device), T2.float().to(device), mask1.float().to(device), mask2.float().to(device), mask3.float().to(device), mask.float().to(device)
        # print(t1ce_img.shape)
        output, output1, output2,output3 = model(C0, DE,T2)

        optimizer.zero_grad()

        # loss = nn.CrossEntropyLoss()(output,target)
        # loss=metrics.SoftDiceLoss()(output,target)
        # loss=nn.MSELoss()(output,target)
        loss1 = metrics.DiceMeanLoss()(output1, mask1)
        loss2 = metrics.DiceMeanLoss()(output2, mask2)
        loss3 = metrics.DiceMeanLoss()(output3, mask3)
        loss = metrics.DiceMeanLoss()(output, mask)
        loss_all =loss+0.2*loss1+0.3*loss2+0.3*loss3
        # loss=metrics.WeightDiceLoss()(output,target)
        # loss=metrics.CrossEntropy()(output,target)
        loss_all.backward()
        optimizer.step()

        train_loss = loss
        train_dice0 = metrics.dice(output, mask, 0)
        train_dice1 = metrics.dice(output, mask, 1)
        train_dice2 = metrics.dice(output, mask, 2)
        train_dice3 = metrics.dice(output, mask, 3)
        print(
            'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tdice0: {:.6f}\tdice1: {:.6f}\tdice2: {:.6f}\tT: {:.6f}\tP: {:.6f}\tTP: {:.6f}'.format(
                epoch, batch_idx+1, len(train_loader),
                100. * (batch_idx+1) / len(train_loader), loss.item(),
                train_dice0, train_dice1, train_dice2,
                metrics.T(output, mask), metrics.P(output, mask), metrics.TP(output, mask)))

    logger.scalar_summary('train/train_loss', float(train_loss), epoch)
    logger.scalar_summary('train/train_dice0', float(train_dice0), epoch)
    logger.scalar_summary('train/train_dice1', float(train_dice1), epoch)
    logger.scalar_summary('train/train_dice2', float(train_dice2), epoch)
    logger.scalar_summary('train/train_dice3', float(train_dice3), epoch)


if __name__ == '__main__':
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_set = MultiModalityData_load(args.dataset_path)#Lits_DataSet(args.crop_size, args.resize_scale, args.dataset_path, mode='train')
    val_set = MultiModalityData_load(args.valid_path)##Lits_DataSet(args.crop_size, args.resize_scale, args.valid_path, mode='val')
    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,num_workers=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set,batch_size=args.batch_size,num_workers=1, shuffle=True)
    # model info
    # model = UNet(1, [32, 48, 64, 96, 128], 4, net_mode='3d',conv_block=RecombinationBlock).to(device)
    model =PRSN(1, 4, 32).to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.99))#optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    init_util.print_network(model)
    # model = nn.DataParallel(model, device_ids=[0,1])  # multi-GPU

    logger = logger.Logger('./output/{}'.format(args.save))
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train(model, train_loader, optimizer, epoch, logger)
        val(model, val_loader, epoch, logger)
        torch.save(model, './output/{}/state{}.pkl'.format(args.save,epoch))  # Save model with parameters
        # torch.save(model.state_dict(), './output/{}/param.pkl'.format(args.save))  # Only save parameters