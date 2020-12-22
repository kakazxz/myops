#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 0010 15:20
# @Author  : Kaka
# @File    : model.py
# @Software: PyCharm


import torch
import torch.nn as nn
from .scmodel import scSE
from .gcblock import  ContextBlock
class nodo(nn.Module):
    def __init__(self):
        super(nodo, self).__init__()
    def forward(self, input):
        return input


def up(x):
    return nn.functional.interpolate(x, scale_factor=2)


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def avgpool():
    pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    return pool

# def conv_decod_block(in_dim, out_dim, act_fn):
#     model = nn.Sequential(
#         nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm2d(out_dim),
#         act_fn,
#     )
#     return model

def conv_decod_block(in_dim, out_dim, act_fn,):
    model = nn.Sequential(
        nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim), act_fn,
        ContextBlock(inplanes=out_dim, ratio=1. / 16., pooling_type='att'),
        nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),# nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim), act_fn,

    )
    return model


def nonloacl_decod_block(chanel, act_fn):
    model = nn.Sequential(
        NonLocalBlock(chanel),
        nn.BatchNorm2d(chanel),
        act_fn,
    )
    return model


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.std_pool = nn.AdaptiveStdPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MixedFusion_Block3(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(MixedFusion_Block3, self).__init__()
        self.layer = nn.Sequential(ContextBlock(inplanes=in_dim*3, ratio=1. / 16., pooling_type='att'),nn.Conv2d(in_dim*3, out_dim, kernel_size=3, stride=1, padding=1),#nn.Dropout(p=0.5),
                                    nn.BatchNorm2d(out_dim), act_fn, )
    def forward(self, x1, x2, x3):
        modal_cat = torch.cat((x1, x2, x3), dim=1)
        out_fusion=self.layer(modal_cat)
        return out_fusion


class MixedFusion_Block4(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(MixedFusion_Block4, self).__init__()
        self.layer = nn.Sequential(ContextBlock(inplanes=in_dim*4, ratio=1. / 16., pooling_type='att'),nn.Conv2d(in_dim*4, out_dim, kernel_size=3, stride=1, padding=1),#nn.Dropout(p=0.5),NonLocalBlock(in_dim*4),
                                    nn.BatchNorm2d(out_dim), act_fn, )
    def forward(self, x1, x2, x3,xx):
        modal_cat = torch.cat((x1, x2, x3,xx), dim=1)
        out_fusion=self.layer(modal_cat)
        return out_fusion

class MixedFusion_BlockS4(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(MixedFusion_BlockS4, self).__init__()
        # self.layer = nn.Sequential(ContextBlock(inplanes=in_dim*4, ratio=1. / 16., pooling_type='att'),nn.Conv2d(in_dim*4, out_dim, kernel_size=3, stride=1, padding=1),#nn.Dropout(p=0.5),NonLocalBlock(in_dim*4),
        #                             nn.BatchNorm2d(out_dim), act_fn, )
        self.layer0 = nn.Sequential(ContextBlock(inplanes=4, ratio=1. / 4., pooling_type='att'),
                                   nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1), act_fn, )
        self.layer1 = nn.Sequential(ContextBlock(inplanes=2, ratio=1. / 2., pooling_type='att'),
                                   nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1), act_fn, )
        self.layer2 = nn.Sequential(ContextBlock(inplanes=4, ratio=1. / 4., pooling_type='att'),
                                   nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1), act_fn, )
        self.layer3 = nn.Sequential(ContextBlock(inplanes=3, ratio=1. / 3., pooling_type='att'),
                                   nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1), act_fn, )
    def forward(self, x1, x2, x3,xx):
        # print(x1.shape,x2.shape,x3.shape,xx.shape,)
        cat0 = torch.cat((x1[:, 0, :, :].unsqueeze(dim=0), x2[:, 0, :, :].unsqueeze(dim=0), x3[:, 0, :, :].unsqueeze(dim=0), xx[:, 0, :, :].unsqueeze(dim=0)), dim=1)
        cat1 =torch.cat((x1[:,1,:,:].unsqueeze(dim=0),xx[:,1,:,:].unsqueeze(dim=0)), dim=1)
        cat2 = torch.cat((x1[:, 1, :, :].unsqueeze(dim=0), x2[:, 2, :, :].unsqueeze(dim=0),x3[:, 2, :, :].unsqueeze(dim=0),xx[:, 2, :, :].unsqueeze(dim=0)), dim=1)
        cat3 = torch.cat((x1[:, 1, :, :].unsqueeze(dim=0), x3[:, 2, :, :].unsqueeze(dim=0), xx[:, 3, :, :].unsqueeze(dim=0)), dim=1)
        # print(cat0.shape, cat1.shape, cat2.shape, cat3.shape, )
        cat0 = self.layer0(cat0)
        cat1 = self.layer1(cat1)
        cat2 = self.layer2(cat2)
        cat3 = self.layer3(cat3)
        modal_cat = torch.cat((cat0, cat1, cat2,cat3), dim=1)
        return modal_cat

# class MixedFusion_Block5(nn.Module):
#     def __init__(self, in_dim, out_dim, act_fn):
#         super(MixedFusion_Block5, self).__init__()
#         self.layer = nn.Sequential(ContextBlock(inplanes=in_dim*5, ratio=1. / 16., pooling_type='att'),nn.Conv2d(in_dim*5, out_dim, kernel_size=3, stride=1, padding=1),#nn.Dropout(p=0.5),NonLocalBlock(in_dim*4),
#                                     nn.BatchNorm2d(out_dim), act_fn, )
#     def forward(self, x1, x2, x3,xx1,xx2):
#         modal_cat = torch.cat((x1, x2, x3,xx1,xx2), dim=1)
#         out_fusion=self.layer(modal_cat)
#         return out_fusion


class PRSN4(nn.Module):

    def __init__(self, input_nc, output_nc, ngf):
        super(PRSN4, self).__init__()

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        act_fn = nn.ReLU(inplace=True)  # act_fn = nn.LeakyReLU(0.2, inplace=True)#

        act_fn2 = nn.ReLU(inplace=True)  # nn.ReLU()
        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1)

        #######################################################################
        # Encoder **Modality 1
        #######################################################################
        self.down_1_1 = conv_decod_block(self.in_dim,self.out_dim,act_fn)
        self.pool_1_1 = maxpool()

        self.down_1_2 = conv_decod_block(self.out_dim,self.out_dim*2,act_fn)
        self.pool_1_2 = maxpool()

        self.down_1_3 = conv_decod_block(self.out_dim * 2,self.out_dim*4,act_fn)
        self.pool_1_3 = maxpool()


        #######################################################################
        # Encoder **Modality 2
        #######################################################################
        self.down_2_1 = conv_decod_block(self.in_dim, self.out_dim, act_fn)
        self.pool_2_1 = maxpool()

        self.down_2_2 = conv_decod_block(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2_2 = maxpool()

        self.down_2_3 = conv_decod_block(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_2_3 = maxpool()

        #######################################################################
        # Encoder **Modality 3
        #######################################################################
        self.down_3_1 = conv_decod_block(self.in_dim*2, self.out_dim, act_fn)
        self.pool_3_1 = maxpool()

        self.down_3_2 = conv_decod_block(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_3_2 = maxpool()

        self.down_3_3 = conv_decod_block(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3_3 = maxpool()

        #######################################################################
        # fusion layer
        #######################################################################
        #
        # self.fu1 = MixedFusion_Block3(self.in_dim, self.out_dim, act_fn)
        # self.down_fu_1 = conv_decod_block(self.out_dim, self.out_dim, act_fn)
        # self.pool_fu_1 = maxpool()

        self.down_fu_1 = conv_decod_block(self.in_dim*3, self.out_dim, act_fn)
        self.pool_fu_1 = maxpool()

        self.fu2 = MixedFusion_Block4(self.out_dim, self.out_dim, act_fn)#MixedFusion_Block3(self.out_dim, self.out_dim, act_fn)
        self.down_fu_2 = conv_decod_block(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_fu_2 = maxpool()

        self.fu3 = MixedFusion_Block4(self.out_dim*2, self.out_dim*2, act_fn)
        self.down_fu_3 = conv_decod_block(self.out_dim* 2, self.out_dim * 4, act_fn)
        self.pool_fu_3 = maxpool()

        # ~~~ Decoding Path ~~~~~~ #
        self.fu4 = MixedFusion_Block4(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.deconv_fu_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)

        self.fu5 = MixedFusion_Block4(self.out_dim * 2, self.out_dim * 2, act_fn2)
        self.deconv_fu_2 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn2)

        self.fu6 = MixedFusion_Block4(self.out_dim * 1, self.out_dim * 1, act_fn2)
        self.deconv_fu_3 = conv_decod_block(self.out_dim * 1, int(self.out_dim), act_fn2)
        self.out = nn.Conv2d(int(self.out_dim), self.final_out_dim, kernel_size=3, stride=1, padding=1)


        # Modality 1

        self.deconv_1_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)
        self.deconv_1_2 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn2)
        self.deconv_1_3 = conv_decod_block(self.out_dim * 1, int(self.out_dim), act_fn2)
        self.out1 = nn.Conv2d(int(self.out_dim), 2, kernel_size=3, stride=1, padding=1)

        # modality 2
        self.deconv_2_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)
        self.deconv_2_2 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn2)
        self.deconv_2_3 = conv_decod_block(self.out_dim * 1, int(self.out_dim), act_fn2)
        self.out2 = nn.Conv2d(int(self.out_dim), 2, kernel_size=3, stride=1, padding=1)#3


        # modality 3
        self.deconv_3_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)
        self.deconv_3_2 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn2)
        self.deconv_3_3 = conv_decod_block(self.out_dim * 1, int(self.out_dim), act_fn2)
        self.out3 = nn.Conv2d(int(self.out_dim), 2, kernel_size=3, stride=1, padding=1)#3

    def forward(self, x1,x2 , x3):
        # ############################# #
        i1 = x1
        i2 = x2
        i3 = torch.cat((x2,x3),dim=1)
        ifu = torch.cat((x1,x2, x3), dim=1)

        # -----  modality 1 --------
        down_1_1 = self.down_1_1(i1)
        down_1_1 = self.pool_1_1(down_1_1)
        down_1_2 = self.down_1_2(down_1_1)
        down_1_2 = self.pool_1_2(down_1_2)
        down_1_3 = self.down_1_3(down_1_2)
        down_1_3 = self.pool_1_3(down_1_3)

        deconv_1_1 = self.deconv_1_1((down_1_3))
        deconv_1_1 = up(deconv_1_1)
        deconv_1_2 = self.deconv_1_2(deconv_1_1)
        deconv_1_2 = up(deconv_1_2)
        deconv_1_3 = self.deconv_1_3(deconv_1_2)
        deconv_1_3 = up(deconv_1_3)
        output1 = self.out1(deconv_1_3)
        output1 = nn.Softmax(1)(output1)

        # -----  modality 2 --------
        down_2_1 = self.down_2_1(i2)
        down_2_1 = self.pool_2_1(down_2_1)
        down_2_2 = self.down_2_2(down_2_1)
        down_2_2 = self.pool_2_2(down_2_2)
        down_2_3 = self.down_2_3(down_2_2)
        down_2_3 = self.pool_2_3(down_2_3)

        deconv_2_1 = self.deconv_2_1((down_2_3))
        deconv_2_1 = up(deconv_2_1)
        deconv_2_2 = self.deconv_2_2(deconv_2_1)
        deconv_2_2 = up(deconv_2_2)
        deconv_2_3 = self.deconv_2_3(deconv_2_2)
        deconv_2_3 = up(deconv_2_3)
        output2 = self.out2(deconv_2_3)
        output2 = nn.Softmax(1)(output2)
        # -----  modality 3 --------
        down_3_1 = self.down_3_1(i3)
        down_3_1 = self.pool_3_1(down_3_1)
        down_3_2 = self.down_3_2(down_3_1)
        down_3_2 = self.pool_3_2(down_3_2)
        down_3_3 = self.down_3_3(down_3_2)
        down_3_3 = self.pool_3_3(down_3_3)

        deconv_3_1 = self.deconv_3_1((down_3_3))
        deconv_3_1 = up(deconv_3_1)
        deconv_3_2 = self.deconv_3_2(deconv_3_1)
        deconv_3_2 = up(deconv_3_2)
        deconv_3_3 = self.deconv_3_3(deconv_3_2)
        deconv_3_3 = up(deconv_3_3)
        output3 = self.out3(deconv_3_3)
        output3 = nn.Softmax(1)(output3)
        # -----  fusion --------
        # down_fu_1 = self.down_fu_1(self.fu1(i1,i2,i3))
        # down_fu_1 = self.pool_fu_1(down_fu_1)
        down_fu_1 = self.down_fu_1(ifu)
        down_fu_1 = self.pool_fu_1(down_fu_1)
        down_fu_2 = self.down_fu_2(self.fu2(down_1_1,down_2_1,down_2_1,down_fu_1))
        down_fu_2 = self.pool_fu_2(down_fu_2)
        # print(down_1_2.shape,down_2_2.shape,down_3_2.shape,down_fu_2.shape)
        down_fu_3 = self.down_fu_3(self.fu3(down_1_2,down_2_2,down_3_2,down_fu_2))

        down_fu_3 = self.pool_fu_3(down_fu_3)

        deconv_fu_1 = self.deconv_fu_1(self.fu4(down_1_3,down_2_3,down_3_3,down_fu_3))
        deconv_fu_1 = up(deconv_fu_1)
        # deconv_fu_2 = self.deconv_fu_2(self.fu5(deconv_1_1,deconv_2_1,deconv_3_1,deconv_fu_1))
        deconv_fu_2 = self.deconv_fu_2(self.fu5(down_1_2, down_2_2, down_3_2, deconv_fu_1))
        deconv_fu_2 = up(deconv_fu_2)
        # deconv_fu_3 = self.deconv_fu_3(self.fu6(deconv_1_2,deconv_2_2,deconv_3_2,deconv_fu_2))
        deconv_fu_3 = self.deconv_fu_3(self.fu6(down_1_1, down_2_1, down_2_1, deconv_fu_2))
        deconv_fu_3 = up(deconv_fu_3)
        output = self.out(deconv_fu_3)
        output = nn.Softmax(1)(output)
        return output, output1, output2,output3