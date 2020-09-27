#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 0010 15:20
# @Author  : Kaka
# @File    : model.py
# @Software: PyCharm

import torch
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.relu(z)
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

def encoder_block(in_dim, out_dim, act_fn):
    model=nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim), act_fn,
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim), act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
    return model

def decoder_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),act_fn,
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim), act_fn,
            # nn.functional.interpolate(x, scale_factor=2)
            nn.Upsample(scale_factor=2, mode='nearest')
    )
    return model

class CAFB3(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(CAFB3, self).__init__()
        self.layer = nn.Sequential(SE(in_dim*3),nn.Conv2d(in_dim*3, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_dim), act_fn, )
    def forward(self, x1, x2, x3):
        modal_cat = torch.cat((x1, x2, x3), dim=1)
        out_fusion=self.layer(modal_cat)
        return out_fusion

class CAFB4(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(CAFB4, self).__init__()
        self.layer = nn.Sequential(SE(in_dim*4),nn.Conv2d(in_dim*4, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_dim), act_fn, )
    def forward(self, x1, x2, x3,xx):
        modal_cat = torch.cat((x1, x2, x3,xx), dim=1)
        out_fusion=self.layer(modal_cat)
        return out_fusion

class PRSN(nn.Module):

    def __init__(self, channel, nclass, ngf):
        super(PRSN, self).__init__()

        self.in_dim = channel
        self.out_dim = ngf
        self.final_out_dim = nclass

        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn2 = nn.ReLU(inplace=True)  # nn.ReLU()

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1)

        #######################################################################
        # Modality 1
        #######################################################################
        self.m1_encoder1 = encoder_block(self.in_dim, self.out_dim, act_fn)
        self.m1_encoder2 = encoder_block(self.out_dim, self.out_dim*2, act_fn)
        self.m1_encoder3 = encoder_block(self.out_dim*2, self.out_dim * 4, act_fn)

        self.m1_decoder1 = decoder_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.m1_decoder2 = decoder_block(self.out_dim * 4, self.out_dim * 2, act_fn2)
        self.m1_decoder3 = decoder_block(self.out_dim * 2, self.out_dim * 1, act_fn2)

        self.out1 = nn.Sequential(nn.Conv2d(int(self.out_dim), 2, kernel_size=3, stride=1, padding=1),
                                  nn.Softmax(dim=1))


        #######################################################################
        # Modality 2
        #######################################################################
        self.m2_encoder1 = encoder_block(self.in_dim, self.out_dim, act_fn)
        self.m2_encoder2 = encoder_block(self.out_dim, self.out_dim*2, act_fn)
        self.m2_encoder3 = encoder_block(self.out_dim*2, self.out_dim * 4, act_fn)

        self.m2_decoder1 = decoder_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.m2_decoder2 = decoder_block(self.out_dim * 4, self.out_dim * 2, act_fn2)
        self.m2_decoder3 = decoder_block(self.out_dim * 2, self.out_dim * 1, act_fn2)

        self.out2 = nn.Sequential(nn.Conv2d(int(self.out_dim), 3, kernel_size=3, stride=1, padding=1),
                                  nn.Softmax(dim=1))

        #######################################################################
        # Modality 3
        #######################################################################
        self.m3_encoder1 = encoder_block(self.in_dim, self.out_dim, act_fn)
        self.m3_encoder2 = encoder_block(self.out_dim, self.out_dim*2, act_fn)
        self.m3_encoder3 = encoder_block(self.out_dim*2, self.out_dim * 4, act_fn)

        self.m3_decoder1 = decoder_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.m3_decoder2 = decoder_block(self.out_dim * 4, self.out_dim * 2, act_fn2)
        self.m3_decoder3 = decoder_block(self.out_dim * 2, self.out_dim * 1, act_fn2)

        self.out3 = nn.Sequential(nn.Conv2d(int(self.out_dim), 3, kernel_size=3, stride=1, padding=1),
                                  nn.Softmax(dim=1))
        #######################################################################
        # fusion layer
        #######################################################################

        # self.fu_encoder1 = encoder_block(self.in_dim, self.out_dim, act_fn)
        self.fu1=CAFB3(self.out_dim, self.out_dim, act_fn)
        self.fu_encoder2 = encoder_block(self.out_dim, self.out_dim*2, act_fn)
        self.fu2 = CAFB4(self.out_dim*2, self.out_dim*2, act_fn)
        self.fu_encoder3 = encoder_block(self.out_dim*2, self.out_dim * 4, act_fn)
        self.fu3 = CAFB4(self.out_dim * 4, self.out_dim * 4,act_fn)

        self.fu_decoder1 = decoder_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.fu4 = CAFB4(self.out_dim * 4, self.out_dim * 4, act_fn)
        self.fu_decoder2 = decoder_block(self.out_dim * 4, self.out_dim * 2, act_fn2)
        self.fu5 = CAFB4(self.out_dim * 2, self.out_dim * 2, act_fn)
        self.fu_decoder3 = decoder_block(self.out_dim * 2, self.out_dim * 1, act_fn2)
        self.out =nn.Conv2d(int(self.out_dim), self.final_out_dim, kernel_size=3, stride=1, padding=1)# self.final_out_dim

    def forward(self, x1, x2, x3):
        # -----  m1 --------
        m1e1 = self.m1_encoder1(x1)
        m1e2 = self.m1_encoder2(m1e1)
        m1e3 = self.m1_encoder3(m1e2)

        m1d1 = self.m1_decoder1(m1e3)
        m1d2 = self.m1_decoder2(m1d1)
        m1d3 = self.m1_decoder3(m1d2)
        output1=self.out1(m1d3)


        # -----  m2 --------
        m2e1 = self.m2_encoder1(x2)
        m2e2 = self.m2_encoder2(m2e1)
        m2e3 = self.m2_encoder3(m2e2)

        m2d1 = self.m2_decoder1(m2e3)
        m2d2 = self.m2_decoder2(m2d1)
        m2d3 = self.m2_decoder3(m2d2)
        output2=self.out2(m2d3)

        # -----  m3 --------
        m3e1 = self.m3_encoder1(x3)
        m3e2 = self.m3_encoder2(m3e1)
        m3e3 = self.m3_encoder3(m3e2)

        m3d1 = self.m3_decoder1(m3e3)
        m3d2 = self.m3_decoder2(m3d1)
        m3d3 = self.m3_decoder3(m3d2)
        output3=self.out3(m3d3)

        # -----  fusion --------
        fu1=self.fu1(m1e1,m2e1,m3e1)
        fue2=self.fu_encoder2(fu1)
        fu2=self.fu2(m1e2,m2e2,m3e2,fue2)
        fue3 = self.fu_encoder3(fu2)
        fu3 = self.fu3(m1e3, m2e3, m3e3, fue3)

        fud1= self.fu_decoder1(fu3)
        fu4 = self.fu4(m1d1,m2d1,m3d1,fud1)
        fud2= self.fu_decoder2(fu4)
        fu5 = self.fu5(m1d2,m2d2,m3d2,fud2)
        fud3= self.fu_decoder3(fu5)
        output= self.out(fud3)


        return output, output1, output2,output3


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = PRSN(1, 3, 32)
    model = nn.DataParallel(model)
    model = model.to(device)
    x = torch.rand(4, 1, 128, 128)
    x = x.to(device)
    out, x1, x2, x3 = model.forward(x, x, x)
    print("sccuess")
    print(out.shape, x1.shape, x2.shape, x3.shape)
