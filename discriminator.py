'''
  the author is leilei
  you have so many choices:u_net ,deeplab_v3 ...;here we choose u_net 
'''

import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F

'''
base vgg11 fine-tune U_Net : ternausNet 直接借鉴一下，本地使用的基于vgg16的u_net.
'''

class DecoderBlock(nn.Module):
    def __init__(self,in_,middle_,out_):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_,middle_,3,1,1),
            nn.ReLU(inplace=True),
            # ConvTranspose2d need add relu and inplace=True and out_padding=1 and upsample need't relu
            nn.ConvTranspose2d(middle_,out_,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(inplace=True))
    def forward(self,x):
        return self.block(x)
    
class ConvRelu(nn.Module):
    def __init__(self,in_,out_):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_,out_,3,1,1),
            nn.ReLU(inplace=True))
    def forward(self,x):
        return self.conv(x)
        
class U_Net(nn.Module):
    def __init__(self,num_filters=32,class_number=5):
        super().__init__()
        # pytorch create net,in __init__ not use self if pretrained net,just use it to transfer we needed layers
        # otherwise,it appear in state_dict
        encoder=torchvision.models.vgg11(pretrained=True).features# if exist load quickly
        self.pool=nn.MaxPool2d(2,2)
        
        self.conv1=encoder[0:2] #64 256*256
        self.conv2=encoder[3:5] #128 128*128
        self.conv3=encoder[6:10] #256 64*64
        self.conv4=encoder[11:15] #512 32*32
        self.conv5=encoder[16:20] #512 16*16
        # 5 convtranspose
        self.center=DecoderBlock(num_filters*16,num_filters*16,num_filters*8)#8*8 -> 16*16
        
        self.dec5=DecoderBlock(num_filters*(16+8),num_filters*16,num_filters*8)
        self.dec4=DecoderBlock(num_filters*(16+8),num_filters*16,num_filters*4)
        self.dec3=DecoderBlock(num_filters*(8+4),num_filters*8,num_filters*2)
        self.dec2=DecoderBlock(num_filters*(4+2),num_filters*4,num_filters)
        
        self.dec1=ConvRelu(num_filters*(2+1),num_filters*(2+1))# modified
        
        self.score=nn.Conv2d(num_filters*(2+1),class_number,1,1)
        
    def forward(self,x):
        conv1=self.conv1(x)
        conv2=self.conv2(self.pool(conv1))
        conv3=self.conv3(self.pool(conv2))
        conv4=self.conv4(self.pool(conv3))
        conv5=self.conv5(self.pool(conv4))
        
        center=self.center(self.pool(conv5))
        
        dec5=self.dec5(torch.cat([center,conv5],dim=1))
        dec4=self.dec4(torch.cat([dec5,conv4],dim=1))
        dec3=self.dec3(torch.cat([dec4,conv3],dim=1))
        dec2=self.dec2(torch.cat([dec3,conv2],dim=1))
        dec1=self.dec1(torch.cat([dec2,conv1],dim=1))
        
        score=self.score(dec1)
        return score
        
def u_net(class_number=5):
    model=U_Net(class_number=class_number)
    return model

