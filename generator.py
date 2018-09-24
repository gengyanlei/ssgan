'''
  the author is leilei
'''

from torch import nn

class Generator(nn.Module):
    def __init__(self,class_number):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(10*10,768*16*16),nn.ReLU(inplace=True))
        # reshape
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(768,384,3,2,1,1),
                                     nn.BatchNorm2d(384),nn.ReLU(inplace=True))#32*32
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(384,256,3,2,1,1),
                                     nn.BatchNorm2d(256),nn.ReLU(inplace=True))#64*64
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256,192,3,2,1,1),
                                     nn.BatchNorm2d(192),nn.ReLU(inplace=True))#128*128
        # last layer no relu
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(192,class_number,3,2,1,1),nn.Tanh())#256*256
        
    def forward(self,x):
        x = self.linear(x)
        x = x.reshape([-1,768,16,16])
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        
        return x


def generator1(class_number):
    model = Generator(class_number)
    return model

######################################################################
class Generator(nn.Module):
    def __init__(self,class_number):
        super().__init__()
        # input [N,50*50] 由于全连接层 4096*4096 就很大了，因此这里不能设置那么大
        self.linear = nn.Sequential(nn.Linear(50*50,64*16*16),nn.ReLU(inplace=True))
        # reshape
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64,128,3,2,1,1),
                                     nn.BatchNorm2d(128),nn.ReLU(inplace=True))#32*32
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128,256,3,2,1,1),
                                     nn.BatchNorm2d(256),nn.ReLU(inplace=True))#64*64
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256,128,3,2,1,1),
                                     nn.BatchNorm2d(128),nn.ReLU(inplace=True))#128*128
        # last layer no relu
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(128,class_number,3,2,1,1),nn.Tanh())#256*256
        
    def forward(self,x):
        x = self.linear(x)
        x = x.reshape([-1,64,16,16])
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        
        return x


def generator(class_number):
    model = Generator(class_number)
    return model
  
  
