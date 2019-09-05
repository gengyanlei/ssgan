'''
  the author is leilei
'''
# 开始写主函数

import os
import cv2
import torch
from torchvision import transforms
import numpy as np
from torch import nn
from torch.autograd import Variable # 可以不用
from torch.utils import data
# imread data
from data_imread import batch_data
# models
from generator import generator
from discriminator import discriminator
# losses
from losses import Loss_label, Loss_fake, Loss_unlabel

################### Hyper parameter ################### 
batch_size=16
class_number=5
lr_g=2e-4
lr_d=1e-4
power=0.9
weight_decay=5e-4
max_iter=20000
dataset_path=r'**/Dataset/hdf5/f5.hdf5'
dataset_nl_path=r'**/Dataset/hdf5/f2.hdf5'
save_path=r'**/Pytorch_Code/ALL/ssgan/'

#loss_s_path=os.path.join(save_path,'loss.npy')
model_s_path=os.path.join(save_path,'model.pth')
#loss_s_figure=os.path.join(save_path,'loss.tif')
model_g_spath=os.path.join(save_path,'g/model_g.pth')

################### update lr ###################
def lr_poly(base_lr,iters,max_iter,power):
    return base_lr*((1-float(iters)/max_iter)**power)
def adjust_lr(optimizer,base_lr,iters,max_iter,power):
    lr=lr_poly(base_lr,iters,max_iter,power)
    optimizer.param_groups[0]['lr']=lr
    if len(optimizer.param_groups)>1:
        optimizer.param_groups[1]['lr']=lr*10

################### dataset loader ###################
img_transform=transforms.ToTensor()#  hwc=>chw and 0-255=>0-1
dataset=batch_data.Data(dataset_path,transform=img_transform,augmentation=False)
trainloader=data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=3)

dataset_nl=batch_data.data(dataset_nl_path,transform=img_transform)
trainloader_nl=data.DataLoader(dataset_nl,batch_size=batch_size,shuffle=True,num_workers=3)

trainloader_iter=enumerate(trainloader)
trainloader_nl_iter=enumerate(trainloader_nl)

################### build model ###################
model_g = generator(3)
model_d = discriminator(class_number+1)

#### fine-tune ####
#new_params=model.state_dict()
#pretrain_dict=torch.load(r'**/model.pth')
#pretrain_dict={k:v for k,v in pretrain_dict.items() if k in new_params and v.size()==new_params[k].size()}# default k in m m.keys
#new_params.update(pretrain_dict)
#model.load_state_dict(new_params)

model_g.train()
model_g.cuda()

model_d.train()
model_d.cuda()

################### optimizer ###################
optimizer_g=torch.optim.Adam(model_g.parameters(),lr=lr_g,betas=(0.9,0.99),weight_decay=weight_decay)
#optimizer_g.zero_grad()

optimizer_d=torch.optim.Adam(model_d.parameters(),lr=lr_d,betas=(0.9,0.99),weight_decay=weight_decay)
#optimizer_d.zero_grad()

################### iter train ###################
for iters in range(max_iter):
    loss_g_v=0
    loss_d_v=0
    
    ####### train D ##################
    optimizer_d.zero_grad()
    adjust_lr(optimizer_d,lr_d,iters,max_iter,power)

    # labeled data
    try:
        _,batch=next(trainloader_iter)
    except:
        trainloader_iter=enumerate(trainloader)
        _,batch=next(trainloader_iter)
    
    images,labels=batch
    images=Variable(images).cuda()
    labels=Variable(labels).cuda()
    
    # unlabeled data
    try:
        _,batch_nl=next(trainloader_nl_iter)
    except:
        trainloader_nl_iter=enumerate(trainloader_nl)
        _,batch_nl=next(trainloader_nl_iter)
    
    images_nl=batch_nl
    images_nl=Variable(images_nl).cuda()
    if images.shape[0] != images_nl.shape[0]:
        continue
    # noise data
    noise = torch.rand([images.shape[0],50*50]).uniform_().cuda()
    # predict
    pred_labeled = model_d(images)
    pred_unlabel = model_d(images_nl)
    pred_fake    = model_d( model_g(noise) )
    # compute loss
    loss_labeled = Loss_label(pred_labeled,labels)
    loss_unlabel = Loss_unlabel(pred_unlabel)
    loss_fake    = Loss_fake(pred_fake)
    
    loss_d       = loss_labeled + 0.5*loss_fake + 0.5*loss_unlabel
    loss_d_v += loss_d.data.cpu().numpy().item()
    loss_d.backward()
    optimizer_d.step()
    
    ####### train G ##################
    optimizer_g.zero_grad()
    adjust_lr(optimizer_g,lr_g,iters,max_iter,power)
    # predict
    pred_fake    = model_d( model_g(noise) )
    loss_g    = -Loss_fake1(pred_fake)
    loss_g_v += loss_g.data.cpu().numpy().item()
    loss_g.backward()
    optimizer_g.step()
    
    # output loss value
    print('iter=%d , loss_g=%.2f , loss_d=%.2f'%(iters,loss_g_v,loss_d_v))
    # save model
    if iters%1000==0 and iters!=0:
        # test image
#        img=Image.open(os.path.join(test_path,names[i]))
#        r,g,b=img.split()
#        img=Image.merge('RGB',(b,g,r))
#        img_=img_transform(img)
#        img_=torch.unsqueeze(img_,dim=0)
#        image=Variable(img_).cuda()
#        predict=model(image)
#        P=torch.max(predict,1)[1].cuda().data.cpu().numpy()[0]
#        P=np.uint8(P)
#        cv2.imwrite(os.path.join(pre_path,names[i]),P)
        
        torch.save(model_d.state_dict(),model_s_path)
        torch.save(model_g.state_dict(),model_g_spath)
torch.save(model_d.state_dict(),model_s_path)
torch.save(model_g.state_dict(),model_g_spath)





