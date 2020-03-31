#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:51:11 2019

@author: wei
"""
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'datasets'))
sys.path.append(os.path.join(os.getcwd(), 'models'))
import torch
import pprint
import argparse

import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

from config import config
from models.fcos import FCOS
from datasets.dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=int, default=1e-3, help='learing rate for training')
parser.add_argument('--batch_size', type=int, default=8, help='size of each image batch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()
 
cfg = config
pprint.pprint(opt)
pprint.pprint(cfg)


os.makedirs('checkpoints',exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## fcos model
model = FCOS(cfg)
model = model.to(device)

if opt.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoints/ckpt.pth')
    model.load_state_dict(checkpoint['weights'])
    best_loss = checkpoint['best_loss']
    start_epoch = checkpoint['epoch']
else:
    print('initial model from pretrained resnet..')
    start_epoch = 0
    pre_trained_model = torch.load(r'script/net.pth')
    model.load_state_dict(pre_trained_model)
    best_loss = float('inf')
    
# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

    
train_loader = torch.utils.data.DataLoader(
        Dataset(cfg['images_root'], cfg['train_path'],img_size=cfg['img_size'],
                    transform=transform, train=True),
                    batch_size=opt.batch_size,
                    shuffle=True,)

test_loader = torch.utils.data.DataLoader(
        Dataset(cfg['images_root'], cfg['test_path'],img_size=cfg['img_size'],
                    transform=transform, train=False),
                    batch_size=16,
                    shuffle=False,)
        
optimizer = optim.SGD(model.parameters(),lr=opt.lr, momentum=0.9,weight_decay=cfg['weight_decay'])


for epoch in range(start_epoch,start_epoch+opt.epochs):
    model.train()
    cur_loss = 0
    for i, (_,imgs,targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        losses = model(imgs,targets)  
        
        cls_loss = losses['cls_loss']
        box_loss = losses['box_loss']
        centerness_loss = losses['centerness_loss']
        
        loss = cls_loss + box_loss + centerness_loss
        
        loss.backward()
        optimizer.step()
        print('[Epoch %d/%d, Batch %d/%d] [cls_loss: %f, box_loss: %f, centerness_loss: %f, loss: %f]'%(
                epoch, opt.epochs, i, len(train_loader), cls_loss.item(), box_loss.item(), centerness_loss.item(), loss.item()))
        
        cur_loss += loss.item()
  
    cur_loss /= i
    
    if cur_loss < best_loss:
        print('\nSaving ....  | the val loss is: ',cur_loss)
        print('\n')
        state = {
                 'weights':    model.state_dict(),
                 'best_loss':       cur_loss,
                 'epoch':      epoch,
                }
        torch.save(state,'./checkpoints/%s'%(cfg['ckpt_name']))
        best_loss = cur_loss
    


