# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:40:09 2018

Init FCOS with pretrained ResNet50 model.

Download pretrained ResNet50 params from:
  https://download.pytorch.org/models/resnet50-19c8e357.pth

"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from models.layers import FPN50
from models.fcos import FCOS
from config import config

cfg = config


print('Loading pretrained ResNet50 model..')
d = torch.load('./resnet50.pth')
#
print('Loading into FPN50..')
fpn = FPN50()
dd = fpn.state_dict()
for k in d.keys():
    if not k.startswith('fc'):  # skip fc layers
        dd[k] = d[k]
#
print('Saving FCOS..')
net = FCOS(cfg)
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

net.head.fpn.load_state_dict(dd)
torch.save(net.state_dict(), 'net.pth')
print('Done!')
