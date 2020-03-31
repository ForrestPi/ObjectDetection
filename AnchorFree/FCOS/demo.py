#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:37:20 2019

@author: wei
"""
import os
import sys
sys.path.append(os.getcwd()+'/models')
sys.path.append(os.getcwd()+'/datasets')
import cv2
import time
import torch
import random
import pprint
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from config import config
from models.fcos import FCOS
from torch.utils.data import DataLoader
from datasets.dataset import ImageFolder

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--test_path', type=str, default=r'./images/demo_images', help='size of each image dimension')
opt = parser.parse_args()
cfg = config
pprint.pprint(opt)
pprint.pprint(cfg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FCOS(cfg).to(device)
ckpt = torch.load('./checkpoints/checkpoint.pth')['weights']
model.load_state_dict(ckpt)
model.eval()
print('loading weights successfully...')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

dataset = ImageFolder(opt.test_path, cfg['img_size'], transform)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

imgs_path = []
imgs_detection = []
prev_time = time.time()
print ('\nPerforming object detection: %d samples...'%len(dataset))
for b, (image_path, input_img) in enumerate(dataloader):
    #import pdb
    #pdb.set_trace()
    input_img = input_img.to(device)
    with torch.no_grad():
        detections = model(input_img)
    
    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (b, inference_time))
    
    for idx, boxList in enumerate(detections):
        if len(boxList.bbox):
            imgs_path.append(image_path[idx])
            imgs_detection.append(boxList)
#            

# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
print ('\nSaving images:')

for img_i, (path, boxList) in enumerate(zip(imgs_path, imgs_detection)):
#
    print ("(%d) Image: '%s'" % (img_i, path))
    # Create plot
    img = np.array(Image.open(path))[...,:3]
    img = img.copy()
    
    #The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (cfg['img_size'] / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (cfg['img_size'] / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = cfg['img_size'] - pad_y
    unpad_w = cfg['img_size'] - pad_x
    
    if len(boxList.bbox):
        boxes = boxList.bbox.cpu()
        labels = boxList.get_field('labels').cpu()
        scores = boxList.get_field('scores').cpu()

        unique_labels = labels.unique()
        bbox_colors = random.sample(colors, len(unique_labels))
        for idx in range(len(boxes)):
            box = boxes[idx]
            label = labels[idx].item()
            score = scores[idx].item()

            print('\t+ Label: %s, Conf: %.5f' % (cfg['classes'][int(label)], score))
            x1,y1,x2,y2 = box
            box_h = int((((y2 - y1) / unpad_h) * img.shape[0]).item())
            box_w = int((((x2 - x1) / unpad_w) * img.shape[1]).item())
            y1 = int((((y1 - pad_y // 2) / unpad_h) * img.shape[0]).item())
            x1 = int((((x1 - pad_x // 2) / unpad_w) * img.shape[1]).item())
            x2 = x1 + box_w
            y2 = y1 + box_h
            color = bbox_colors[int(np.where(unique_labels == int(label))[0])]
            color = list(map(lambda a: a*255, color))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            #cv2.rectangle(img, (x1, y1), (x1+30, y1+30),(255, 0, 0), thickness=-1)
            #cv2.putText(img, cfg['classes'][int(label)], (int(x1), int(y1+13)), font, 0.6,(255, 255, 255), 1)
            #cv2.putText(img,str(score), (int(x1), int(y1)), font, 0.6,(255, 255, 255), 1)
        cv2.imwrite(r'./images/results/%d.png' % (img_i), img)
        
            
            
            
            
            
            
        
    
    
    
    
    



