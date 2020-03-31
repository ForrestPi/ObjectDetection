# -*- coding: utf-8 -*-
import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .augmentations import random_flip, resize, up_down_flip

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416, transform=None):
        self.files = sorted(glob.glob(r'%s/*.*' % folder_path))  
        self.img_size = (img_size, img_size)
        self.transform = transform
    def __getitem__(self,index):
        image_path = self.files[index % len(self.files)]
        #extract images
        img = np.array(Image.open(image_path))  # h w 
        img = img[...,:3]
        #pdb.set_trace()
        h, w ,_ = img.shape
        dim_diff = np.abs(h-w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff // 2
        #Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, 'constant', constant_values=127.5)
        # Resize and normalize
        input_img = cv2.resize(input_img, self.img_size, interpolation=cv2.INTER_CUBIC)
        # Channels-first
        if self.transform is not None:
            input_img = Image.fromarray(input_img)
            input_img = self.transform(input_img)
        else:
            input_img = np.transpose(input_img, (2, 0, 1)) / 255.
            # As pytorch tensor
            input_img = torch.from_numpy(input_img).float()

        return image_path, input_img
    def __len__(self):
        return len(self.files)


class Dataset(Dataset):
    '''
        args: 
	    root: image path
            list_path: img_name x y x_max y_max pieces ......
        return:
            img_path, input_img, labels[xmin,ymin,xmax,ymax,piece,box_area..]
    '''
    def __init__(self,root, list_path, img_size=416, transform=None, train=False):
        with open(list_path,'r') as file:
            files = file.readlines()
            self.num_samples = len(files)
        files = [i.strip() for i in files]
        self.img_files = [os.path.join(root, i.split(' ')[0]) for i in files]
        
        self.label_files = [i.split(' ')[1:] for i in files]
        self.img_size = (img_size, img_size)
        self.max_objects = 128
        self.transform = transform
        self.train = train
        
    def __getitem__(self,index):
        #-----------
        #image
        #-----------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)
        # Black and white images
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        
        img = img[...,:3]
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = input_img.shape
        ratio_ = padded_h / self.img_size[0]  #  500/416
        
        # Resize and normalize
        input_img = cv2.resize(input_img, self.img_size, interpolation=cv2.INTER_CUBIC)
        #---------
        #  Label
        #---------
        label_path = self.label_files[index % len(self.img_files)]
        label_path = [float(i) for i in label_path]
        labels = np.array(label_path).reshape(-1,5).astype('float64')
         # Extract coordinates for unpadded + unscaled image
        
        x1 = labels[:,0]
        y1 = labels[:,1]
        x2 = labels[:,2]
        y2 = labels[:,3]
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        #### normalize
        x1,y1,x2,y2 = x1/ratio_, y1/ratio_, x2/ratio_, y2/ratio_
        
        boxes = np.zeros((x1.shape[0],4))
        boxes[:,0] = x1
        boxes[:,1] = y1
        boxes[:,2] = x2
        boxes[:,3] = y2
        #----------------------------------------------------------------------
        ##  data Augmentation
        #--------------------------------------------------------------------
        input_img = Image.fromarray(input_img)
        if self.train == True:
            input_img, boxes = random_flip(input_img, boxes)
            input_img, boxes = up_down_flip(input_img, boxes)
            input_img, boxes = resize(input_img, boxes,self.img_size)
            
        
        area = []
        for box in boxes:
            area.append(((box[2]-box[0])*(box[3]-box[1])).item())
        area = np.array(area)
        
        labels[:,:4] = boxes  
        labels[:, -1] += 1  # class + 1  *********
        labels_ = np.zeros((len(labels), 6))
        labels_[:,:5] = labels
        labels_[:,-1] = area
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 6))
        
        if labels is not None:
            filled_labels[range(len(labels_))[:self.max_objects]] = labels_[:self.max_objects]
        
        if self.transform is None:
            #Channels-first
            input_img = np.transpose(input_img, (2, 0, 1))/255.
            # As pytorch tensor
            input_img = torch.from_numpy(input_img).float()
        else:
            input_img = self.transform(input_img)
        return img_path, input_img, filled_labels
    def __len__(self):
        return (self.num_samples)
