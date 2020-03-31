# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from layers import FPN50
from utils import get_detector
from loss_modules import mul_task_loss

class FCOS_head(nn.Module):
    def __init__(self, num_classes = 20):
        super(FCOS_head,self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.fpn_stride = [8, 16, 32, 64, 128]
        self.head_cls = self._make_head(self.num_classes)
        self.head_boxes = self._make_head(4)
        self.head_center = self._make_head(1)
        
    def forward(self, x):
        features = self.fpn(x)
        pred_cls = []
        pred_boxes = []
        pred_centerness = []
        locations = []
        for idx, ft in enumerate(features):
            cls = self.head_cls(ft)
            box = self.head_boxes(ft)
            center = self.head_center(ft)
            cls = cls.permute(0,2,3,1)        #[N, cls, H, W] --> [N, H, W, cls]
            box = box.permute(0,2,3,1)    #[N, 4, H, W] --> [N, H, W, 4]
            center = center.permute(0,2,3,1)  #[N, 1, H, W] --> [N, H, W, 1]
            pred_cls.append(cls)
            pred_boxes.append(box)
            pred_centerness.append(center)            
            #compute locations
            stride = self.fpn_stride[idx]
            h, w = ft.size()[2:]
            location = self._make_location(h, w, stride)
            #print(location.shape,'***')
            locations.append(location.type(cls.type()))
       
        return pred_cls, pred_boxes, pred_centerness, locations
    
    def _make_location(self, h, w, stride, device=None):
        x = torch.arange(0, w*stride, step=stride, dtype=torch.float32, device=device)
        y = torch.arange(0, h*stride, step=stride, dtype=torch.float32, device=device)
        
        yy, xx = torch.meshgrid((y, x))
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        
        location = torch.stack((xx, yy), dim=1) + stride // 2
        return location

    def _make_head(self, planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3,padding=1, stride=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)
            

class FCOS(nn.Module):
    def __init__(self, cfg):
        super(FCOS,self).__init__()
        self.head = FCOS_head(cfg['num_classes'])#the number of classes including background        
        self.compute_loss = mul_task_loss(cfg)
        self.get_detector = get_detector(cfg)
        
    def forward(self, images, targets=None):
        pred_cls, pred_boxes, pred_centerness, locations = self.head(images)
        image_sizes = [list(images.size()[2:])]
        # testing
        if targets is None:
            detections = self.get_detector(locations, pred_cls, pred_boxes, pred_centerness, image_sizes)
            return detections
        # training
        else:
            losses = self.compute_loss(pred_cls, pred_boxes, pred_centerness, locations, targets)
            return losses

if __name__ == "__main__":
    head = FCOS_head(num_classes = 20)
    x = torch.randn(1,3,256,256)  
    pred_cls, pred_boxes, pred_centerness, locations =  head(x)
    print("pred_cls:",pred_cls[3].size())  #torch.Size([1, 4, 4, 20])
    print("pred_boxes:",pred_boxes[3].size()) #torch.Size([1, 4, 4, 4])
    print("pred_centerness:",pred_centerness[3].size()) #torch.Size([1, 4, 4, 1])
    print("locations:",locations[3].size()) #torch.Size([16, 2]) #4*4,2
    
            

        



'''
if __name__=="__main__":
    from config import config 
    from datasets.datasets import Dataset
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
   
    ii = r'./Data/UCAS/ucas_train.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(),])
 
    img_size = 256
    da = Dataset(ii,transform=transform,img_size=img_size, train=False)
    dataloader = torch.utils.data.DataLoader(da,batch_size=1,shuffle=False)
    #x = torch.randn(1,3,128,128)
    f = FCOS(config)
    #checkpoint = torch.load('./checkpoint/ckpt.pth')
    #f.load_state_dict(checkpoint['weights'])
 
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        images = imgs
        targets = targets
        #loss = f(images, targets)
        detections = f(images)
        
        
        break
'''    
    
    
    
    
    
    
    
    
    
    
