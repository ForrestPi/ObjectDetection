"""

simply demo on inference on YoloV3ASFF

"""
import time
import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable
import torch.nn.init as init
import torch
from utils.utils import *
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.voc_evaluator import VOCEvaluator
from utils import distributed_util
from utils.distributed_util import reduce_loss_dict
from dataset.cocodataset import *
from dataset.vocdataset import *
from dataset.data_augment import TrainTransform
from dataset.dataloading import *
from utils.utils import *
from dataset.data_augment import ValTransform

import os
import sys
import argparse
import yaml
import random
import math
import cv2
import glob
from models.yolov3_asff import YOLOv3
from alfred.dl.torch.common import device
from alfred.vis.image.det import visualize_det_cv2_part
from alfred.vis.image.get_dataset_label_map import coco_label_map_list
from alfred.vis.image.common import get_unique_color_by_id


num_classes = 80
checkpoint = 'weights/YOLOv3-ASFF_800_43.9.pth'
target_size = (800, 800)
rgb_means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
data_f = './images'


def build_model():
    model = YOLOv3(num_classes=num_classes, ignore_thre=0.5,
                   label_smooth=True, rfb=True, vis=False)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print('model loaded.')
    return model


def demo():
    model = build_model()
    if os.path.isdir(data_f):
        all_imgs = glob.glob(os.path.join(data_f, '*.jpg'))
        for img in all_imgs:
            print('~~~~~ predict on img: {}'.format(img))
            im = cv2.imread(img)
            ori_im = im.copy()
            height, width, _ = im.shape
            transform = ValTransform(rgb_means=(
                0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            im_input, _ = transform(im, None, target_size)
            im_input = im_input.to(device).unsqueeze(0)

            with torch.no_grad():
                out = model(im_input)
                outputs = postprocess(out, num_classes, 0.01, 0.65)
                outputs = outputs[0].cpu().data
                bboxes = outputs[:, 0:4]
                bboxes[:, 0::2] *= width / target_size[0]
                bboxes[:, 1::2] *= height / target_size[1]
                cls = outputs[:, 6]
                scores = outputs[:, 4] * outputs[:, 5]
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                res = visualize_det_cv2_part(
                    im, scores, cls, bboxes, coco_label_map_list[1:], 0.1)
                cv2.imshow('rr', res)
                cv2.waitKey(0)


if __name__ == "__main__":
    demo()
