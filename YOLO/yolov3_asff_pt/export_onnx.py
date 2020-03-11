from alfred.dl.torch.common import device
import glob
import time
from torch.autograd import Variable
import torch
from utils.utils import *
from dataset.vocdataset import VOC_CLASSES
from dataset.cocodataset import COCO_CLASSES
from dataset.data_augment import ValTransform
from utils.vis_utils import vis

import os
import sys
import argparse
import yaml
import cv2
cv2.setNumThreads(0)


"""
export this tiny yolov3 asff model to onnx
to see if we can get something, possibely 
deploy this model to TensorRT or ncnn via ONNX
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolov3_baseline.cfg',
                        help='config file. see readme')
    parser.add_argument('-d', '--dataset', type=str, default='COCO')
    parser.add_argument('-i', '--img', type=str, default='images',)
    parser.add_argument('-c', '--checkpoint', type=str, default='./weights/YOLOv3-ASFF_800_43.9.pth',
                        help='pytorch checkpoint file path')
    parser.add_argument('-s', '--test_size', type=int, default=608)
    parser.add_argument('--half', dest='half', action='store_true', default=False,
                        help='FP16 training')
    parser.add_argument('--rfb', dest='rfb', action='store_true', default=True,
                        help='Use rfb block')
    parser.add_argument('--asff', dest='asff', action='store_true', default=True,
                        help='Use ASFF module for yolov3')
    parser.add_argument('--use_cuda', type=bool, default=True)
    return parser.parse_args()


def demo():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    cuda = torch.cuda.is_available() and args.use_cuda
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    print("successfully loaded config file: ", cfg)
    backbone = cfg['MODEL']['BACKBONE']
    test_size = (args.test_size, args.test_size)

    if args.dataset == 'COCO':
        class_names = COCO_CLASSES
        num_class = 80
    elif args.dataset == 'VOC':
        class_names = VOC_CLASSES
        num_class = 20
    else:
        raise Exception("Only support COCO or VOC model now!")

    # Initiate model
    if args.asff:
        if backbone == 'mobile':
            from models.yolov3_mobilev2 import YOLOv3
            print(
                "For mobilenet, we currently don't support dropblock, rfb and FeatureAdaption")
        else:
            from models.yolov3_asff import YOLOv3
        print('Training YOLOv3 with ASFF!')
        model = YOLOv3(num_classes=num_class, rfb=args.rfb, asff=args.asff)
    else:
        if backbone == 'mobile':
            from models.yolov3_mobilev2 import YOLOv3
        else:
            from models.yolov3_baseline import YOLOv3
        print('Training YOLOv3 strong baseline!')
        model = YOLOv3(num_classes=num_class, rfb=args.rfb)

    if args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        cpu_device = torch.device("cpu")
        ckpt = torch.load(args.checkpoint, map_location=cpu_device)
        # model.load_state_dict(ckpt,strict=False)
        model.load_state_dict(ckpt)
    if cuda:
        print("using cuda")
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        model = model.to(device)
    if args.half:
        model = model.half()

    model = model.eval()
    print('model is ready..')

    # input width is 608 and 608
    dummy_input = torch.zeros((3, 800, 800)).to(device).unsqueeze(0)
    onnx_f = 'weights/yolov3_asff.onnx'
    torch.onnx.export(model, dummy_input, onnx_f, 
                      verbose=True)
    print('onnx exported successfully.')

if __name__ == '__main__':
    demo()
