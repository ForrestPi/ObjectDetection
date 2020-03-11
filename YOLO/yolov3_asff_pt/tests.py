from models.yolov3_asff import build_yolov3_modules
from models.yolov3_asff import ASFF, YOLOv3Head

import torch
from torch import nn

import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable
import torch.nn.init as init
import os
import cv2
import glob
from alfred.dl.torch.common import device
from alfred.utils.log import logger as logging

from dataset.data_augment import ValTransform
from alfred.dl.torch.common import device
from alfred.vis.image.det import visualize_det_cv2_part
from alfred.vis.image.get_dataset_label_map import coco_label_map_list
from alfred.vis.image.common import get_unique_color_by_id
from utils.utils import postprocess

class YOLOv3(nn.Module):

    def __init__(self, num_classes=80, ignore_thre=0.7, label_smooth=False, rfb=False, vis=False, asff=False):
        super(YOLOv3, self).__init__()
        self.module_list = build_yolov3_modules(
            num_classes, ignore_thre, label_smooth, rfb)

        self.level_0_fusion = ASFF(level=0, rfb=rfb, vis=vis)
        self.level_0_header = YOLOv3Head(anch_mask=[6, 7, 8], n_classes=num_classes, stride=32, in_ch=1024,
                                         ignore_thre=ignore_thre, label_smooth=label_smooth, rfb=rfb)
        self.level_1_fusion = ASFF(level=1, rfb=rfb, vis=vis)
        self.level_1_header = YOLOv3Head(anch_mask=[3, 4, 5], n_classes=num_classes, stride=16, in_ch=512,
                                         ignore_thre=ignore_thre, label_smooth=label_smooth, rfb=rfb)
        self.level_2_fusion = ASFF(level=2, rfb=rfb, vis=vis)
        self.level_2_header = YOLOv3Head(anch_mask=[0, 1, 2], n_classes=num_classes, stride=8, in_ch=256,
                                         ignore_thre=ignore_thre, label_smooth=label_smooth, rfb=rfb)

    def forward(self, x, targets=None, epoch=0):
        output = []
        route_layers = []

        for i, module in enumerate(self.module_list):
            x = module(x)
            if i in [6, 8, 17, 24, 32]:
                route_layers.append(x)
            if i == 19:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 26:
                x = torch.cat((x, route_layers[0]), 1)
        print(len(route_layers))
        fused_0 = self.level_0_fusion(route_layers[2], route_layers[3], route_layers[4])
        print(fused_0)
        x = self.level_0_header(fused_0)
        # x = fused_0
        print(x)
        # output.append(x)

        # fused_1 = self.level_1_fusion(route_layers[2], route_layers[3], route_layers[4])
        # x = self.level_1_header(fused_1)
        # output.append(x)

        # fused_2 = self.level_2_fusion(route_layers[2], route_layers[3], route_layers[4])
        # x = self.level_2_header(fused_2)
        # output.append(x)
        # return torch.cat(output, 1)
        return x


class TestModel(nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()
        self.module_list = build_yolov3_modules(80, 0.6, True, True)

    def forward(self, x):
        route_layers = []
        for i, module in enumerate(self.module_list):
            # yolo layers
            x = module(x)
            # route layers
            if i in [6, 8, 17, 24, 32]:
                route_layers.append(x)
            if i == 19:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 26:
                x = torch.cat((x, route_layers[0]), 1)
        print(x)
        return x


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
                res = visualize_det_cv2_part(
                    im, scores, cls, bboxes, coco_label_map_list[1:], 0.1)
                cv2.imshow('rr', res)
                cv2.waitKey(0)


if __name__ == "__main__":
    demo()


# if __name__ == "__main__":
#     model = YOLOv3()
#     model.to(device)
#     model.eval()
#     dummy_data = torch.randn([1, 3, 800, 800]).to(device)
#     logging.info('start to run model')
#     a = model(dummy_data)
#     print(a.shape)
#     torch.onnx.export(model, dummy_data, 'test.onnx', verbose=False)
