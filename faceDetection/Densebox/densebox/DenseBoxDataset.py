import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from pycocotools.coco import COCO

#test
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class DenseBoxDatasetOnline(Dataset):
    """
    自定义数据集
    """
    CLASSES = ('chepai', )

    def __init__(self, root, ann_file = 'train.json', size=(240, 240), test_mode=False):
        """
        初始化
        """
        self.root = root
        self.size = size
        self.img_infos = self.load_annotations(root+ann_file)
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.test_mode = test_mode
        self.bboxes = []

        for idx in range(len(self.img_infos)):
            img_info = self.img_infos[idx]

            ann = self.get_ann_info(idx)
            gt_bboxes = ann['bboxes']
            gt_labels = ann['labels']
            ori_shape = (img_info['width'], img_info['height'], 3)
            dw = 1.0/ori_shape[0]
            dh = 1.0/ori_shape[1]
            bbox = []
            for i in range(len(gt_bboxes)):
                gt_bboxes[i][0] = gt_bboxes[i][0]*dw*self.size[0]
                gt_bboxes[i][1] = gt_bboxes[i][1]*dh*self.size[1]
                gt_bboxes[i][2] = gt_bboxes[i][2]*dw*self.size[0]
                gt_bboxes[i][3] = gt_bboxes[i][3]*dh*self.size[1]
                #一张图片中有多个bbox
                bbox.append([gt_bboxes[i][0], gt_bboxes[i][1], gt_bboxes[i][2], gt_bboxes[i][3]])
                #一张图片中只有一个bbox
                #bbox=[gt_bboxes[i][0], gt_bboxes[i][1], gt_bboxes[i][2], gt_bboxes[i][3]]
            self.bboxes.append(bbox)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def __len__(self):
        """
        :return:
        """
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        """
        先获取图片id,再得到对应的ann_id,最后得到该图片的annotation信息
        """
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info)
    

    def _parse_ann_info(self, ann_info):
        """
        返回某张图片中annotation的详细信息
        同时将bbox由[x,y,w,h]转换为[xmin,ymin,xmax,ymax]
        """
        gt_bboxes = []
        gt_labels = []
        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann['bbox']
            # 将bbox转换成[xmin, ymin, xmax, ymax]的形式
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels)
        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        # 加载图片
        img = Image.open(self.root+ 'JPEG/' + img_info['filename'])
        #print(img_info['filename'])
        # 转换为RGB格式以及规定尺寸
        if img.mode == 'L' or img.mode == 'I':  # 8bit or 32bit gray-scale
            img = img.convert('RGB')
        img = self.transform(img)
        #show(img, self.bboxes[idx])

        return img, self.bboxes[idx]  #返回img,bbox

# 显示图片
def show(img, bboxes):
    """
    img为tensor格式，bbox为list
    """
    img = np.array(img)
    img = np.transpose(img,(1,2,0))
    plt.imshow(img)
    #画矩形框
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        top = ([xmin, xmax], [ymin, ymin])
        right = ([xmax, xmax], [ymin, ymax])
        botton = ([xmax, xmin], [ymax, ymax])
        left = ([xmin, xmin], [ymax, ymin])
        lines = [top, right, botton, left]
        for line in lines:
            plt.plot(*line, color = 'r')
            plt.scatter(*line, color = 'b')
    #调整原点到左上角
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    plt.show()


