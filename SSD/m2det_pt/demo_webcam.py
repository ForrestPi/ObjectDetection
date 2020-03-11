"""
Simple demo inference on m2det
"""
import os
import cv2
import numpy as np
import sys
from layers.functions import Detect, PriorBox
from m2det import build_net
from data import BaseTransform
from data.custom_voc_like_helmet import HELMET_CLASSES
from utils.core import *
from alfred.dl.torch.common import device
from alfred.utils.log import logger as logging
from alfred.vis.image.det import visualize_det_cv2, visualize_det_cv2_fancy
from alfred.vis.image.get_dataset_label_map import coco_label_map


# config_f = 'configs/m2det320_vgg.py'
# checkpoint_path = 'weights/m2det320_vgg_coco_epoch_16.pth'

config_f = 'configs/m2det512_vgg.py'
checkpoint_path = 'weights/m2det512_vgg_coco_epoch_14.pth'

# config_f = 'configs/m2det512_vgg_helmet.py'
# checkpoint_path = 'weights/m2det512_vgg_helmet_epoch_199.pth'
config_f = 'configs/m2det512_vgg_helmet.py'
# checkpoint_path = 'weights/m2det512_vgg_helmet_epoch_550.pth'

# config_f = 'configs/m2det320_vgg_helmet.py'
# checkpoint_path = 'weights/m2det320_vgg_helmet_epoch_246.pth'

# checkpoint_path = 'weights/M2Det_COCO_size320_netresnet101_epoch0.pth'
# config_f = 'configs/m2det320_resnet101.py'


classes = list(coco_label_map.values())
# classes = HELMET_CLASSES


def demo(v_f):
    cfg = Config.fromfile(config_f)
    anchor_config = anchors(cfg)
    priorbox = PriorBox(anchor_config)
    net = build_net('test',
                    size=cfg.model.input_size,
                    config=cfg.model.m2det_config)
    init_net(net, cfg, checkpoint_path)
    net.eval().to(device)
    with torch.no_grad():
        priors = priorbox.forward().to(device)
    _preprocess = BaseTransform(
        cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
    detector = Detect(cfg.model.m2det_config.num_classes,
                      cfg.loss.bkg_label, anchor_config)
    logging.info('detector initiated.')

    cap = cv2.VideoCapture(v_f)
    logging.info('detect on: {}'.format(v_f))
    logging.info('video width: {}, height: {}'.format(int(cap.get(3)), int(cap.get(4))))
    out_video = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'MJPG'), 24, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, image = cap.read()
        if not ret:
            out_video.release()
            cv2.destroyAllWindows()
            cap.release()
            break
        w, h = image.shape[1], image.shape[0]
        img = _preprocess(image).unsqueeze(0).to(device)
        scale = torch.Tensor([w, h, w, h])
        out = net(img)
        boxes, scores = detector.forward(out, priors)
        boxes = (boxes[0]*scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        allboxes = []
        for j in range(1, cfg.model.m2det_config.num_classes):
            inds = np.where(scores[:, j] > cfg.test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            soft_nms = cfg.test_cfg.soft_nms
            # min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
            keep = nms(c_dets, cfg.test_cfg.iou, force_cpu=soft_nms)
            keep = keep[:cfg.test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            allboxes.extend([_.tolist()+[j] for _ in c_dets])
        if len(allboxes) > 0:
            allboxes = np.array(allboxes)
            # [boxes, scores, label_id] -> [id, score, boxes] 0, 1, 2, 3, 4, 5
            allboxes = allboxes[:, [5, 4, 0, 1, 2, 3]]
            logging.info('allboxes shape: {}'.format(allboxes.shape))
            res = visualize_det_cv2(image, allboxes, classes=classes, thresh=0.2)
            # res = visualize_det_cv2_fancy(image, allboxes, classes=classes, thresh=0.2, r=4, d=6)
            cv2.imshow('rr', res)
            out_video.write(res)
            cv2.waitKey(1)


if __name__ == "__main__":
    v_f = sys.argv[1]
    demo(v_f)
