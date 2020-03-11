
import os
import warnings
import sys
import time
import torch
import shutil
import argparse
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


from layers.functions import PriorBox
from data import detection_collate
from configs.CC import Config
from utils.core import *
from m2det import build_net

from alfred.utils.log import logger as logging
from alfred.dl.torch.common import device


checkpoint_path = 'weights/M2Det_COCO_size320_netresnet101_epoch0.pth'


def train(cfg):
    cfg = Config.fromfile(cfg)
    net = build_net('train',
                    size=cfg.model.input_size,  # Only 320, 512, 704 and 800 are supported
                    config=cfg.model.m2det_config)
    net.to(device)
    if os.path.exists(checkpoint_path):
        checkpoints = torch.load(checkpoint_path)
        net.load_state_dict(checkpoints)
        logging.info('checkpoint loaded.')

    optimizer = optim.SGD(net.parameters(),
                          lr=cfg.train_cfg.lr[0],
                          momentum=cfg.optimizer.momentum,
                          weight_decay=cfg.optimizer.weight_decay)
    criterion = MultiBoxLoss(cfg.model.m2det_config.num_classes,
                             overlap_thresh=cfg.loss.overlap_thresh,
                             prior_for_matching=cfg.loss.prior_for_matching,
                             bkg_label=cfg.loss.bkg_label,
                             neg_mining=cfg.loss.neg_mining,
                             neg_pos=cfg.loss.neg_pos,
                             neg_overlap=cfg.loss.neg_overlap,
                             encode_target=cfg.loss
                             .encode_target)
    priorbox = PriorBox(anchors(cfg))
    with torch.no_grad():
        priors = priorbox.forward().to(device)
    net.train()

    dataset = get_dataloader(cfg, 'COCO', 'train_sets')
    train_ds = DataLoader(dataset, cfg.train_cfg.per_batch_size,
                          shuffle=True,
                          num_workers=0,
                          collate_fn=detection_collate)
    logging.info('dataset loaded, start to train...')

    for epoch in range(cfg.model.epochs):
        for i, data in enumerate(train_ds):
            try:
                lr = adjust_learning_rate_v2(optimizer, epoch, i, 10320, cfg)
                images, targets = data
                images = images.to(device)
                targets = [anno.to(device) for anno in targets]
                out = net(images)

                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, priors, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()

                if i % 30 == 0:
                    logging.info('Epoch: {}, iter: {}, loc_loss: {}, conf_loss: {}, loss: {}, lr: {}'.format(
                        epoch, i, loss_l.item(), loss_c.item(), loss.item(), lr
                    ))

                if iter % 2000 == 0:
                    torch.save(net.state_dict(), checkpoint_path)
                    logging.info('model saved.')
            except KeyboardInterrupt:
                torch.save(net.state_dict(), checkpoint_path)
                logging.info('model saved.')
                exit(0)
    torch.save(net.state_dict(), checkpoint_path)


if __name__ == '__main__':
    train(sys.argv[1])
