"""
we are trying to training m2det from scratch
on vgg model

"""

from torch.utils.data import DataLoader
from utils.core import *
from m2det import build_net
from layers.functions import Detect, PriorBox

from alfred.utils.log import logger as logging
from alfred.dl.torch.common import device
from alfred.vis.image.get_dataset_label_map import coco_label_map
from alfred.vis.image.det import visualize_det_cv2

import cv2


start_epoch = 9
checkpoint_path = 'weights/m2det320_resnet50_coco_epoch_{}.pth'
config_f = 'configs/m2det320_resnet50.py'
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

classes = list(coco_label_map.values())
classes.insert(0, 'bk')


def train(cfg):
    cfg = Config.fromfile(cfg)
    net = build_net('train',
                    size=cfg.model.input_size,  # Only 320, 512, 704 and 800 are supported
                    config=cfg.model.m2det_config)
    init_net(net, cfg, False)
    net.to(device)
    if os.path.exists(checkpoint_path.format(start_epoch)):
        checkpoints = torch.load(checkpoint_path.format(start_epoch))
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

    anchor_config = anchors(cfg)
    detector = Detect(cfg.model.m2det_config.num_classes,
                      cfg.loss.bkg_label, anchor_config)
    logging.info('detector initiated.')

    dataset = get_dataloader(cfg, 'COCO', 'train_sets')
    train_ds = DataLoader(dataset, cfg.train_cfg.per_batch_size,
                          shuffle=True,
                          num_workers=0,
                          collate_fn=detection_collate)
    logging.info('dataset loaded, start to train...')

    for epoch in range(start_epoch, cfg.model.epochs):
        for i, data in enumerate(train_ds):
            try:
                lr = adjust_learning_rate_v2(optimizer, epoch, cfg)
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

                if i % 2000 == 0:
                    # two_imgs = images[0:2, :]
                    # out = net(two_imgs)
                    # snap_middle_result(two_imgs[0], out[0], priors, detector, cfg, epoch)
                    torch.save(net.state_dict(), checkpoint_path.format(epoch))
                    logging.info('model saved.')
            except KeyboardInterrupt:
                torch.save(net.state_dict(), checkpoint_path.format(epoch))
                logging.info('model saved.')
                exit(0)
    torch.save(net.state_dict(), checkpoint_path.format(epoch))


# def snap_middle_result(one_img, one_out, priors, detector, cfg, epoch):
#     print(one_out.shape)
#     one_out = one_out.unsqueeze(dim=0)
#     print(one_out.shape)
#     _, h, w = one_img.shape
#     scale = torch.Tensor([w, h, w, h])
#     boxes, scores = detector.forward(one_out, priors)
#     boxes = (boxes[0] * scale).cpu().numpy()
#     scores = scores[0].cpu().numpy()
#     allboxes = []
#     for j in range(1, cfg.model.m2det_config.num_classes):
#         inds = np.where(scores[:, j] > cfg.test_cfg.score_threshold)[0]
#         if len(inds) == 0:
#             continue
#         c_bboxes = boxes[inds]
#         c_scores = scores[inds, j]
#         c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
#             np.float32, copy=False)
#         soft_nms = cfg.test_cfg.soft_nms
#         # min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
#         keep = nms(c_dets, cfg.test_cfg.iou, force_cpu=soft_nms)
#         keep = keep[:cfg.test_cfg.keep_per_class]
#         c_dets = c_dets[keep, :]
#         allboxes.extend([_.tolist() + [j] for _ in c_dets])
#     allboxes = np.array(allboxes)
#     # [boxes, scores, label_id] -> [id, score, boxes] 0, 1, 2, 3, 4, 5
#     allboxes = allboxes[:, [5, 4, 0, 1, 2, 3]]
#     logging.info('allboxes shape: {}'.format(allboxes.shape))
#     image = np.array(one_img.cpu().numpy(), dtype=np.uint8)
#     res = visualize_det_cv2(image, allboxes, classes=classes, thresh=0.5)
#     cv2.imwrite('results/pred_{}.png'.format(epoch), res)


if __name__ == '__main__':
    logging.info('loading from config file: {}'.format(config_f))
    train(config_f)
