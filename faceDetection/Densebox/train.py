from densebox import DenseBoxDatasetOnline
from densebox import DenseBox

import os
import torch
import torchvision
from torch import nn
import numpy as np
import visdom

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

root = "./chepai/"

#调整学习率
def adjust_LR(optimizer,
              epoch):
    """

    :param optimizer:
    :param epoch:
    :return:
    """
    lr = 1e-9
    if epoch < 5:
        lr = 1e-9
    elif epoch >= 5 and epoch < 10:
        lr = 2e-9
    elif epoch >= 10 and epoch < 15:
        lr = 4e-9
    else:
        lr = 1e-9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

# 通过正负样本的indices更新mask,确定那些样本用于训练
def mask_by_sel(loss_mask,
                pos_indices,
                neg_indices):
    """
    cpu side calculation
    :param loss_mask:
    :param pos_indices: N×4dim
    :param neg_indices:
    :return:
    """

    assert loss_mask.size() == torch.Size([loss_mask.size(0), 1, 60, 60])

    # print('=> before fill loss mask:%d non_zeros.' % torch.nonzero(loss_mask).size(0))

    for pos_idx in pos_indices:
        loss_mask[pos_idx[0], pos_idx[1], pos_idx[2], pos_idx[3]] = 1.0
    
    for row in range(neg_indices.size(0)):
        for col in range(neg_indices.size(1)):
            idx = int(neg_indices[row][col])

            if idx < 0 or idx >= 3600: #60*60
                # print('=> idx: ', idx)
                continue

            y = idx // 60
            x = idx % 60

            try:
                # row相当于batch维
                loss_mask[row, 0, y, x] = 1.0
            except Exception as e:
                print(row, y, x)

    # print('=> before fill loss mask:%d non_zeros.' % torch.nonzero(loss_mask).size(0))


def mask_gray_zone_cls_pn(loss_mask,
                          bboxes,
                          labels,
                          ratio=0.3,
                          gray_border=2.0):
    """
    only used in online training mode, including positive and negative patches
    process a batch                 0         1          2            3
    :param loss_mask:
    :param bboxes:  batch_size×4: leftup_x, leftup_y, rightdown_x, rightdown_y
    :param loss_mask:
    :param bboxes:
    :param labels:
    :param ratio:
    :param gray_border:
    :return:
    """
    assert loss_mask.size(1) == 1 \
           and loss_mask.size(2) == 60 \
           and loss_mask.size(3) == 60

    # assert bboxes.size() == torch.Size([bboxes.size(0), 4])

    for item_i, (coord, lb) in enumerate(zip(bboxes.numpy(), labels.numpy())):
        # process each item in the batch
        if lb == 0.0:
            continue

        bbox_center_x = float(coord[0] + coord[2]) * 0.5
        bbox_center_y = float(coord[1] + coord[3]) * 0.5

        bbox_w = coord[2] - coord[0]
        bbox_h = coord[3] - coord[1]

        org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5)
                    - gray_border + 0.5)
        org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5)
                    - gray_border + 0.5)
        end_x = int(float(org_x) + float(ratio * bbox_w) + gray_border * 2.0 + 0.5)
        end_y = int(float(org_y) + float(ratio * bbox_h) + gray_border * 2.0 + 0.5)

        # fill gray zone with 0
        loss_mask[item_i, 0, org_y: end_y, org_x: end_x] = 0.0

        loss_mask[item_i, 0,
        org_y + int(gray_border): end_y - int(gray_border) + 1,
        org_x + int(gray_border): end_x - int(gray_border) + 1] = 1.0

def collate_fn_customer(batch):
    """
    这个函数的作用是将读取到的batch中的多组数据,融合成整体
    也就是增加一个batch维度
    """
    images = []
    bboxes = []
    for i, data in enumerate(batch):
        # data[0]为img维度
        images.append(data[0])
        # data[1]为bbox维度
        bboxes.append(data[1])
    
    #images类型转换:list==>torch.tensor
    images = torch.stack(images)
    batch = (images, bboxes)
    return batch
    

# 初始化score_map
def init_score_map(bboxes, ratio=0.3):
    """
    初始化score_map
    ratio为保留的中心区域的比率
    """
    score_map = torch.zeros([1, 60, 60], dtype=torch.float32)
    #初始化score_map
    for bbox in bboxes:
        #首先转换到60*60坐标空间
        leftup_x = bbox[0] / 4.0
        leftup_y = bbox[1] / 4.0
        rightdown_x = bbox[2] / 4.0
        rightdown_y = bbox[3] / 4.0

        bbox_center_x = float(leftup_x + rightdown_x) * 0.5
        bbox_center_y = float(leftup_y + rightdown_y) * 0.5
        bbox_w = rightdown_x - leftup_x
        bbox_h = rightdown_y - leftup_y

        org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
        org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
        end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
        end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
        score_map[:, org_y: end_y + 1, org_x: end_x + 1] = 1.0
    return score_map

def init_dist_map(bboxes, ratio = 0.3):
    dist_map = torch.zeros([4, 60, 60], dtype=torch.float32)
    dxt_map, dyt_map = dist_map[0], dist_map[1]
    dxb_map, dyb_map = dist_map[2], dist_map[3]
    for bbox in bboxes:
        #首先转换到60*60坐标空间
        leftup_x = bbox[0] / 4.0
        leftup_y = bbox[1] / 4.0
        rightdown_x = bbox[2] / 4.0
        rightdown_y = bbox[3] / 4.0

        bbox_w = rightdown_x - leftup_x
        bbox_h = rightdown_y - leftup_y

        for y in range(dxt_map.size(0)):  # dim H
            for x in range(dxt_map.size(1)):  # dim W
                dist_xt = (float(x) - leftup_x)
                dist_yt = (float(y) - leftup_y)
                dist_xb = (float(x) - rightdown_x)
                dist_yb = (float(y) - rightdown_y)
                # 行和列分别是y和x
                dxt_map[y, x] = dist_xt
                dyt_map[y, x] = dist_yt
                dxb_map[y, x] = dist_xb
                dyb_map[y, x] = dist_yb
    return dist_map

def init_mask_map(bboxes, ratio=0.3):
    mask_map = torch.zeros([1, 60, 60], dtype=torch.float32)
    for bbox in bboxes:
        #首先转换到输出特征图的60*60坐标空间
        leftup_x = bbox[0] / 4.0
        leftup_y = bbox[1] / 4.0
        rightdown_x = bbox[2] / 4.0
        rightdown_y = bbox[3] / 4.0

        bbox_center_x = float(leftup_x + rightdown_x) * 0.5
        bbox_center_y = float(leftup_y + rightdown_y) * 0.5
        bbox_w = rightdown_x - leftup_x
        bbox_h = rightdown_y - leftup_y

        org_x = int(bbox_center_x - float(ratio * bbox_w * 0.5) + 0.5)
        org_y = int(bbox_center_y - float(ratio * bbox_h * 0.5) + 0.5)
        end_x = int(float(org_x) + float(ratio * bbox_w) + 0.5)
        end_y = int(float(org_y) + float(ratio * bbox_h) + 0.5)
        mask_map[:, org_y: end_y + 1, org_x: end_x + 1] = 1.0
    return mask_map

def train_online(root,
                 num_epoch=100,
                 batch_size = 8,
                 lambda_cls=1.0,
                 lambda_loc=3.0,
                 base_lr=1e-5,
                 resume=None):
    #数据读取
    train_set =  DenseBoxDatasetOnline(root)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               collate_fn = collate_fn_customer
                                               )


    # 网络初始化
    vgg19_pretrain = torchvision.models.vgg19()
    vgg19_pretrain.load_state_dict(torch.load('vgg19.pth'))
    net = DenseBox(vgg19=vgg19_pretrain).to(device)
    #print('=> net:\n', net)
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Note]: invalid resume path @ %s, resume failed.' % resume)
    # 损失函数设置
    loss_func = nn.MSELoss(reduce=False).to(device)


    # 优化策略
    """
    optimizer = torch.optim.Adam(net.parameters(),lr = base_lr)
    """
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=base_lr,
                                momentum=9e-1,
                                weight_decay=5e-8)  # 5e-4 or 5e-8
    
    # 设置网络为训练模式
    print('\n训练中...')
    net.train()

    print('初始学习率为:', base_lr)

    for epoch_i in range(num_epoch):
        # 每一轮训练
        epoch_loss = []
        vis = visdom.Visdom()
        for batch_idx, batch in enumerate(train_loader):
            #初始化图像,同时放到GPU上
            img = batch[0]
            img = img.to(device)

            #初始化GT_maps,同时将它们放到GPU上
            bboxes_batch = batch[1]
            score_maps = []
            dist_maps = []
            mask_maps = []
            for bboxes_img in bboxes_batch:
                score_map = init_score_map(bboxes=bboxes_img, ratio=0.3)
                score_maps.append(score_map)
                dist_map = init_dist_map(bboxes=bboxes_img, ratio=0.3)
                dist_maps.append(dist_map)
                mask_map = score_map.clone()
                mask_maps.append(mask_map)
            cls_maps_gt = torch.stack(score_maps).to(device)
            loc_maps_gt = torch.stack(dist_maps).to(device)
            mask_maps = torch.stack(mask_maps).to(device)

            #清空梯度
            optimizer.zero_grad()

            #前向传播
            score_out, loc_out = net.forward(img)
            #score_out [batchsize,1,60,60]
            #loc_out   [batchsize,4,60,60]
            
            #分类损失
            cls_loss = loss_func(score_out, cls_maps_gt)
            
            #定位损失
            bbox_loc_loss = loss_func(loc_out, loc_maps_gt)

            #负样本挖掘

            pos_indices = torch.nonzero(cls_maps_gt)
            positive_num = pos_indices.size(0)

            #保证正负样本比为1
            #注:img.size(0)即batch_size,这里使用batch_size会在最后不足一个完整batch时报错
            neg_num = int(float(positive_num) / float(img.size(0)) + 0.5)
            ones_mask = torch.ones([img.size(0), 1, 60, 60],
                                   dtype=torch.float32).to(device)
            neg_mask = ones_mask - cls_maps_gt
            neg_cls_loss = cls_loss * neg_mask

            #一半负样本挖掘获得,一半从负样本中随机采样
            half_neg_num = int(neg_num * 0.5 + 0.5)
            neg_cls_loss = neg_cls_loss.view(img.size(0), -1)
            hard_negs, hard_neg_indices = torch.topk(input=neg_cls_loss,
                                                     k=half_neg_num,
                                                     dim=1)
            rand_neg_indices = torch.zeros([img.size(0), half_neg_num],
                                           dtype=torch.long).to(device)
            
            #可改进：这里的随机采样,并不是从负样本中采样,而是全体样本中随机
            for i in range(img.size(0)):
                indices = np.random.choice(3600, #60*60
                                           half_neg_num,
                                           replace=False)
                indices = torch.Tensor(indices)
                rand_neg_indices[i] = indices
            #汇总负样本indices
            neg_indices = torch.cat((hard_neg_indices,
                                     rand_neg_indices),
                                    dim=1)
            neg_indices = neg_indices.cpu()
            pos_indices = pos_indices.cpu()
            
            #更新mask_map,用于确定哪些样本拿来计算损失
            mask_by_sel(loss_mask=mask_maps,
                        pos_indices=pos_indices,
                        neg_indices=neg_indices)
            
            #设置grayzone
            mask_gray_zone_cls_pn(loss_mask=mask_maps,
                                  bboxes=torch.squeeze(torch.FloatTensor(bboxes_batch)),
                                  labels=torch.FloatTensor([1.0]),
                                  ratio=0.3,
                                  gray_border=2.0)
            
            #计算最终的损失
            mask_cls_loss = mask_maps * cls_loss    #分类损失
            mask_bbox_loc_loss = mask_maps * cls_maps_gt * bbox_loc_loss    #定位损失
            full_loss = lambda_cls * (torch.sum(mask_cls_loss)
                        + lambda_loc * torch.sum(mask_bbox_loc_loss))

            # 记录当前epoch当前batch损失
            epoch_loss.append(full_loss.item())

            # 反向传播
            full_loss.backward()
            optimizer.step()

            # 损失的日志信息
            iter_count = epoch_i * len(train_loader) + batch_idx
            if iter_count % 10 == 0:
                print('=> 第{}轮epoch进程：{:>3d}/{:>3d},已迭代{:>5d}次|当前损失为{:>5.3f}'
                      .format(epoch_i + 1,batch_idx,len(train_loader),iter_count,full_loss.item()))
                #vis.heatmap(cls_maps_gt[0,0])
                #vis.heatmap(score_out[0,0])
            
        # 当前epoch的平均损失
        print('=> 第 %d 轮epoch的平均损失: %.3f'% (epoch_i + 1, sum(epoch_loss) / len(epoch_loss)))
        if (epoch_i+1)%5 == 0:
            # 保存checkpoint文件
            torch.save(net.state_dict(), './checkpoints/chepai/temp{}.pth'.format(epoch_i+1))
            print('临时checkpoint文件: {} 已经保存.'.format(root+'checkpoints/chepai/temp{}.pth'.format(epoch_i+1)))
            
        # 调整下一轮的学习率
        lr = adjust_LR(optimizer=optimizer,
                    epoch=epoch_i)
        print('=> 当前学习率为: ', lr)
  
    torch.save(net.state_dict(), './checkpoints/chepai/final.pth')
    print('<=最终的checkpoint文件: {} 已经保存.\n'.format('./checkpoints/chepai/final.pth'))




if __name__ == "__main__":
    train_online(root,batch_size=4,base_lr=1e-8,resume=None)