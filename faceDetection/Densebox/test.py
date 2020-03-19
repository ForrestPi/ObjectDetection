from densebox import DenseBox

import os
import torch
import torchvision
from torch import nn
import numpy as np
from PIL import Image
import cv2

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')



def parse_out_MN(score_map,
                 loc_map,
                 M,
                 N,
                 K=10):
    """
    解析任意尺寸图像的输出结果
    M: image height, M rows
    N: image width, N cols
    """
    assert score_map.size() == torch.Size([1, 1, M // 4, N // 4])  # N×C×H×W
    assert loc_map.size() == torch.Size([1, 4, M // 4, N // 4])

    # 删除batch维度
    score_map, loc_map = score_map.squeeze(), loc_map.squeeze()

    # 调整输出shape, score_map: 1×(M×N), loc_map:4×(M×N)
    score_map, loc_map = score_map.view(1, -1), loc_map.view(4, -1)

    # 找到前k个高置信度
    scores, indices = torch.topk(input=score_map,
                                 k=K,
                                 dim=1)

    indices = indices.squeeze()
    score_map = score_map.squeeze().data

    dets = []
    cols_out = N // 4
    for idx in indices:
        idx = int(idx)
        xi, yi = idx % cols_out, idx // cols_out

        xt = xi - loc_map[0, idx]
        yt = yi - loc_map[1, idx]
        xb = xi - loc_map[2, idx]
        yb = yi - loc_map[3, idx]

        # 调整到输入坐标空间
        xt = float(xt.data) * 4.0
        yt = float(yt.data) * 4.0
        xb = float(xb.data) * 4.0
        yb = float(yb.data) * 4.0

        det = [xt, yt, xb, yb, float(score_map[idx])]
        dets.append(det)

    return np.array(dets)

def NMS(dets,
        nms_thresh=0.4):
    """
    Pure Python NMS baseline
    :param dets:
    :param nms_thresh:
    :return:
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]

    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)

        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= nms_thresh)[0]

        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep

import matplotlib.pyplot as plt
def show(img, bboxes):
    """
    img为PIL的Image格式
    bbox为list
    """
    img = np.array(img)
    img = np.transpose(img,(1,2,0))
    plt.imshow(img)
    #画矩形框
    #print(bboxes)
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

def test(img_path, resume, ):

    #初始化网络
    vgg19_pretrain = torchvision.models.vgg19()
    vgg19_pretrain.load_state_dict(torch.load('vgg19.pth'))
    net = DenseBox(vgg19=vgg19_pretrain).to(device)
    net.load_state_dict(torch.load(resume))
    print('=> 网络从 {} 加载'.format(resume))

    # 网络切换到inference模式
    net.eval()
    net.to(device)

    # 读取图片
    img = Image.open(img_path)
    W, H = img.size
    from torchvision import transforms
    img_tensor = transforms.Compose([transforms.ToTensor()])(img)
    
    # 图片变换
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(H, W)),
        torchvision.transforms.CenterCrop(size=(H, W)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225))
    ])

    img = transform(img)
    img = img.view(1, 3, H, W)
    
    # inference过程(GPU)
    img = img.to(device)
    score_out, loc_out = net.forward(img)

    import visdom
    vis = visdom.Visdom()
    vis.heatmap(loc_out[0,0])
    vis.heatmap(loc_out[0,1])
    vis.heatmap(loc_out[0,2])
    vis.heatmap(loc_out[0,3])
    #vis.heatmap(score_out[0,0])

    # 解析输出,将结果转换为bbox形式
    dets = parse_out_MN(score_map=score_out.cpu(),
                                    loc_map=loc_out.cpu(),
                                    M=H,
                                    N=W,
                                    K=10)
    
    # 非极大抑制
    keep = NMS(dets=dets, nms_thresh=0.4)
    dets = dets[keep]

    # 可视化结果
    show(img_tensor, [[60,101,181,143]])
    show(img_tensor, dets)
    #viz_result(img_path=img_path,
    #            dets=dets,
    #            dst_root='./')
resume = './checkpoints/chepai/temp45.pth'
test('./det_2018_09_13_001527_label_60_101_181_143.jpg',resume)



