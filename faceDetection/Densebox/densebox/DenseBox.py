# coding=utf-8

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

#注：没有BN层

# ------------------------------- network

class DenseBox(torch.nn.Module):
    """
    implemention of densebox network with py-torch
    """

    def __init__(self, vgg19):
        """
        init the first 12 layers with pre-trained weights
        :param vgg19: pre-trained net
        """
        super(DenseBox, self).__init__()

        feats = vgg19.features._modules

        # print(feats), print(classifier)

        # ----------------- Conv1
        self.conv1_1_1 = copy.deepcopy(feats['0'])  # (0)
        self.conv1_1_2 = copy.deepcopy(feats['1'])  # (1)
        self.conv1_1 = nn.Sequential(
            self.conv1_1_1,
            self.conv1_1_2
        )  # conv_layer1

        self.conv1_2_1 = copy.deepcopy(feats['2'])  # (2) conv_layer2
        self.conv1_2_2 = copy.deepcopy(feats['3'])  # (3)
        self.conv1_2 = nn.Sequential(
            self.conv1_2_1,
            self.conv1_2_2
        )  # conv_layer2

        self.pool1 = copy.deepcopy(feats['4'])  # (4)

        # ----------------- Conv2
        self.conv2_1_1 = copy.deepcopy(feats['5'])  # (5)
        self.conv2_1_2 = copy.deepcopy(feats['6'])  # (6)
        self.conv2_1 = nn.Sequential(
            self.conv2_1_1,
            self.conv2_1_2
        )  # conv_layer3

        self.conv2_2_1 = copy.deepcopy(feats['7'])  # (7)
        self.conv2_2_2 = copy.deepcopy(feats['8'])  # (8)
        self.conv2_2 = nn.Sequential(
            self.conv2_2_1,
            self.conv2_2_2
        )  # conv_layer4

        self.pool2 = copy.deepcopy(feats['9'])  # (9)

        # ----------------- Conv3
        self.conv3_1_1 = copy.deepcopy(feats['10'])  # (10)
        self.conv3_1_2 = copy.deepcopy(feats['11'])  # (11)
        self.conv3_1 = nn.Sequential(
            self.conv3_1_1,
            self.conv3_1_2
        )  # conv_layer5

        self.conv3_2_1 = copy.deepcopy(feats['12'])  # (12)
        self.conv3_2_2 = copy.deepcopy(feats['13'])  # (13)
        self.conv3_2 = nn.Sequential(
            self.conv3_2_1,
            self.conv3_2_2
        )  # conv_layer6

        self.conv3_3_1 = copy.deepcopy(feats['14'])  # (14)
        self.conv3_3_2 = copy.deepcopy(feats['15'])  # (15)
        self.conv3_3 = nn.Sequential(
            self.conv3_3_1,
            self.conv3_3_2
        )  # conv_layer7

        self.conv3_4_1 = copy.deepcopy(feats['16'])  # (16)
        self.conv3_4_2 = copy.deepcopy(feats['17'])  # (17)
        self.conv3_4 = nn.Sequential(
            self.conv3_4_1,
            self.conv3_4_2
        )  # conv_layer8

        self.pool3 = copy.deepcopy(feats['18'])  # (18)

        # ----------------- Conv4
        self.conv4_1_1 = copy.deepcopy(feats['19'])  # (19)
        self.conv4_1_2 = copy.deepcopy(feats['20'])  # (20)
        self.conv4_1 = nn.Sequential(
            self.conv4_1_1,
            self.conv4_1_2
        )  # conv_layer9

        self.conv4_2_1 = copy.deepcopy(feats['21'])  # (21)
        self.conv4_2_2 = copy.deepcopy(feats['22'])  # (22)
        self.conv4_2 = nn.Sequential(
            self.conv4_2_1,
            self.conv4_2_2
        )  # conv_layer10

        self.conv4_3_1 = copy.deepcopy(feats['23'])  # (23)
        self.conv4_3_2 = copy.deepcopy(feats['24'])  # (24)
        self.conv4_3 = nn.Sequential(
            self.conv4_3_1,
            self.conv4_3_2
        )  # conv_layer11

        self.conv4_4_1 = copy.deepcopy(feats['25'])  # (25)
        self.conv4_4_2 = copy.deepcopy(feats['26'])  # (26)
        self.conv4_4 = nn.Sequential(
            self.conv4_4_1,
            self.conv4_4_2
        )

        # route: up-sample and concatenate
        # self.up_sampling = nn.Upsample(size=(120, 120),
        #                                mode='bilinear',
        #                                align_corners=True)

        # -------------------------------------- ouput layers
        # scores output
        self.conv5_1_det = nn.Conv2d(in_channels=768,
                                     out_channels=512,
                                     kernel_size=(1, 1))
        self.conv5_2_det = nn.Conv2d(in_channels=512,
                                     out_channels=1,
                                     kernel_size=(1, 1))
        torch.nn.init.xavier_normal_(self.conv5_1_det.weight.data)
        torch.nn.init.xavier_normal_(self.conv5_2_det.weight.data)

        self.output_score = nn.Sequential(
            self.conv5_1_det,
            nn.Dropout(),
            self.conv5_2_det
        )

        # locs output
        self.conv5_1_loc = nn.Conv2d(in_channels=768,
                                     out_channels=512,
                                     kernel_size=(1, 1))
        self.conv5_2_loc = nn.Conv2d(in_channels=512,
                                     out_channels=4,
                                     kernel_size=(1, 1))
        torch.nn.init.xavier_normal_(self.conv5_1_loc.weight.data)
        torch.nn.init.xavier_normal_(self.conv5_2_loc.weight.data)

        self.output_loc = nn.Sequential(
            self.conv5_1_loc,
            nn.Dropout(),
            self.conv5_2_loc
        )

    def forward(self, X):
        """
        :param X:
        :return:
        """
        X = self.conv1_1(X)
        X = self.conv1_2(X)
        X = self.pool1(X)

        X = self.conv2_1(X)
        X = self.conv2_2(X)
        X = self.pool2(X)

        X = self.conv3_1(X)
        X = self.conv3_2(X)
        X = self.conv3_4(X)

        # conv3_4 result
        conv3_4_X = X.clone()
        # conv3_4_X = torch.Tensor.new_tensor(data=X,
        #                                     dtype=torch.float32,
        #                                     device=device,
        #                                     requires_grad=True)

        X = self.pool3(X)

        X = self.conv4_1(X)
        X = self.conv4_2(X)
        X = self.conv4_3(X)
        conv4_4_X = self.conv4_4(X)

        # upsample_X = self.up_sampling
        #  upsample of conv4_4
        conv4_4_X_us = nn.Upsample(size=(conv3_4_X.size(2),
                                         conv3_4_X.size(3)),
                                   mode='bilinear',
                                   align_corners=True)(conv4_4_X)
                                   
        # feature fusion: concatenate along channel axis
        fusion = torch.cat((conv4_4_X_us, conv3_4_X), dim=1)
        # print('=> fusion shape', fusion.shape)

        # output layer
        scores = self.output_score(fusion)
        locs = self.output_loc(fusion)
        # print('=> scores shape: ', scores.shape)
        # print('=> locs shape: ', locs.shape)

        return scores, locs

if __name__ == "__main__":
    # 网络初始化
    import torchvision
    vgg19_pretrain = torchvision.models.vgg19()
    #vgg19_pretrain.load_state_dict(torch.load('vgg19.pth'))
    net = DenseBox(vgg19=vgg19_pretrain)
    input = torch.randn([4,3,240,240])
    scores, locs = net(input)
    print(scores.size())#torch.Size([4, 1, 60, 60])
    print(locs.size()) #torch.Size([4, 4, 60, 60])