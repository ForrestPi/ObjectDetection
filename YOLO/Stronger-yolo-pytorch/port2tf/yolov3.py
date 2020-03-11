# coding:utf-8

import numpy as np
import tensorflow as tf
from layers import *
from MobilenetV2 import MobilenetV2,MobilenetV2_dynamic

class YOLOV3(object):
    def __init__(self, training,numcls=20):
        self.__training = training
        self.__num_classes = numcls
        self.__strides=[8,16,32]
    def build_nework(self, input_data, val_reuse=False,gt_per_grid=3):
        """
        :param input_data: shape为(batch_size, input_size, input_size, 3)
        :return: conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox
        conv_sbbox的shape为(batch_size, input_size / 8, input_size / 8, gt_per_grid * (5 + num_classes))
        conv_mbbox的shape为(batch_size, input_size / 16, input_size / 16, gt_per_grid * (5 + num_classes))
        conv_lbbox的shape为(batch_size, input_size / 32, input_size / 32, gt_per_grid * (5 + num_classes))
        conv_?是YOLO的原始卷积输出(raw_dx, raw_dy, raw_dw, raw_dh, raw_conf, raw_prob)
        pred_sbbox的shape为(batch_size, input_size / 8, input_size / 8, gt_per_grid, 5 + num_classes)
        pred_mbbox的shape为(batch_size, input_size / 16, input_size / 16, gt_per_grid, 5 + num_classes)
        pred_lbbox的shape为(batch_size, input_size / 32, input_size / 32, gt_per_grid, 5 + num_classes)
        pred_?是YOLO预测bbox的信息(x, y, w, h, conf, prob)，(x, y, w, h)的大小是相对于input_size的
        """
        net_name = 'YoloV3'
        with tf.variable_scope(net_name, reuse=val_reuse):
            feature_map_s, feature_map_m, feature_map_l = MobilenetV2(input_data, self.__training)
            #jiangwei
            conv = convolutional(name='conv0', input_data=feature_map_l, filters_shape=(1, 1, 1280, 512),
                                 training=self.__training)
            conv = separable_conv(name='conv1', input_data=conv, input_c=512, output_c=1024, training=self.__training)
            conv = convolutional(name='conv2', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                 training=self.__training)
            conv = separable_conv(name='conv3', input_data=conv, input_c=512, output_c=1024, training=self.__training)
            conv = convolutional(name='conv4', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                 training=self.__training)

            # ----------**********---------- Detection branch of large object ----------**********----------
            conv_lbbox = separable_conv(name='conv5', input_data=conv, input_c=512, output_c=1024,
                                        training=self.__training)
            conv_lbbox = convolutional(name='conv6', input_data=conv_lbbox,
                                       filters_shape=(1, 1, 1024, gt_per_grid * (self.__num_classes + 5)),
                                       training=self.__training, downsample=False, activate=False, bn=False)
            pred_lbbox = decode(name='pred_lbbox', conv_output=conv_lbbox,
                                num_classes=self.__num_classes, stride=self.__strides[2])
            # ----------**********---------- Detection branch of large object ----------**********----------

            # ----------**********---------- up sample and merge features map ----------**********----------
            conv = convolutional(name='conv7', input_data=conv, filters_shape=(1, 1, 512, 256),
                                 training=self.__training)
            conv = upsample(name='upsample0', input_data=conv)
            conv = route(name='route0', previous_output=feature_map_m, current_output=conv)
            # ----------**********---------- up sample and merge features map ----------**********----------

            conv = convolutional('conv8', input_data=conv, filters_shape=(1, 1, 96 + 256, 256),
                                 training=self.__training)
            conv = separable_conv('conv9', input_data=conv, input_c=256, output_c=512, training=self.__training)
            conv = convolutional('conv10', input_data=conv, filters_shape=(1, 1, 512, 256),
                                 training=self.__training)
            conv = separable_conv('conv11', input_data=conv, input_c=256, output_c=512, training=self.__training)
            conv = convolutional('conv12', input_data=conv, filters_shape=(1, 1, 512, 256),
                                 training=self.__training)

            # ----------**********---------- Detection branch of middle object ----------**********----------
            conv_mbbox = separable_conv(name='conv13', input_data=conv, input_c=256, output_c=512,
                                        training=self.__training)
            conv_mbbox = convolutional(name='conv14', input_data=conv_mbbox,
                                       filters_shape=(1, 1, 512, gt_per_grid * (self.__num_classes + 5)),
                                       training=self.__training, downsample=False, activate=False, bn=False)
            pred_mbbox = decode(name='pred_mbbox', conv_output=conv_mbbox,
                                num_classes=self.__num_classes, stride=self.__strides[1])
            # ----------**********---------- Detection branch of middle object ----------**********----------

            # ----------**********---------- up sample and merge features map ----------**********----------
            conv = convolutional(name='conv15', input_data=conv, filters_shape=(1, 1, 256, 128),
                                 training=self.__training)
            conv = upsample(name='upsample1', input_data=conv)
            conv = route(name='route1', previous_output=feature_map_s, current_output=conv)
            # ----------**********---------- up sample and merge features map ----------**********----------

            conv = convolutional(name='conv16', input_data=conv, filters_shape=(1, 1, 32 + 128, 128),
                                 training=self.__training)
            conv = separable_conv(name='conv17', input_data=conv, input_c=128, output_c=256, training=self.__training)
            conv = convolutional(name='conv18', input_data=conv, filters_shape=(1, 1, 256, 128),
                                 training=self.__training)
            conv = separable_conv(name='conv19', input_data=conv, input_c=128, output_c=256, training=self.__training)
            conv = convolutional(name='conv20', input_data=conv, filters_shape=(1, 1, 256, 128),
                                 training=self.__training)

            # ----------**********---------- Detection branch of small object ----------**********----------
            conv_sbbox = separable_conv(name='conv21', input_data=conv, input_c=128, output_c=256,
                                        training=self.__training)
            conv_sbbox = convolutional(name='conv22', input_data=conv_sbbox,
                                       filters_shape=(1, 1, 256, gt_per_grid * (self.__num_classes + 5)),
                                       training=self.__training, downsample=False, activate=False, bn=False)
            pred_sbbox = decode(name='pred_sbbox', conv_output=conv_sbbox,
                                num_classes=self.__num_classes, stride=self.__strides[0])
            # ----------**********---------- Detection branch of small object ----------**********----------
        for var in tf.global_variables(net_name):
            tf.add_to_collection(net_name, var)
        return conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox

    def build_network_dynamic(self, input_data,statedict,val_reuse=False,inputsize=544,gt_per_grid=3):
        net_name = 'YoloV3'
        with tf.variable_scope(net_name, reuse=val_reuse):
            feature_map_s, feature_map_m, feature_map_l = MobilenetV2_dynamic(input_data, self.__training,statedict)

            conv = convolutional(name='conv0', input_data=feature_map_l, filters_shape=(1, 1, 1280, 512),
                                 training=self.__training,statedict=statedict['headslarge.conv0'])
            conv = separable_conv(name='conv1', input_data=conv, input_c=512, output_c=1024, training=self.__training,statedict=statedict['headslarge.conv1'])
            conv = convolutional(name='conv2', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                 training=self.__training,statedict=statedict['headslarge.conv2'])
            conv = separable_conv(name='conv3', input_data=conv, input_c=512, output_c=1024, training=self.__training,statedict=statedict['headslarge.conv3'])
            conv = convolutional(name='conv4', input_data=conv, filters_shape=(1, 1, 1024, 512),
                                 training=self.__training,statedict=statedict['headslarge.conv4'])

            # ----------**********---------- Detection branch of large object ----------**********----------
            conv_lbbox = separable_conv(name='conv5', input_data=conv, input_c=512, output_c=1024,
                                        training=self.__training,statedict=statedict['detlarge.conv5'])
            conv_lbbox = convolutional(name='conv6', input_data=conv_lbbox,
                                       filters_shape=(1, 1, 1024, gt_per_grid * (self.__num_classes + 5)),
                                       training=self.__training, downsample=False, activate=False, bn=False,statedict=statedict['detlarge.conv6'])
            pred_lbbox = decode_validate(name='pred_lbbox', conv_output=conv_lbbox,
                                    num_classes=self.__num_classes, stride=self.__strides[2], shape=inputsize // 32,
                                    gt_pergrid=gt_per_grid)
            # ----------**********---------- Detection branch of large object ----------**********----------

            # ----------**********---------- up sample and merge features map ----------**********----------
            conv = convolutional(name='conv7', input_data=conv, filters_shape=(1, 1, 512, 256),
                                 training=self.__training,statedict=statedict['mergelarge.conv7'])
            conv = upsample_decode(name='upsample0', input_data=conv,shape1=inputsize//32,shape2=inputsize//32)
            conv = route(name='route0', previous_output=feature_map_m, current_output=conv)
            # ----------**********---------- up sample and merge features map ----------**********----------

            conv = convolutional('conv8', input_data=conv, filters_shape=(1, 1, 96 + 256, 256),
                                 training=self.__training,statedict=statedict['headsmid.conv8'])
            conv = separable_conv('conv9', input_data=conv, input_c=256, output_c=512, training=self.__training,statedict=statedict['headsmid.conv9'])
            conv = convolutional('conv10', input_data=conv, filters_shape=(1, 1, 512, 256),
                                 training=self.__training,statedict=statedict['headsmid.conv10'])
            conv = separable_conv('conv11', input_data=conv, input_c=256, output_c=512, training=self.__training,statedict=statedict['headsmid.conv11'])
            conv = convolutional('conv12', input_data=conv, filters_shape=(1, 1, 512, 256),
                                 training=self.__training,statedict=statedict['headsmid.conv12'])

            # ----------**********---------- Detection branch of middle object ----------**********----------
            conv_mbbox = separable_conv(name='conv13', input_data=conv, input_c=256, output_c=512,
                                        training=self.__training,statedict=statedict['detmid.conv13'])
            conv_mbbox = convolutional(name='conv14', input_data=conv_mbbox,
                                       filters_shape=(1, 1, 512, gt_per_grid * (self.__num_classes + 5)),
                                       training=self.__training, downsample=False, activate=False, bn=False,statedict=statedict['detmid.conv14'])
            pred_mbbox = decode_validate(name='pred_mbbox', conv_output=conv_mbbox,
                                    num_classes=self.__num_classes, stride=self.__strides[1], shape=inputsize // 16,
                                    gt_pergrid=gt_per_grid)
            # ----------**********---------- Detection branch of middle object ----------**********----------

            # ----------**********---------- up sample and merge features map ----------**********----------
            conv = convolutional(name='conv15', input_data=conv, filters_shape=(1, 1, 256, 128),
                                 training=self.__training,statedict=statedict['mergemid.conv15'])
            conv = upsample_decode(name='upsample1', input_data=conv,shape1=inputsize//16,shape2=inputsize//16)
            conv = route(name='route1', previous_output=feature_map_s, current_output=conv)
            # ----------**********---------- up sample and merge features map ----------**********----------

            conv = convolutional(name='conv16', input_data=conv, filters_shape=(1, 1, 32 + 128, 128),
                                 training=self.__training,statedict=statedict['headsmall.conv16'])
            conv = separable_conv(name='conv17', input_data=conv, input_c=128, output_c=256, training=self.__training,statedict=statedict['headsmall.conv17'])
            conv = convolutional(name='conv18', input_data=conv, filters_shape=(1, 1, 256, 128),
                                 training=self.__training,statedict=statedict['headsmall.conv18'])
            conv = separable_conv(name='conv19', input_data=conv, input_c=128, output_c=256, training=self.__training,statedict=statedict['headsmall.conv19'])
            conv = convolutional(name='conv20', input_data=conv, filters_shape=(1, 1, 256, 128),
                                 training=self.__training,statedict=statedict['headsmall.conv20'])

            # ----------**********---------- Detection branch of small object ----------**********----------
            conv_sbbox = separable_conv(name='conv21', input_data=conv, input_c=128, output_c=256,
                                        training=self.__training,statedict=statedict['detsmall.conv21'])
            conv_sbbox = convolutional(name='conv22', input_data=conv_sbbox,
                                       filters_shape=(1, 1, 256, gt_per_grid * (self.__num_classes + 5)),
                                       training=self.__training, downsample=False, activate=False, bn=False,statedict=statedict['detsmall.conv22'])
            pred_sbbox = decode_validate(name='pred_sbbox', conv_output=conv_sbbox,
                                    num_classes=self.__num_classes, stride=self.__strides[0], shape=inputsize // 8,
                                    gt_pergrid=gt_per_grid)
            # ----------**********---------- Detection branch of small object ----------**********----------
            pred_sbbox = tf.reshape(pred_sbbox, (-1, 5 + self.__num_classes))
            pred_mbbox = tf.reshape(pred_mbbox, (-1, 5 + self.__num_classes))
            pred_lbbox = tf.reshape(pred_lbbox, (-1, 5 + self.__num_classes))
            pred_bbox = tf.concat([pred_sbbox, pred_mbbox, pred_lbbox], 0, name='output/boxconcat')
        for var in tf.global_variables(net_name):
            tf.add_to_collection(net_name, var)
        return pred_bbox
