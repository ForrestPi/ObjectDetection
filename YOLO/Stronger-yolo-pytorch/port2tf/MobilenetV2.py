# coding: utf-8

from layers import *

def MobilenetV2(input_data, training):
    with tf.variable_scope('MobilenetV2'):
        conv = convolutional(name='Conv', input_data=input_data, filters_shape=(3, 3, 3, 32),
                             training=training, downsample=True, activate=True, bn=True)
        conv = inverted_residual(name='expanded_conv', input_data=conv, input_c=32, output_c=16,
                                 training=training, t=1)

        conv = inverted_residual(name='expanded_conv_1', input_data=conv, input_c=16, output_c=24, downsample=True,
                                 training=training)
        conv = inverted_residual(name='expanded_conv_2', input_data=conv, input_c=24, output_c=24, training=training)

        conv = inverted_residual(name='expanded_conv_3', input_data=conv, input_c=24, output_c=32, downsample=True,
                                 training=training)
        conv = inverted_residual(name='expanded_conv_4', input_data=conv, input_c=32, output_c=32, training=training)
        feature_map_s = inverted_residual(name='expanded_conv_5', input_data=conv, input_c=32, output_c=32,
                                          training=training)

        conv = inverted_residual(name='expanded_conv_6', input_data=feature_map_s, input_c=32, output_c=64,
                                 downsample=True, training=training)
        conv = inverted_residual(name='expanded_conv_7', input_data=conv, input_c=64, output_c=64, training=training)
        conv = inverted_residual(name='expanded_conv_8', input_data=conv, input_c=64, output_c=64, training=training)
        conv = inverted_residual(name='expanded_conv_9', input_data=conv, input_c=64, output_c=64, training=training)

        conv = inverted_residual(name='expanded_conv_10', input_data=conv, input_c=64, output_c=96, training=training)
        conv = inverted_residual(name='expanded_conv_11', input_data=conv, input_c=96, output_c=96, training=training)
        feature_map_m = inverted_residual(name='expanded_conv_12', input_data=conv, input_c=96, output_c=96,
                                          training=training)

        conv = inverted_residual(name='expanded_conv_13', input_data=feature_map_m, input_c=96, output_c=160,
                                 downsample=True, training=training)
        conv = inverted_residual(name='expanded_conv_14', input_data=conv, input_c=160, output_c=160, training=training)
        conv = inverted_residual(name='expanded_conv_15', input_data=conv, input_c=160, output_c=160, training=training)

        conv = inverted_residual(name='expanded_conv_16', input_data=conv, input_c=160, output_c=320, training=training)

        feature_map_l = convolutional(name='Conv_1', input_data=conv, filters_shape=(1, 1, 320, 1280),
                                      training=training, downsample=False, activate=True, bn=True)
    return feature_map_s, feature_map_m, feature_map_l


def MobilenetV2_dynamic(input_data, training, statedict):
    with tf.variable_scope('MobilenetV2'):
        conv = convolutional(name='Conv', input_data=input_data, filters_shape=(3, 3, 3, 32),
                             training=training, downsample=True, activate=True, bn=True,
                             statedict=statedict['features.0'])
        conv = inverted_residual(name='expanded_conv', input_data=conv, input_c=32, output_c=16,
                                 training=training, t=1, statedict=statedict['features.1'])

        conv = inverted_residual(name='expanded_conv_1', input_data=conv, input_c=16, output_c=24, downsample=True,
                                 training=training, statedict=statedict['features.2'])
        conv = inverted_residual(name='expanded_conv_2', input_data=conv, input_c=24, output_c=24, training=training,
                                 statedict=statedict['features.3'])

        conv = inverted_residual(name='expanded_conv_3', input_data=conv, input_c=24, output_c=32, downsample=True,
                                 training=training, statedict=statedict['features.4'])
        conv = inverted_residual(name='expanded_conv_4', input_data=conv, input_c=32, output_c=32, training=training,
                                 statedict=statedict['features.5'])
        feature_map_s = inverted_residual(name='expanded_conv_5', input_data=conv, input_c=32, output_c=32,
                                          training=training, statedict=statedict['features.6'])

        conv = inverted_residual(name='expanded_conv_6', input_data=feature_map_s, input_c=32, output_c=64,
                                 downsample=True, training=training, statedict=statedict['features.7'])
        conv = inverted_residual(name='expanded_conv_7', input_data=conv, input_c=64, output_c=64, training=training,
                                 statedict=statedict['features.8'])
        conv = inverted_residual(name='expanded_conv_8', input_data=conv, input_c=64, output_c=64, training=training,
                                 statedict=statedict['features.9'])
        conv = inverted_residual(name='expanded_conv_9', input_data=conv, input_c=64, output_c=64, training=training,
                                 statedict=statedict['features.10'])

        conv = inverted_residual(name='expanded_conv_10', input_data=conv, input_c=64, output_c=96, training=training,
                                 statedict=statedict['features.11'])
        conv = inverted_residual(name='expanded_conv_11', input_data=conv, input_c=96, output_c=96, training=training,
                                 statedict=statedict['features.12'])
        feature_map_m = inverted_residual(name='expanded_conv_12', input_data=conv, input_c=96, output_c=96,
                                          training=training, statedict=statedict['features.13'])

        conv = inverted_residual(name='expanded_conv_13', input_data=feature_map_m, input_c=96, output_c=160,
                                 downsample=True, training=training, statedict=statedict['features.14'])
        conv = inverted_residual(name='expanded_conv_14', input_data=conv, input_c=160, output_c=160, training=training,
                                 statedict=statedict['features.15'])
        conv = inverted_residual(name='expanded_conv_15', input_data=conv, input_c=160, output_c=160, training=training,
                                 statedict=statedict['features.16'])

        conv = inverted_residual(name='expanded_conv_16', input_data=conv, input_c=160, output_c=320, training=training,
                                 statedict=statedict['features.17'])

        feature_map_l = convolutional(name='Conv_1', input_data=conv, filters_shape=(1, 1, 320, 1280),
                                      training=training, downsample=False, activate=True, bn=True,
                                      statedict=statedict['features.18'])
    return feature_map_s, feature_map_m, feature_map_l
