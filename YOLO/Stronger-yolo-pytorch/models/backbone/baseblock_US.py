import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch


class USconv_bn(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding, activate='relu6',us=[True,True]):
        super().__init__()
        if activate == 'relu6':
            self.convbn = nn.Sequential(OrderedDict([
                ('conv', USConv2d(inp, oup, kernel, stride, padding, bias=False,us=us)),
                ('bn', USBatchNorm2d(oup)),
                ('relu', nn.ReLU6(inplace=True))
            ]))
        elif activate == 'leaky':
            self.convbn = nn.Sequential(OrderedDict([
                ('conv', USConv2d(inp, oup, kernel, stride, padding, bias=False,us=us)),
                ('bn', USBatchNorm2d(oup)),
                ('relu', nn.LeakyReLU(0.1))
            ]))
        else:
            raise AttributeError("activate type not supported")

    def forward(self, input):
        return self.convbn(input)


class ASFF_US(nn.Module):
    def __init__(self, level, activate, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = conv_bn(256, self.inter_dim, kernel=3, stride=2, padding=1, activate=activate)
            self.stride_level_2 = conv_bn(128, self.inter_dim, kernel=3, stride=2, padding=1, activate=activate)
            self.expand = conv_bn(self.inter_dim, 512, kernel=3, stride=1, padding=1, activate=activate)
        elif level == 1:
            self.compress_level_0 = conv_bn(512, self.inter_dim, kernel=1, stride=1, padding=0, activate=activate)
            self.stride_level_2 = conv_bn(128, self.inter_dim, kernel=3, stride=2, padding=1, activate=activate)
            self.expand = conv_bn(self.inter_dim, 256, kernel=3, stride=1, padding=1, activate=activate)
        elif level == 2:
            self.compress_level_0 = conv_bn(512, self.inter_dim, kernel=1, stride=1, padding=0, activate=activate)
            self.compress_level_1= conv_bn(256,self.inter_dim,kernel=1,stride=1,padding=0,activate=activate)
            self.expand = conv_bn(self.inter_dim, 128, kernel=3, stride=1, padding=1, activate=activate)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = conv_bn(self.inter_dim, compress_c, 1, 1, 0, activate=activate)
        self.weight_level_1 = conv_bn(self.inter_dim, compress_c, 1, 1, 0, activate=activate)
        self.weight_level_2 = conv_bn(self.inter_dim, compress_c, 1, 1, 0, activate=activate)

        self.weight_levels = conv_bias(compress_c * 3, 3, kernel=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class USconv_bias(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding,us=[True,True]):
        super().__init__()
        self.conv = USConv2d(inp, oup, kernel, stride, padding, bias=True,us=us)

    def forward(self, input):
        return self.conv(input)


class USsepconv_bn(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding, seprelu):
        super().__init__()
        if seprelu:
            self.sepconv_bn = nn.Sequential(OrderedDict([
                ('sepconv', USConv2d(inp, inp, kernel, stride, padding, groups=inp, bias=False,depthwise=True)),
                ('sepbn', USBatchNorm2d(inp)),
                ('seprelu', nn.ReLU6(inplace=True)),
                ('pointconv', USConv2d(inp, oup, 1, 1, 0, bias=False)),
                ('pointbn', USBatchNorm2d(oup)),
                ('pointrelu', nn.ReLU6(inplace=True)),
            ]))
        else:
            self.sepconv_bn = nn.Sequential(OrderedDict([
                ('sepconv', USConv2d(inp, inp, kernel, stride, padding, groups=inp, bias=False,depthwise=True)),
                ('sepbn', USBatchNorm2d(inp)),
                ('pointconv', USConv2d(inp, oup, 1, 1, 0, bias=False)),
                ('pointbn', USBatchNorm2d(oup)),
                ('pointrelu', nn.ReLU6(inplace=True)),
            ]))

    def forward(self, input):
        return self.sepconv_bn(input)


class USInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(OrderedDict([
                ('dw_conv', USConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False,depthwise=True)),
                ('dw_bn', USBatchNorm2d(hidden_dim)),
                ('dw_relu', nn.ReLU6(inplace=True)),
                ('project_conv', USConv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                ('project_bn', USBatchNorm2d(oup))
            ]))
        else:
            self.conv = nn.Sequential(OrderedDict(
                [
                    ('expand_conv', USConv2d(inp, hidden_dim, 1, 1, 0, bias=False)),
                    ('expand_bn', USBatchNorm2d(hidden_dim)),
                    ('expand_relu', nn.ReLU6(inplace=True)),
                    ('dw_conv', USConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False,depthwise=True)),
                    ('dw_bn', USBatchNorm2d(hidden_dim)),
                    ('dw_relu', nn.ReLU6(inplace=True)),
                    ('project_conv', USConv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                    ('project_bn', USBatchNorm2d(oup))
                ]
            )
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DarknetBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(DarknetBlock, self).__init__()
        self.darkblock = nn.Sequential(OrderedDict([
            ('conv1', USConv2d(inplanes, planes[0], kernel_size=1,
                                stride=1, padding=0, bias=False)),
            ('bn1', USBatchNorm2d(planes[0])),
            ('relu1', nn.LeakyReLU(0.1)),
            ('project_conv', USConv2d(planes[0], planes[1], kernel_size=3,
                                       stride=1, padding=1, bias=False)),
            ('project_bn', USBatchNorm2d(planes[1])),
            ('project_relu', nn.LeakyReLU(0.1)),
        ]))

    def forward(self, x):
        out = self.darkblock(x)
        out += x
        return out


def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1]):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.us = us
        self.ratio = ratio

    def forward(self, input):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0]) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1]) * self.ratio[1]
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        # if getattr(FLAGS, 'conv_averaged', False):
        #     y = y * (max(self.in_channels_list)/self.in_channels)
        return y




class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=True)
        self.num_features_max = num_features
        # for tracking performance during training
        self.bn_notsave = nn.ModuleList(
            [nn.BatchNorm2d(i, affine=False)
             for i in [
                     make_divisible(
                         self.num_features_max * width_mult / ratio) * ratio
                     for width_mult in [0.4,1.0]]
             ]
        )
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        c=input.shape[1]
        # y = nn.functional.batch_norm(
        #     input,
        #     self.running_mean[:c],
        #     self.running_var[:c],
        #     weight[:c],
        #     bias[:c],
        #     self.training,
        #     self.momentum,
        #     self.eps)
        if self.width_mult in [0.4,1.0]:
            idx = [0.4,1.0].index(self.width_mult)
            y = nn.functional.batch_norm(
                input,
                self.bn_notsave[idx].running_mean[:c],
                self.bn_notsave[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        #     # print(self.bn[0].running_mean.shape,self.bn[0].running_mean.sum())
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean[:c],
                self.running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y

def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        # if getattr(FLAGS, 'cumulative_bn_stats', False):
        #     m.momentum = None

if __name__ == '__main__':
    model=ASFF(1,activate='leaky')
    l1=torch.ones(1,512,10,10)
    l2=torch.ones(1,256,20,20)
    l3=torch.ones(1,128,40,40)
    out=model(l1,l2,l3)
    print(out.shape)