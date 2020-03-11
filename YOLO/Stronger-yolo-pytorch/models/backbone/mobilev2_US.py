import math
import torch.nn as nn

from models.backbone.baseblock_US import *
from models.backbone.helper import load_mobilev2


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


class USMobileNetV2(nn.Module):
    def __init__(self, out_indices=(6, 13, 18),
                 width_mult=1.):
        super(USMobileNetV2, self).__init__()
        self.backbone_outchannels=[1280,96,32]
        # setting of inverted residual blocks
        self.out_indices = out_indices
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.features = []
        # 1.0 by default
        width_mult = 1.0
        # head
        channels = make_divisible(32 * width_mult)
        self.outp = make_divisible(
            1280 * width_mult) if width_mult > 1.0 else 1280
        self.features.append(
                USconv_bn(3, channels, 3, 2, 1, us=[False, True])
        )

        # body
        for t, c, n, s in self.block_setting:
            outp = make_divisible(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        USInvertedResidual(channels, outp, s, t))
                else:
                    self.features.append(
                        USInvertedResidual(channels, outp, 1, t))
                channels = outp

        # tail
        self.features.append(
            USconv_bn(channels, self.outp, 1, 1, 0,us=[True,True])
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # if FLAGS.reset_parameters:
        #     self.reset_parameters()

    def forward(self, x):
        outs = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


def mobilenetv2(pretrained=None, **kwargs):
    model = USMobileNetV2(width_mult=1.0)
    if pretrained:
        if isinstance(pretrained, str):
            load_mobilev2(model, pretrained)
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


def mobilenetv2_75(pretrained=None, **kwargs):
    model = USMobileNetV2(width_mult=0.75)
    model.backbone_outchannels = [1280, 72, 24]
    if pretrained:
        if isinstance(pretrained, str):
            load_mobilev2(model, pretrained)
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == '__main__':
    from thop import profile
    model = mobilenetv2(pretrained='checkpoints/mobilenet_v2.pth')
    model.apply(lambda m: setattr(m, 'width_mult',0.44))
    inp = torch.ones(1, 3, 320, 320)
    flops, params = profile(model.features[0], inputs=(inp,), verbose=True)
    print(flops,params)
    out = model(inp)
    for o in out:
        print(o.shape)
    # print(model)
