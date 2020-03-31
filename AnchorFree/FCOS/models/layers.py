# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

        # smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
        return x+y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))  # /2
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1) # /4
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)  # / 8, 512
        c4 = self.layer3(c3)  # / 16, 1024
        c5 = self.layer4(c4)  # / 32, 2048
        
        p6 = self.conv6(c5)   # / 64, 256
        p7 = self.conv7(F.relu(p6)) # /128, 256
        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        # smooth
        p3 = self.smooth1(p3) # /8
        p4 = self.smooth1(p4) # /16
        p5 = self.smooth1(p5) # /32
        p6 = self.smooth1(p6) # /64
        p7 = self.smooth1(p7) # /128
        return p3, p4, p5, p6, p7

def FPN50():
    return FPN(Bottleneck, [3,4,6,3])   

if __name__== "__main__":
    x = torch.randn(1,3,256,256)
    b = Bottleneck(3,64,2)
    n = FPN50()
    y = b(x)#torch.Size([1, 256, 128, 128])
    yy = n(x)
    print(f'x: {x.shape}')
    print(f'y: {y.shape}')
    for i in yy:
        print(i.shape)
    '''
    torch.Size([1, 256, 32, 32])
    torch.Size([1, 256, 16, 16])
    torch.Size([1, 256, 8, 8])
    torch.Size([1, 256, 4, 4])
    torch.Size([1, 256, 2, 2])
    '''
#    print(f'fpn: {[i.shape for i in yy]}')
    
    
    
    
    
    
