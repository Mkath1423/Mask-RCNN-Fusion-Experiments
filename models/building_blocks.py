import torch
from torch import nn
import torch.nn.functional as F

# https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin

__all__=[
    'BNDoubleConv',
    'Down',
    'Up',
    'OutConv'
    ]

# basic BN - C - C block from above
class BNDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bias=False):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# down scale a layer, and then do the BNDoubleConve section
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            BNDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, up_conv = True):
        super().__init__()

        self.up_conv = up_conv

        if up_conv:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = BNDoubleConv(in_channels, out_channels)
        else:
            # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
            # assumed to be of the form minibatch x channels x [optional depth] x [optional height] x width
            # for spatial inputs, we expect a 4D Tensor
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.conv = BNDoubleConv(in_channels, out_channels, in_channels//2)  # the up-conv cuts num channels by 2, but upsampling does not
            # TODO:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=1, bias=False)
            )
            self.conv = BNDoubleConv(in_channels, out_channels)  # the up-conv cuts num channels by 2, but upsampling does not
            


    def forward(self, x1, x2):
        # print("up in x1: ", x1.shape)
        # print("up in x2: ", x2.shape)
        x1 = self.up(x1)  # up sample
        # print("up up x1: ", x1.shape)
        x = torch.cat([x2, x1], dim=1) # concatenate
        # print("up x->c: ", x.shape)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)