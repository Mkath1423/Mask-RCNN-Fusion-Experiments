import torch
import torch.nn as nn
import torchvision
from models import building_blocks


class ResNetEarlyFusion(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.out_channels = num_classes
        # fusion layer
        self.in_conv = building_blocks.BNDoubleConv(in_channels, 64)

        resnet = torchvision.models.resnet18(pretrained=True, progress=True)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.out_conv = building_blocks.OutConv(512, num_classes)

    def forward(self, x):
        x_in = self.in_conv(x)

        x1 = self.layer1(x_in)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = self.out_conv(x4)

        return out


class ResNetMidFusion(nn.Module):
    def __init__(self, c_channels, d_channels, num_classes, concat=True):
        super().__init__()
        self.out_channels = num_classes
        # fusion layer
        self.c_in_conv = building_blocks.BNDoubleConv(c_channels, 64)
        self.d_in_conv = building_blocks.BNDoubleConv(d_channels, 64)

        c_resnet = torchvision.models.resnet18(pretrained=True, progress=True)
        self.c_layer1 = c_resnet.layer1
        self.c_layer2 = c_resnet.layer2

        d_resnet = torchvision.models.resnet18(pretrained=True, progress=True)
        self.d_layer1 = d_resnet.layer1
        self.d_layer2 = d_resnet.layer2

        resnet = torchvision.models.resnet18(pretrained=True, progress=True)
        self.layer4 = resnet.layer4

        self.out_conv = building_blocks.OutConv(512, num_classes)

    def forward(self, x):
        c_in = x[:, 0:3, :, :]
        d_in = x[:, 3:, :, :]

        c0 = self.c_in_conv(c_in)
        d0 = self.d_in_conv(d_in)

        c1 = self.c_layer1(c0)
        d1 = self.d_layer1(d0)

        c2 = self.c_layer2(c1)
        d2 = self.d_layer2(d1)

        fusion = torch.concat([c2, d2], 1)

        x4 = self.layer4(fusion)

        out = self.out_conv(x4)

        return out

class ResNetLateFusion(nn.Module):
    def __init__(self, c_channels, d_channels, num_classes, concat=True):
        super().__init__()
        self.out_channels = num_classes
        # fusion layer
        self.c_in_conv = building_blocks.BNDoubleConv(c_channels, 64)
        self.d_in_conv = building_blocks.BNDoubleConv(d_channels, 64)

        c_resnet = torchvision.models.resnet18(pretrained=True, progress=True)
        self.c_layer1 = c_resnet.layer1
        self.c_layer2 = c_resnet.layer2
        self.c_layer3 = c_resnet.layer3
        self.c_layer4 = c_resnet.layer4

        d_resnet = torchvision.models.resnet18(pretrained=True, progress=True)
        self.d_layer1 = d_resnet.layer1
        self.d_layer2 = d_resnet.layer2
        self.d_layer3 = d_resnet.layer3
        self.d_layer4 = d_resnet.layer4

        self.out_conv = building_blocks.OutConv(512*2, num_classes)

    def forward(self, x):
        c_in = x[:, 0:3, :, :]
        d_in = x[:, 3:, :, :]

        c0 = self.c_in_conv(c_in)
        d0 = self.d_in_conv(d_in)

        c1 = self.c_layer1(c0)
        d1 = self.d_layer1(d0)

        c2 = self.c_layer2(c1)
        d2 = self.d_layer2(d1)

        c3 = self.c_layer3(c2)
        d3 = self.d_layer3(d2)

        c4 = self.c_layer4(c3)
        d4 = self.d_layer4(d3)

        x_out = torch.concat([c4, d4], 1)

        out = self.out_conv(x_out)

        return out


if __name__ == "__main__":
    model = ResNetEarlyFusion(4, 3)
