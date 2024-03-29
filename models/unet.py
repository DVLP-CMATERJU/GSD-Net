# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding = kernel_size // 2)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding = kernel_size // 2)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))

        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size= 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding = kernel_size // 2)#1st parameter in_size
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding = kernel_size // 2)
        self.activation = activation

    def center_crop(self, layer, target_size):

        batch_size, n_channels, layer_width, layer_height = layer.size()
        x1 = (layer_width - target_size[0]) // 2
        y1 = (layer_height - target_size[1]) // 2
        return layer[:, :, x1:(x1 + target_size[0]), y1:(y1 + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x,output_size=bridge.size())
        crop1 = self.center_crop(bridge, (up.size()[2],up.size()[3]))   # IMPROVEMENT + ALL PADDING
        out = torch.cat([up, crop1], 1)

        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))

        return out

class UNet(nn.Module):
    def __init__(self,no_class):#
        super(UNet, self).__init__()
        #self.imsize = imsize
        self.activation = F.relu

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = UNetConvBlock(3, 64)#3--1
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.conv_block512_1024 = UNetConvBlock(512, 1024)

        self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)

        self.last = nn.Conv2d(64, no_class, 1)#11--2


    def forward(self, x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)

        block5 = self.conv_block512_1024(pool4)

        up1 = self.up_block1024_512(block5, block4)
        up2 = self.up_block512_256(up1, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)

        return self.last(up4)

if  __name__ == "__main__":
    from torchsummary import summary
    model = UNet(10).cuda()
    summary(model, input_size=(3, 360, 480))