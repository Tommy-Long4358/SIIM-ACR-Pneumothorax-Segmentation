import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Save convolution layer for skip connection
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Skip connection
        x = torch.cat([x1, x2], 1)

        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_conv_1 = Downsample(in_channels, 64)
        self.down_conv_2 = Downsample(64, 128)
        self.down_conv_3 = Downsample(128, 256)
        self.down_conv_4 = Downsample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_conv_1 = Upsample(1024, 512)
        self.up_conv_2 = Upsample(512, 256)
        self.up_conv_3 = Upsample(256, 128)
        self.up_conv_4 = Upsample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_conv_1(x)
        down_2, p2 = self.down_conv_2(p1)
        down_3, p3 = self.down_conv_3(p2)
        down_4, p4 = self.down_conv_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_conv_1(b, down_4)
        up_2 = self.up_conv_2(up_1, down_3)
        up_3 = self.up_conv_3(up_2, down_2)
        up_4 = self.up_conv_4(up_3, down_1)

        out = self.out(up_4)
        return out