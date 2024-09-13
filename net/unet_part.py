import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter

class DoubleConv3d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=1,kernel_size=3):
        super(DoubleConv3d,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding='same', dilation=dilation),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding='same', dilation=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        #print("x size: ", x.size() )
        out = self.double_conv(x)
        #print("DoubleConv size: ", out.size() )
        return out

class Down3d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3,dilation=1):
        super(Down3d,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(in_channels, out_channels, dilation=dilation,kernel_size=kernel_size)
            #RRCNN_block(in_channels, out_channels, dilation=dilation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=3,bilinear=True):
        super(Up3d, self).__init__()
        self.in_channels = in_channels
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2) #, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size = 2, stride = 2)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding="same")
        
        self.pool = nn.AvgPool3d(2)
        
        fea_channels = out_channels +  out_channels
        self.conv = DoubleConv3d(fea_channels, out_channels, dilation=dilation,kernel_size=kernel_size)
        #self.conv = RRCNN_block(fea_channels, out_channels, dilation=dilation)
        #self.se = SELayer(out_channels)
    
    def forward(self, x_decode, x_encode):
        x_decode = self.up(x_decode)
        x_decode = self.conv1(x_decode)
        # input is CHW
        diffY = x_encode.size()[2] - x_decode.size()[2]
        diffX = x_encode.size()[3] - x_decode.size()[3]
        diffZ = x_encode.size()[4] - x_decode.size()[4]
        x_decode = F.pad(x_decode, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat((x_encode, x_decode), dim = 1)
        x = self.conv(x)
        #x = self.se(x)
        return x

class OutConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)