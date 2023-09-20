import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(EncoderBlock, self).__init__()

        self.relu = nn.LeakyReLU(inplace=True)
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels)
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = nn.ConvTranspose3d(
            in_ch, in_ch // 2, kernel_size=2, stride=2, bias=False
        )
        self.norm1 = nn.InstanceNorm3d(in_ch // 2)

        self.conv2 = nn.Conv3d(
            in_ch // 2, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.InstanceNorm3d(out_ch)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        out = self.relu(x)

        return out


class NormalBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(NormalBlock, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = nn.Conv3d(
            in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm1 = nn.InstanceNorm3d(in_ch // 2)

        self.conv2 = nn.Conv3d(
            in_ch // 2, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.InstanceNorm3d(out_ch)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        out = self.relu(x)

        return out
