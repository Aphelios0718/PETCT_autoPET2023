import torch
import torch.nn as nn


class CoLearnUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CoLearnUnit, self).__init__()

        self.co_conv = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=(2, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
                bias=False
            ),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, y):
        B, C, D, H, W = x.shape
        z = torch.zeros(size=(B, C, 2 * D, H, W)).to(x.get_device())
        z[:, :, 0::2, :, :] = x
        z[:, :, 1::2, :, :] = y

        z = self.co_conv(z)
        z = self.conv2(z)
        return z

