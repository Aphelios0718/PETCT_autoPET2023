
import torch.nn as nn
from models.special_design import *
from models.common_block import *


# baseline model
class UNet_baseline(nn.Module):
    def __init__(self, in_channels, n_seg_classes):
        super(UNet_baseline, self).__init__()

        self.stem = EncoderBlock(in_channels, 32)

        self.encoder_1 = EncoderBlock(32, 64)
        self.encoder_2 = EncoderBlock(64, 128)
        self.encoder_3 = EncoderBlock(128, 256)

        self.decoder_3 = DecoderBlock(256, 128)
        self.decoder_2 = DecoderBlock(128 * 2, 64)
        self.decoder_1 = DecoderBlock(64 * 2, 32)

        self.final_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, n_seg_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input):
        # x = pet modality y = ct modality

        e0 = self.stem(input)  # 32, D, H, W

        e1 = self.encoder_1(e0)  # 64, D/2, H/2, W/2
        e2 = self.encoder_2(e1)  # 128, D/4, H/4, W/4
        e3 = self.encoder_3(e2)  # 256, D/8, H/8, W/8

        d3 = self.decoder_3(e3)  # 256, D/8, H/8, W/8 -> # 128, D/4, H/4, W/4
        d2 = self.decoder_2(
            torch.concatenate((d3, e2), dim=1)
        )  # 128 + 128, D/4, H/4, W/4 -> # 64, D/2, H/2, W/2
        d1 = self.decoder_1(
            torch.concatenate((d2, e1), dim=1)
        )  # 64 + 64, D/2, H/2, W/2 -> # 32, D, H, W

        out = self.final_conv(
            torch.concatenate((d1, e0), dim=1)
        )  # 32 + 32, D, H, W -> # n_seg_classes, D, H, W

        return out


class CoLearnUNet(nn.Module):
    def __init__(self, in_channels, n_seg_classes, version: str):
        super(CoLearnUNet, self).__init__()
  
        self.stem = CoLearnUnit(1, 32)

        self.encoder_1 = EncoderBlock(32, 64, stride=2)
        self.encoder_2 = EncoderBlock(64, 128,stride=2)
        self.encoder_3 = EncoderBlock(128, 256,stride=2)
        self.encoder_4 = EncoderBlock(256, 512,stride=2)

        self.decoder_4 = DecoderBlock(512, 256)
        self.decoder_3 = DecoderBlock(256*2, 128)
        self.decoder_2 = DecoderBlock(128 * 2, 64)
        self.decoder_1 = DecoderBlock(64 * 2, 32)

        self.final_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, n_seg_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x0):
        # x = pet modality y = ct modality
        x = torch.unsqueeze(x0[:, 0], dim=1)
        y = torch.unsqueeze(x0[:, 1], dim=1)

        e0 = self.stem(x, y)  # 32, D, H, W

        e1 = self.encoder_1(e0)  # 64, D/2, H/2, W/2
        e2 = self.encoder_2(e1)  # 128, D/4, H/4, W/4
        e3 = self.encoder_3(e2)  # 256, D/8, H/8, W/8
        e4 = self.encoder_4(e3)
        
        d4 = self.decoder_4(e4)
        d3 = self.decoder_3(torch.concatenate((d4, e3), dim=1))  # 256, D/8, H/8, W/8 -> # 128, D/4, H/4, W/4
        d2 = self.decoder_2(torch.concatenate((d3, e2), dim=1))  # 128 + 128, D/4, H/4, W/4 -> # 64, D/2, H/2, W/2
        d1 = self.decoder_1(torch.concatenate((d2, e1), dim=1))  # 64 + 64, D/2, H/2, W/2 -> # 32, D, H, W
        out = self.final_conv(torch.concatenate((d1, e0), dim=1))  # 32 + 32, D, H, W -> # n_seg_classes, D, H, W

        return out
