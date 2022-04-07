import torch

from einops import rearrange
from torch import nn
from torch.nn.utils import weight_norm


class Conv(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=3, groups=1, bias=True, a=1):
        super().__init__()

        conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,
                         padding=(kernel_size-1)//2, groups=groups, bias=bias)
        self.conv = weight_norm(conv)

        self.init_weight(a)

    def forward(self, x):
        out = self.conv(x)
        return out

    def init_weight(self, a):
        nn.init.kaiming_normal_(self.conv.weight, a=a, mode='fan_in', nonlinearity='leaky_relu')


class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.model = nn.Sequential(
            Conv(channels, channels, a=0.2),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(channels, channels, a=1)
        )

    def forward(self, x):
        return self.model(x) + x


class RDB(nn.Module):
    """Modifed Residual Dense Block"""

    def __init__(self, channels, n_layer, growth_rate):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            in_channel = channels+i*growth_rate
            if i == 0:
                self.layers.append(Conv(in_channel, growth_rate, bias=False, a=0.2))
            elif i == n_layer-1:
                self.layers.append(Conv(in_channel, growth_rate, bias=False, a=1))
            else:
                self.layers.append(nn.Sequential(
                    nn.LeakyReLU(0.2),
                    Conv(in_channel, growth_rate, bias=False, a=0.2)
                ))

        self.last = Conv(channels+n_layer*growth_rate, channels, kernel_size=1, bias=False, a=1)

    def forward(self, x):
        prev = x
        for model in self.layers:
            out = model(prev)
            prev = torch.cat([prev, out], dim=1)

        out = self.last(prev)
        return out + x


class RED(nn.Module):
    """Network Structure of RED

    Paper: Restoring Extremely Dark Images in Real Time (CVPR'2021)
    """

    def __init__(self):
        super().__init__()

        self.up4 = nn.PixelShuffle(4)
        self.up2 = nn.PixelShuffle(2)

        self.conv32x = nn.Sequential(
            Conv(1024, 128, groups=128, bias=True, a=1),
            Conv(128, 64, groups=1, bias=True, a=1)
        )

        self.RDB1 = RDB(channels=64, n_layer=4, growth_rate=32)
        self.RDB2 = RDB(channels=64, n_layer=5, growth_rate=32)
        self.RDB3 = RDB(channels=64, n_layer=5, growth_rate=32)

        self.rdball = Conv(64*3, 64, kernel_size=1, bias=False, a=1)
        self.conv_rdb8x = Conv(64//16, 64, bias=True, a=1)
        self.resblock8x = ResBlock(64)

        self.conv32_8_cat = nn.Sequential(
            Conv(128, 32, groups=4, bias=True, a=1),
            Conv(32, 192, groups=1, bias=True, a=0.2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(4)
        )

        self.conv2x = Conv(4, 12, kernel_size=5, bias=True, a=1)
        self.conv_2_8_32 = Conv(24, 12, kernel_size=5, bias=True, a=1)

    def pixelshuffle_inv(self, tensor, r):
        return rearrange(tensor, 'b c (h r1) (w r2) -> b (c r1 r2) h w', r1=r, r2=r)

    def channel_shuffle(self, x, groups):
        return rearrange(x, 'b (c1 c2) h w -> b (c2 c1) h w', c1=groups)

    def forward(self, low):
        low2x = self.pixelshuffle_inv(low, 2)  # 2x branch starts
        low8x = self.pixelshuffle_inv(low2x, 4)  # 8x branch starts
        low32x = self.pixelshuffle_inv(low2x, 16)  # 32x branch starts

        # 32x branch
        feat32x = self.conv32x(low32x)
        rdb1 = self.RDB1(feat32x)
        rdb2 = self.RDB2(rdb1)
        rdb3 = self.RDB3(rdb2)

        out32x = torch.cat([rdb1, rdb2, rdb3], dim=1)
        out32x = self.rdball(out32x)+feat32x
        out32x = self.up4(out32x)

        # 8x branch
        feat8x = self.resblock8x(low8x)
        rdb8x = self.conv_rdb8x(out32x)

        out8x = torch.cat([feat8x, rdb8x], dim=1)
        out8x = self.channel_shuffle(out8x, groups=2)
        out8x = self.conv32_8_cat(out8x)

        # 2x branch
        feat2x = self.conv2x(low2x)
        out2x = torch.cat([feat2x, out8x], dim=1)
        out2x = self.up2(self.conv_2_8_32(out2x))

        return torch.clamp(out2x, min=0.0, max=1.0)
