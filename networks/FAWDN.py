import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, MeanShift


class SingleLayer(nn.Module):
    def __init__(self, inChannels,growthRate):
        super(SingleLayer, self).__init__()
        self.n_weight = inChannels//growthRate
        self.growthRate = growthRate
        self.conv =nn.Conv2d(inChannels, growthRate,kernel_size=3 ,padding=1, bias=True)
        self.w = nn.Parameter(torch.ones(self.n_weight)) # the adaptive weight

    def forward(self, x):
        out = self.w.expand(self.growthRate, self.n_weight).permute(1,0).reshape(1,-1,1,1)
        out = x * out
        out = F.relu(self.conv(out))
        out = torch.cat((x, out), 1)
        return out


class SingleBlock(nn.Module):
    def __init__(self, inChannels, growthRate, nDenselayer):
        super(SingleBlock, self).__init__()
        self.block = self._make_dense(inChannels, growthRate, nDenselayer)

    def _make_dense(self, inChannels, growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels, growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out=self.block(x)
        return out


class AWDB(nn.Module):
    def __init__(self, num_features, growthRate, nBlock, nDenselayer):
        super(AWDB, self).__init__()

        self.compress_in = ConvBlock(num_features+256, num_features,
                                     kernel_size=1,
                                     act_type='relu', norm_type=None)
        blocks = []
        inChannels = num_features
        for i in range(nBlock):
            blocks.append(SingleBlock(inChannels, growthRate, nDenselayer))
            inChannels += growthRate * nDenselayer
        self.block = nn.Sequential(*blocks)

        self.Bottleneck = nn.Conv2d(in_channels=inChannels, out_channels=256, kernel_size=1, padding=0, bias=True)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            x = self.Bottleneck(self.block(x))
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False
            return x
        else:
            x = torch.cat((x, self.last_hidden), dim=1)
            x = self.compress_in(x)
            x = self.Bottleneck(self.block(x))
            self.last_hidden = x
            return x

    def reset_state(self):
        self.should_reset = True

class FAWDN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, nBlock, nDenselayer, upscale_factor):
        super(FAWDN, self).__init__()

        if upscale_factor == 2:
            kernel_size, stride, padding = 4, 2, 1
        elif upscale_factor == 3:
            kernel_size, stride, padding = 7, 3, 2

        self.num_steps = num_steps
        self.num_features = num_features
        growthRate = num_features

        self.upscale_factor = upscale_factor

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # LR feature extraction block
        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True)
        self.block = AWDB(num_features, growthRate, nBlock, nDenselayer)

        if upscale_factor == 4:
            self.convt = nn.Sequential(*[
                nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1,
                                   bias=True),
                nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1,
                                   bias=True)
            ])
        else:
            self.convt = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=kernel_size,
                                            stride=upscale_factor, padding=padding, bias=True)


        self.conv2 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1, bias=True)

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x):
        self._reset_state()

        x = self.sub_mean(x)

        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)

        x = self.conv_in(x)

        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)
            h = self.convt(h)
            h = self.conv2(h)
            h = torch.add(inter_res, h)
            h = self.add_mean(h)
            outs.append(h)

        return outs # return output of every timesteps

    def _reset_state(self):
        self.block.reset_state()