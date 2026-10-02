import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _select_by_domain(x, y):
    idx = torch.arange(y.size(0), device=y.device)
    return x[idx, y]


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, normalize=False, downsample=False):
        super().__init__()
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = in_dim != out_dim

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)
        if normalize:
            self.norm1 = nn.InstanceNorm2d(in_dim, affine=True)
            self.norm2 = nn.InstanceNorm2d(in_dim, affine=True)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        return x

    def forward(self, x):
        return (self._shortcut(x) + self._residual(x)) / math.sqrt(2)


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, style_dim, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.learned_sc = in_dim != out_dim

        self.norm1 = AdaIN(style_dim, in_dim)
        self.norm2 = AdaIN(style_dim, out_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = F.leaky_relu(x, 0.2)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        return (self._shortcut(x) + self._residual(x, s)) / math.sqrt(2)


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512):
        super().__init__()
        in_dim = 2**14 // img_size
        self.from_rgb = nn.Conv2d(3, in_dim, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(in_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dim, 3, 1, 1, 0),
        )

        repeat_num = int(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            out_dim = min(in_dim * 2, max_conv_dim)
            self.encode.append(ResBlock(in_dim, out_dim, normalize=True, downsample=True))
            self.decode.insert(0, AdaINResBlock(out_dim, in_dim, style_dim, upsample=True))
            in_dim = out_dim

        for _ in range(2):
            self.encode.append(ResBlock(in_dim, in_dim, normalize=True))
            self.decode.insert(0, AdaINResBlock(in_dim, in_dim, style_dim))

    def forward(self, x, s):
        h = self.from_rgb(x)
        for block in self.encode:
            h = block(h)
        for block in self.decode:
            h = block(h, s)
        return self.to_rgb(h)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=3):
        super().__init__()
        layers = [nn.Linear(latent_dim, 512), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512), nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, style_dim),
                )
            )

    def forward(self, z, y=None):
        h = self.shared(z)
        out = [layer(h) for layer in self.unshared]
        out = torch.stack(out, dim=1)
        if y is None:
            return out
        return _select_by_domain(out, y)


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=3, max_conv_dim=512):
        super().__init__()
        in_dim = 2**14 // img_size
        blocks = [nn.Conv2d(3, in_dim, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            out_dim = min(in_dim * 2, max_conv_dim)
            blocks.append(ResBlock(in_dim, out_dim, downsample=True))
            in_dim = out_dim

        blocks += [
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dim, in_dim, 4, 1, 0),
            nn.LeakyReLU(0.2),
        ]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.ModuleList([nn.Linear(in_dim, style_dim) for _ in range(num_domains)])

    def forward(self, x, y=None):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = [layer(h) for layer in self.unshared]
        out = torch.stack(out, dim=1)
        if y is None:
            return out
        return _select_by_domain(out, y)


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=3, max_conv_dim=512):
        super().__init__()
        in_dim = 2**14 // img_size
        blocks = [nn.Conv2d(3, in_dim, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            out_dim = min(in_dim * 2, max_conv_dim)
            blocks.append(ResBlock(in_dim, out_dim, downsample=True))
            in_dim = out_dim

        blocks += [
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dim, in_dim, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dim, num_domains, 1, 1, 0),
        ]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y=None):
        out = self.main(x)
        out = out.view(out.size(0), -1)
        if y is None:
            return out
        return _select_by_domain(out, y)


AdaIN_ResBlock = AdaINResBlock
