import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------
# Basic Blocks
# --------------------------------------------------------------
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()

        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    
    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1+gamma) * self.norm(x) + beta
    
class ResBlock(nn.Module):
    """Mapping Network, Style Encoder, Discriminator's Shared Layers"""
    def __init__(self, in_dim, out_dim, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim,  out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(in_dim, affine=True)
        self.norm2 = nn.InstanceNorm2d(out_dim, affine=True)
        self.downsample = downsample
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1, 1, 0) if in_dim != out_dim else None

    def forward(self, x):
        out = self.norm1(x)
        out = F.leaky_relu(out, 0.2)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)

        if self.shortcut is not None:
            x = self.shortcut(x)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
            x = F.avg_pool2d(x, 2)
        
        return out + x
    
class AdaIN_ResBlock(nn.Module):
    """ResBlock that injects Style to Generator"""
    def __init__(self, in_dim, out_dim, style_dim, upsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.adain1 = AdaIN(style_dim, in_dim)
        self.adain2 = AdaIN(style_dim, out_dim)
        self.upsample = upsample
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1, 1, 0) if in_dim != out_dim else None

    def forward(self, x, s):
        out = self.adain1(x, s)
        out = F.leaky_relu(out, 0.2)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.conv1(out)
        out = self.adain2(out, s)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return out + x
    
# --------------------------------------------------------------
# Main Modules (G, F, E, D)
# --------------------------------------------------------------

# (a) Generator
class Generator(nn.Module):
    """Input image + Style -> Output image"""
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512):
        super().__init__()
        in_dim = 2**14 // img_size
        self.from_rgb = nn.Conv2d(3, in_dim, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()

        # Down-sampling (Encoder)
        repeat_num = int(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            out_dim = min(2*in_dim, max_conv_dim)
            self.encode.append(ResBlock(in_dim, out_dim, downsample=True))
            in_dim = out_dim

        # Bottleneck
        for _ in range(2):
            self.encode.append(ResBlock(in_dim, in_dim, downsample=False))

        # Up-sampling (Decoder with AdaIN)
        for _ in range(2):
            self.decode.append(AdaIN_ResBlock(in_dim, in_dim, style_dim, upsample=False))
        for _ in range(repeat_num):
            out_dim = min(2*in_dim, max_conv_dim)
            self.decode.append(AdaIN_ResBlock(in_dim, in_dim//2, style_dim, upsample=True))
            in_dim = in_dim//2

        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(in_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dim, 3, 1, 1, 0)
        )

    def forward(self, x, s):
        out = self.from_rgb(x)
        for block in self.encode:
            out = block(out)
        for block in self.decode:
            out = block(out, s)
        return self.to_rgb(out)
    
# (b) Mapping Network
class MappingNetwork(nn.Module):
    """Latent code z -> Multi-domain Style codes s"""
    def __init__(self, latent_dim=16, style_dim=64, num_domains=3):
        super().__init__()
        # Shared Layers
        layers = [nn.Linear(latent_dim, 512), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512), nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        # Domain Specific Branches
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, style_dim)
            ))

    def forward(self, z, y=None):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)   # (batch, num_domains, style_dim)
        
        # --- NEW LOGIC START ---
        if y is not None:
            # If domain labels are provided, select only the relevant styles
            idx = torch.arange(y.size(0)).to(y.device)
            out = out[idx, y]           # Output shape becomes: (batch, style_dim)
        # --- NEW LOGIC END ---
        
        return out
    
# (c) Style Encoder
class StyleEncoder(nn.Module):
    """Input image -> Multi-domain Style codes s"""
    def __init__(self, img_size=256, style_dim=64, num_domains=3, max_conv_dim=512):
        super().__init__()
        in_dim = 2**14 // img_size
        blocks = [nn.Conv2d(3, in_dim, 3, 1, 1)]

        # Shared Layers
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            out_dim = min(2*in_dim, max_conv_dim)
            blocks += [ResBlock(in_dim, out_dim, downsample=True)]
            in_dim = out_dim

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(in_dim, out_dim, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        # Domain Specific Branches (Linear Layers)
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(out_dim, style_dim))

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)   # (batch, num_domains, style_dim)
        return out
    
# (d) Discriminator
class Discriminator(nn.Module):
    """Input image -> Real/Fake scores for each domain"""
    def __init__(self, img_size=256, num_domains=3, max_conv_dim=512):
        super().__init__()
        in_dim = 3
        out_dim = 64

        # Shared Layers
        blocks = [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.LeakyReLU(0.2)]
        in_dim = out_dim

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            out_dim = min(2*in_dim, max_conv_dim)
            blocks += [ResBlock(in_dim, out_dim, downsample=True)]
            in_dim = out_dim
        self.shared = nn.Sequential(*blocks)

        # Domain Specific Branches (Binary Classification)
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(nn.Conv2d(out_dim, 1, 4, 1, 0))

    def forward(self, x):
        h = self.shared(x)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)   # (batch, num_domains, 1, 1, 1)
        return out.squeeze()