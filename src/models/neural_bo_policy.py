"""
Neural Bayesian Optimisation policy for placement search.

Frozen reward UNet provides a prior (heatmap). A small U-Net policy reads
[heatmap, ifp_mask, visited_mask, visited_rewards] and outputs per-pixel
logits for the next pixel to query. After T Shapely queries the best
observed reward is returned.

The encoder is a 4-level U-Net so each output pixel has a global receptive
field over the 128x128 input — necessary for shape-agnostic acquisition
strategies that depend on global heatmap structure (e.g. "the second-best
basin is on the other side of the IFP").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _SmallUNet(nn.Module):
    """4-level U-Net for 128x128 -> 128x128 single-channel output.
    base=32 -> ~480k params; base=48 -> ~1.1M params."""

    def __init__(self, in_ch=4, base=32, out_ch=1):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = _ConvBlock(in_ch, c1)             # 128
        self.enc2 = _ConvBlock(c1, c2)                # 64
        self.enc3 = _ConvBlock(c2, c3)                # 32
        self.enc4 = _ConvBlock(c3, c4)                # 16
        self.bottleneck = _ConvBlock(c4, c4)          # 16
        self.up3 = _ConvBlock(c4 + c3, c3)            # 32
        self.up2 = _ConvBlock(c3 + c2, c2)            # 64
        self.up1 = _ConvBlock(c2 + c1, c1)            # 128
        self.out_conv = nn.Conv2d(c1, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(e4)
        d3 = self.up3(torch.cat([F.interpolate(b, scale_factor=2,
                                               mode='bilinear', align_corners=False),
                                 e3], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, scale_factor=2,
                                               mode='bilinear', align_corners=False),
                                 e2], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, scale_factor=2,
                                               mode='bilinear', align_corners=False),
                                 e1], dim=1))
        return self.out_conv(d1)


class NeuralBOPolicy(nn.Module):
    def __init__(self, reward_unet, hidden=32, T_max=8):
        """`hidden` is the U-Net base channel count (32 default ~ 480k params)."""
        super().__init__()
        self.reward_unet = reward_unet
        for p in self.reward_unet.parameters():
            p.requires_grad = False

        self.T_max = T_max
        in_ch = 4   # [heatmap, ifp, visited_mask, visited_rewards]
        self.encoder = _SmallUNet(in_ch=in_ch, base=hidden, out_ch=1)

    def train(self, mode=True):
        super().train(mode)
        self.reward_unet.eval()
        return self

    @torch.no_grad()
    def predict_heatmap(self, fs_mask, rot_part_mask, ifp_mask):
        inp = torch.stack([fs_mask, rot_part_mask, ifp_mask], dim=1)
        return self.reward_unet(inp).squeeze(1)

    def forward(self, heatmap, ifp_mask, visited_mask, visited_rewards):
        """All inputs (B, H, W). Returns logits (B, H, W)."""
        x = torch.stack([heatmap, ifp_mask, visited_mask, visited_rewards], dim=1)
        return self.encoder(x).squeeze(1)
