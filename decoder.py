# decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from terratorch.registry import TERRATORCH_DECODER_REGISTRY


# ---------- small blocks ----------
class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed due to odd shapes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        return self.relu(x_cat + self.conv_res(x))

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def _normalize_embed_dim(positional_channel_list, expected_embed_dim=None, fallback=(160, 320, 640, 1280)):
    """
    Terratorch passes 'channel_list' as the first positional arg.
    This function normalizes it into a list of length 4.
    Priority:
      1) if positional is a list/tuple of len==4 -> use it
      2) elif expected_embed_dim kwarg is provided and len==4 -> use it
      3) elif positional is an int -> broadcast [c,c,c,c]
      4) else -> fallback to default tuple
    """
    # 1) exact list/tuple of 4
    if isinstance(positional_channel_list, (list, tuple)) and len(positional_channel_list) == 4:
        return list(positional_channel_list)

    # 2) kw override
    if isinstance(expected_embed_dim, (list, tuple)) and len(expected_embed_dim) == 4:
        return list(expected_embed_dim)

    # 3) broadcast single int
    if isinstance(positional_channel_list, int):
        return [positional_channel_list] * 4

    # 4) final fallback
    return list(fallback)


# ---------- PT2Decoder (robust) ----------
@TERRATORCH_DECODER_REGISTRY.register
class PT2Decoder(nn.Module):
    def __init__(self, embed_dim=None, *, expected_embed_dim=None):
        """
        Terratorch passes channel_list as the FIRST positional arg into 'embed_dim'.
        We normalize here so we always end up with 4 entries.
        Optional kw-only 'expected_embed_dim' provides a clean override.
        """
        super().__init__()

        c1, c2, c3, c4 = _normalize_embed_dim(embed_dim, expected_embed_dim)

        # shrink each scale to 64 with RFB blocks
        self.rfb1 = RFB_modified(c1, 64)
        self.rfb2 = RFB_modified(c2, 64)
        self.rfb3 = RFB_modified(c3, 64)
        self.rfb4 = RFB_modified(c4, 64)

        # UNet-style ups
        self.up1 = Up(64 + 64, 64)  # fuse x4 -> x3
        self.up2 = Up(64 + 64, 64)  # -> x2
        self.up3 = Up(64 + 64, 64)  # -> x1

        # optional side heads (not returned)
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)

        self.out_channels = 64

    def forward(self, feats):
        x1, x2, x3, x4 = feats  # largest -> smallest
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        _ = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear', align_corners=False)
        x = self.up2(x, x2)
        _ = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear', align_corners=False)
        x = self.up3(x, x1)
        out = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        return out


# ---------- UPerDecoder ----------
@TERRATORCH_DECODER_REGISTRY.register
class UPerDecoder(nn.Module):
    def __init__(self, embed_dim=None, *, out_ch: int = 256, ppm_channels: int = 128, expected_embed_dim=None):
        """
        Same normalization as PT2Decoder; keeps compatibility with Terratorch factory.
        """
        super().__init__()
        embed_dim = _normalize_embed_dim(embed_dim, expected_embed_dim)
        self.embed_dim = embed_dim
        self.out_channels = out_ch

        # 1) PPM on top (smallest) feature f4
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=scale),
                ConvBNReLU(embed_dim[3], ppm_channels, k=1, padding=0)
            )
            for scale in (1, 2, 3, 6)
        ])
        self.ppm_conv = ConvBNReLU(embed_dim[3] + len(self.ppm) * ppm_channels, ppm_channels, k=3)

        # 2) laterals
        self.lateral4 = ConvBNReLU(embed_dim[3], ppm_channels, k=1, padding=0)
        self.lateral3 = ConvBNReLU(embed_dim[2], ppm_channels, k=1, padding=0)
        self.lateral2 = ConvBNReLU(embed_dim[1], ppm_channels, k=1, padding=0)
        self.lateral1 = ConvBNReLU(embed_dim[0], ppm_channels, k=1, padding=0)

        # 3) fusions
        self.fuse3 = ConvBNReLU(ppm_channels * 2, ppm_channels)
        self.fuse2 = ConvBNReLU(ppm_channels * 2, ppm_channels)
        self.fuse1 = ConvBNReLU(ppm_channels * 2, ppm_channels)

        # 4) final projection
        self.out_conv = nn.Sequential(
            ConvBNReLU(ppm_channels, ppm_channels),
            nn.Conv2d(ppm_channels, out_ch, kernel_size=1),
        )

    def _ppm_forward(self, x):
        h, w = x.shape[-2:]
        outs = [x]
        for p in self.ppm:
            y = p(x)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
            outs.append(y)
        return self.ppm_conv(torch.cat(outs, dim=1))

    def forward(self, features):
        f1, f2, f3, f4 = features  # largest -> smallest
        ppm_feat = self._ppm_forward(f4)
        lat4 = self.lateral4(f4)
        top = lat4 + ppm_feat

        lat3 = self.lateral3(f3)
        top_up_3 = F.interpolate(top, size=lat3.shape[-2:], mode='bilinear', align_corners=False)
        m3 = self.fuse3(torch.cat([lat3, top_up_3], dim=1))

        lat2 = self.lateral2(f2)
        m3_up_2 = F.interpolate(m3, size=lat2.shape[-2:], mode='bilinear', align_corners=False)
        m2 = self.fuse2(torch.cat([lat2, m3_up_2], dim=1))

        lat1 = self.lateral1(f1)
        m2_up_1 = F.interpolate(m2, size=lat1.shape[-2:], mode='bilinear', align_corners=False)
        m1 = self.fuse1(torch.cat([lat1, m2_up_1], dim=1))

        out = self.out_conv(m1)
        return out


# --- add to decoder.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from terratorch.registry import TERRATORCH_DECODER_REGISTRY

def _norm4(x, fallback=(160,320,640,1280)):
    if isinstance(x, (list, tuple)) and len(x)==4: return list(x)
    if isinstance(x, int): return [x]*4
    return list(fallback)

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False, groups=groups)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = ConvBNAct(ch, ch, 3, 1, 1)
        self.c2 = ConvBNAct(ch, ch, 3, 1, 1)
    def forward(self, x): return x + self.c2(self.c1(x))

class DSRefine(nn.Module):
    """Depthwise-separable refine to sharpen boundaries cheaply."""
    def __init__(self, ch):
        super().__init__()
        self.dw = ConvBNAct(ch, ch, k=3, p=1, groups=ch)
        self.pw = ConvBNAct(ch, ch, k=1, p=0)
    def forward(self, x): return self.pw(self.dw(x))

class UpLearned(nn.Module):
    """ConvTranspose2d upsampling + light refine."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.refine = ConvBNAct(in_ch, out_ch, k=3, p=1)
    def forward(self, x, size=None):
        x = self.up(x)
        if size is not None and x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.refine(x)

class ASPPLite(nn.Module):
    """Small atrous pyramid for the top (coarsest) feature.

    in_ch -> out_ch, split evenly across branches with dilations=(1,3,6,9).
    All layers are defined in __init__ so Lightning can move them to CUDA.
    """
    def __init__(self, in_ch: int, out_ch: int, dilations=(1, 3, 6, 9)):
        super().__init__()
        assert out_ch % len(dilations) == 0, "out_ch must be divisible by number of branches"
        br_ch = out_ch // len(dilations)

        self.branches = nn.ModuleList()
        for d in dilations:
            k = 1 if d == 1 else 3
            p = 0 if d == 1 else d
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, br_ch, kernel_size=k, padding=p, dilation=d, bias=False),
                    nn.BatchNorm2d(br_ch),
                    nn.GELU(),
                )
            )

        self.proj = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        outs = [b(x) for b in self.branches]  # each [B, br_ch, H, W]
        y = torch.cat(outs, dim=1)            # [B, out_ch, H, W]
        return self.proj(y)

def maybe_add_coord(x):
    """Add normalized (x,y) CoordConv channels to feature map."""
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device),
        torch.linspace(-1, 1, w, device=x.device),
        indexing="ij"
    )
    coords = torch.stack([xx, yy], dim=0).expand(b, -1, -1, -1)  # [B,2,H,W]
    return torch.cat([x, coords], dim=1)

@TERRATORCH_DECODER_REGISTRY.register
class PT2DecoderSpatialPlus(nn.Module):
    """
    Attention-free UNet-like decoder optimized for spatial association:
      - ConvTranspose2d upsampling
      - Residual + depthwise-separable refine after each fusion
      - ASPP-lite on top level for minimal context
      - Optional CoordConv for better localization
    Returns high-res feature; sets self.out_channels for Terratorch head.
    """
    def __init__(self, embed_dim=None, *, base_ch: int = 64, use_coord: bool = False, expected_embed_dim=None):
        super().__init__()
        c1, c2, c3, c4 = _norm4(embed_dim if embed_dim is not None else expected_embed_dim)

        # project ViT features to common width
        self.p1 = ConvBNAct(c1 + (2 if use_coord else 0), base_ch, k=1, p=0)
        self.p2 = ConvBNAct(c2, base_ch, k=1, p=0)
        self.p3 = ConvBNAct(c3, base_ch, k=1, p=0)
        self.p4 = ConvBNAct(c4, base_ch, k=1, p=0)

        # top context
        self.aspp = ASPPLite(base_ch, base_ch)

        # top-down
        self.up43 = UpLearned(base_ch, base_ch)
        self.fuse3 = nn.Sequential(ConvBNAct(base_ch*2, base_ch), ResBlock(base_ch), DSRefine(base_ch))

        self.up32 = UpLearned(base_ch, base_ch)
        self.fuse2 = nn.Sequential(ConvBNAct(base_ch*2, base_ch), ResBlock(base_ch), DSRefine(base_ch))

        self.up21 = UpLearned(base_ch, base_ch)
        self.fuse1 = nn.Sequential(ConvBNAct(base_ch*2, base_ch), ResBlock(base_ch), DSRefine(base_ch))

        self.final = nn.Sequential(ResBlock(base_ch), DSRefine(base_ch))
        self.use_coord = use_coord
        self.out_channels = base_ch

    def forward(self, feats):
        f1, f2, f3, f4 = feats  # large -> small
        if self.use_coord:
            f1 = maybe_add_coord(f1)
        f1, f2, f3, f4 = self.p1(f1), self.p2(f2), self.p3(f3), self.p4(f4)

        top = self.aspp(f4)

        d3u = self.up43(top, size=f3.shape[-2:])
        d3  = self.fuse3(torch.cat([f3, d3u], dim=1))

        d2u = self.up32(d3, size=f2.shape[-2:])
        d2  = self.fuse2(torch.cat([f2, d2u], dim=1))

        d1u = self.up21(d2, size=f1.shape[-2:])
        d1  = self.fuse1(torch.cat([f1, d1u], dim=1))

        return self.final(d1)
