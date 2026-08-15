"""Lightweight Real-ESRGAN architectures (Apache-2.0, adapted from xinntao/Real-ESRGAN).

No basicsr / realesrgan package dependency — pure PyTorch modules suitable for
~2 GB VRAM or CPU tiled inference.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style SR network used by ``realesr-general-x4v3``.

    Much smaller / faster than RRDBNet — preferred on CPU and ~2 GB VRAM.
    """

    def __init__(
        self,
        *,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        if act_type == "prelu":
            activation: nn.Module = nn.PReLU(num_parameters=num_feat)
        elif act_type == "relu":
            activation = nn.ReLU(inplace=True)
        else:
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.body.append(activation)

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == "prelu":
                self.body.append(nn.PReLU(num_parameters=num_feat))
            elif act_type == "relu":
                self.body.append(nn.ReLU(inplace=True))
            else:
                self.body.append(nn.LeakyReLU(0.1, inplace=True))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(tensor, scale_factor=self.upscale, mode="nearest")
        return out + base


class _ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(tensor))
        x2 = self.lrelu(self.conv2(torch.cat((tensor, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((tensor, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((tensor, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((tensor, x1, x2, x3, x4), 1))
        return x5 * 0.2 + tensor


class _RRDB(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(tensor)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + tensor


class RRDBNet(nn.Module):
    """RRDBNet used by RealESRGAN_x2plus / x4plus.

    Official Real-ESRGAN behavior (basicsr RRDBNet):
    - scale=2: pixel-unshuffle input (3→12 ch, H/2), then **two** ×2 upsamples → 2H
    - scale=4: plain 3-ch input, then two ×2 upsamples → 4H
    """

    def __init__(
        self,
        *,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.scale = scale
        first_in = num_in_ch * 4 if scale == 2 else num_in_ch
        self.conv_first = nn.Conv2d(first_in, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    @staticmethod
    def _pixel_unshuffle(tensor: torch.Tensor, scale: int = 2) -> torch.Tensor:
        b, c, h, w = tensor.shape
        out_c = c * (scale**2)
        assert h % scale == 0 and w % scale == 0
        tensor = tensor.view(b, c, h // scale, scale, w // scale, scale)
        tensor = tensor.permute(0, 1, 3, 5, 2, 4).contiguous()
        return tensor.view(b, out_c, h // scale, w // scale)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        feat_in = self._pixel_unshuffle(tensor, 2) if self.scale == 2 else tensor
        feat = self.conv_first(feat_in)
        body = self.conv_body(self.body(feat)) + feat
        # Always two ×2 stages (matches basicsr RRDBNet used by RealESRGAN weights).
        body = self.lrelu(self.conv_up1(F.interpolate(body, scale_factor=2, mode="nearest")))
        body = self.lrelu(self.conv_up2(F.interpolate(body, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(body)))
        return out
