"""
CPU PyTorch reference for the *converted* MatchED / PiDiNet model.

This mirrors the Swift `MatchEDKit` port 1:1 (same module keys, same forward
math) so it can serve as the numerical-parity oracle. It is the converted
(vanilla-conv) form: every pixel-difference op is a plain convolution here, and
the `convert_pdc` helper folds a *raw* PiDiNet checkpoint into this form.

Deliberately CPU-only and CUDA-free (the original `models/pidinet.py` hard-codes
`.cuda()`), so it runs on an Apple-Silicon Mac for fixture generation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# carv4 PDC layout (models/config.py). Index 0 is init_block; 1..15 are the
# block conv1s in trunk order.
CARV4 = ["cd", "ad", "rd", "cv"] * 4


# --------------------------------------------------------------------------- #
# PDC -> vanilla-conv weight conversion (models/convert_pidinet.py::convert_pdc)
# --------------------------------------------------------------------------- #
def convert_pdc(op: str, weight: torch.Tensor) -> torch.Tensor:
    if op == "cv":
        return weight
    if op == "cd":
        shape = weight.shape
        weight_c = weight.sum(dim=[2, 3])
        weight = weight.view(shape[0], shape[1], -1).clone()
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        return weight.view(shape)
    if op == "ad":
        shape = weight.shape
        weight = weight.view(shape[0], shape[1], -1)
        idx = [3, 0, 1, 6, 4, 2, 7, 8, 5]
        return (weight - weight[:, :, idx]).view(shape)
    if op == "rd":
        shape = weight.shape
        buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        weight = weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weight[:, :, 1:]
        return buffer.view(shape[0], shape[1], 5, 5)
    raise ValueError(f"wrong op {op}")


# --------------------------------------------------------------------------- #
# Modules
# --------------------------------------------------------------------------- #
class PDCBlock(nn.Module):
    """models/pidinet.py::PDCBlock_converted"""

    def __init__(self, pdc, inplane, ouplane, stride=1):
        super().__init__()
        self.stride = stride
        if stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        if pdc == "rd":
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=5, padding=2,
                                   groups=inplane, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, padding=1,
                                   groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv2(self.relu2(self.conv1(x)))
        if self.stride > 1:
            x = self.shortcut(x)
        return y + x


class CSAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.conv2(self.conv1(self.relu1(x))))
        return x * y


class CDCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)

    def forward(self, x):
        x = self.conv1(self.relu1(x))
        return self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + self.conv2_4(x)


class MapReduce(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv(x)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16, track_running_stats=False), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16, track_running_stats=False), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32, track_running_stats=False), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32, track_running_stats=False), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16, track_running_stats=False), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.dec1(self.enc2(self.enc1(x))))


class PiDiNet(nn.Module):
    """models/pidinet.py::PiDiNet with --sa --dil, converted form."""

    def __init__(self, inplane=60, dil=24, pdcs=CARV4):
        super().__init__()
        self.inplane = inplane
        fuse = [inplane, inplane * 2, inplane * 4, inplane * 4]

        init_k = 5 if pdcs[0] == "rd" else 3
        self.init_block = nn.Conv2d(3, inplane, kernel_size=init_k,
                                    padding=init_k // 2, bias=False)
        self.block1_1 = PDCBlock(pdcs[1], inplane, inplane)
        self.block1_2 = PDCBlock(pdcs[2], inplane, inplane)
        self.block1_3 = PDCBlock(pdcs[3], inplane, inplane)
        self.block2_1 = PDCBlock(pdcs[4], inplane, inplane * 2, stride=2)
        self.block2_2 = PDCBlock(pdcs[5], inplane * 2, inplane * 2)
        self.block2_3 = PDCBlock(pdcs[6], inplane * 2, inplane * 2)
        self.block2_4 = PDCBlock(pdcs[7], inplane * 2, inplane * 2)
        self.block3_1 = PDCBlock(pdcs[8], inplane * 2, inplane * 4, stride=2)
        self.block3_2 = PDCBlock(pdcs[9], inplane * 4, inplane * 4)
        self.block3_3 = PDCBlock(pdcs[10], inplane * 4, inplane * 4)
        self.block3_4 = PDCBlock(pdcs[11], inplane * 4, inplane * 4)
        self.block4_1 = PDCBlock(pdcs[12], inplane * 4, inplane * 4, stride=2)
        self.block4_2 = PDCBlock(pdcs[13], inplane * 4, inplane * 4)
        self.block4_3 = PDCBlock(pdcs[14], inplane * 4, inplane * 4)
        self.block4_4 = PDCBlock(pdcs[15], inplane * 4, inplane * 4)

        self.dilations = nn.ModuleList([CDCM(fuse[i], dil) for i in range(4)])
        self.attentions = nn.ModuleList([CSAM(dil) for _ in range(4)])
        self.conv_reduces = nn.ModuleList([MapReduce(dil) for _ in range(4)])
        self.classifier = nn.Conv2d(4, 1, kernel_size=1)
        self.thinner = SmallUNet()

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.init_block(x)
        x1 = self.block1_3(self.block1_2(self.block1_1(x)))
        x2 = self.block2_4(self.block2_3(self.block2_2(self.block2_1(x1))))
        x3 = self.block3_4(self.block3_3(self.block3_2(self.block3_1(x2))))
        x4 = self.block4_4(self.block4_3(self.block4_2(self.block4_1(x3))))

        edges = []
        for i, xi in enumerate([x1, x2, x3, x4]):
            fused = self.attentions[i](self.dilations[i](xi))
            e = self.conv_reduces[i](fused)
            e = F.interpolate(e, (H, W), mode="bilinear", align_corners=False)
            edges.append(e)
        output = self.classifier(torch.cat(edges, dim=1))
        side = [torch.sigmoid(r) for r in edges + [output]]
        thin = self.thinner(side[-1])
        return side, thin


# --------------------------------------------------------------------------- #
# state_dict (NCHW) -> NHWC dict of numpy arrays for safetensors
# --------------------------------------------------------------------------- #
def state_dict_to_nhwc(state_dict) -> dict[str, np.ndarray]:
    """Transpose all 4D conv weights NCHW (O,I,kH,kW) -> NHWC (O,kH,kW,I)."""
    out = {}
    for k, v in state_dict.items():
        t = v.detach().cpu().float()
        if t.dim() == 4:
            t = t.permute(0, 2, 3, 1).contiguous()
        out[k] = t.numpy()
    return out
