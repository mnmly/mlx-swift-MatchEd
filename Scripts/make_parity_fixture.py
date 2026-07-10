# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "safetensors"]
# ///
"""
Generate a numerical-parity fixture for the Swift port.

Builds a randomly-initialized *converted* PiDiNet (CPU), runs it on a fixed
random input, and writes:
  * weights/matched_random.safetensors  — NHWC weights the Swift loader reads
  * fixtures/parity.safetensors         — golden IO for the Swift parity test:
        input        [1,H,W,3]  (ImageNet-normalized NHWC)
        fused        [1,H,W,1]  (sigmoid fused edge map = side[-1])
        thin         [1,H,W,1]  (SmallUNet crisp output)
        side0..side4 [1,H,W,1]  (all five side outputs, for bisection)

No real checkpoint required — this validates the port's math end to end. Swap
in real weights later via convert_weights.py (identical key layout).

Usage:
    uv run Scripts/make_parity_fixture.py [--height 64 --width 96 --seed 0]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from matched_ref import PiDiNet, state_dict_to_nhwc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def nhwc(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().permute(0, 2, 3, 1).contiguous().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = PiDiNet()
    model.eval()

    # ImageNet-normalized-ish input; exact values don't matter for parity.
    x = torch.randn(1, 3, args.height, args.width)

    with torch.no_grad():
        side, thin = model(x)

    weights = state_dict_to_nhwc(model.state_dict())
    (ROOT / "weights").mkdir(exist_ok=True)
    save_file(weights, str(ROOT / "weights" / "matched_random.safetensors"))

    fixture = {
        "input": nhwc(x),
        "fused": nhwc(side[-1]),
        "thin": nhwc(thin),
    }
    for i, s in enumerate(side):
        fixture[f"side{i}"] = nhwc(s)

    (ROOT / "fixtures").mkdir(exist_ok=True)
    save_file(fixture, str(ROOT / "fixtures" / "parity.safetensors"))

    print("wrote weights/matched_random.safetensors and fixtures/parity.safetensors")
    print(f"  input {tuple(x.shape)} -> fused {tuple(side[-1].shape)}, thin {tuple(thin.shape)}")
    print(f"  fused range [{side[-1].min():.4f}, {side[-1].max():.4f}], "
          f"thin range [{thin.min():.4f}, {thin.max():.4f}]")


if __name__ == "__main__":
    main()
