# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "safetensors"]
# ///
"""
Convert a *raw* MatchED / PiDiNet checkpoint (.pth) into an NHWC safetensors
file the Swift `MatchEDKit` loader consumes.

Steps:
  1. Load `checkpoint['state_dict']`, strip any `module.` (DataParallel) prefix.
  2. Fold every pixel-difference kernel into a vanilla conv (`convert_pdc`),
     matching the paper's `--evaluate-converted` path.
  3. Transpose all 4D conv weights NCHW -> NHWC.
  4. Save as safetensors with keys identical to the PyTorch state_dict.

Usage:
    uv run Scripts/convert_weights.py CHECKPOINT.pth weights/matched.safetensors [--config carv4]
"""
import argparse
from pathlib import Path

import torch
from safetensors.numpy import save_file

from matched_ref import CARV4, convert_pdc, state_dict_to_nhwc

# Which pdc index governs each convertible weight (models/convert_pidinet.py).
_PDC_KEYS = {"init_block.weight": 0}
_blocks = [
    "block1_1", "block1_2", "block1_3",
    "block2_1", "block2_2", "block2_3", "block2_4",
    "block3_1", "block3_2", "block3_3", "block3_4",
    "block4_1", "block4_2", "block4_3", "block4_4",
]
for i, b in enumerate(_blocks, start=1):
    _PDC_KEYS[f"{b}.conv1.weight"] = i

CONFIGS = {"carv4": CARV4}


def strip_module(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}


def convert(sd, pdcs):
    out = {}
    for name, p in sd.items():
        pdc_idx = next((idx for key, idx in _PDC_KEYS.items() if key in name), None)
        out[name] = convert_pdc(pdcs[pdc_idx], p) if pdc_idx is not None else p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--config", default="carv4", choices=list(CONFIGS))
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = strip_module(sd)
    # Drop any non-parameter bookkeeping (e.g. num_batches_tracked, if present).
    sd = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}

    converted = convert(sd, CONFIGS[args.config])
    nhwc = state_dict_to_nhwc(converted)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(nhwc, str(out))
    print(f"wrote {len(nhwc)} tensors -> {out}")


if __name__ == "__main__":
    main()
