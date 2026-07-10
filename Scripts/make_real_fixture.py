# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "safetensors", "pillow"]
# ///
"""
Real-weight, real-image parity fixture + Swift-ready weights, from a *raw*
MatchEd checkpoint.

  1. Load + convert the checkpoint (fold PDC kernels) into `matched_ref.PiDiNet`.
  2. Export NHWC safetensors for the Swift loader.
  3. Load an image, apply the exact dataloader preprocessing (ToTensor + ImageNet
     Normalize), run the reference, and save golden IO.

Unlike make_parity_fixture.py (random weights), this validates the port against
the *trained* model on a real image — real value ranges, real edges.

Usage:
    uv run Scripts/make_real_fixture.py CHECKPOINT.pth [--image IMG] [--config carv4]
    # no --image → a structured synthetic scene is drawn.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.numpy import save_file

from matched_ref import PiDiNet, state_dict_to_nhwc
from convert_weights import CONFIGS, convert, strip_module

ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def synth_scene(w=400, h=300) -> Image.Image:
    img = Image.new("RGB", (w, h), (30, 40, 60))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 180, 160], fill=(220, 180, 90), outline=(255, 255, 255), width=3)
    d.ellipse([220, 60, 360, 200], fill=(80, 200, 160), outline=(20, 20, 20), width=4)
    for i in range(6):
        d.line([(0, i * 50), (w, i * 50 + 30)], fill=(200, 60, 60), width=2)
    d.polygon([(120, 220), (200, 280), (60, 280)], fill=(160, 120, 240))
    return img


def preprocess(img: Image.Image) -> np.ndarray:
    """ToTensor + ImageNet Normalize → NHWC [1,H,W,3] float32."""
    arr = np.asarray(img.convert("RGB"), np.float32) / 255.0  # HWC
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr[None]  # NHWC


def nhwc(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().permute(0, 2, 3, 1).contiguous().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--image", default=None)
    ap.add_argument("--config", default="carv4", choices=list(CONFIGS))
    ap.add_argument("--tag", default="real")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = strip_module(sd)
    sd = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}
    converted = convert(sd, CONFIGS[args.config])

    model = PiDiNet()
    missing, unexpected = model.load_state_dict(converted, strict=False)
    # Stride-1 blocks have no shortcut; those are the only expected "missing".
    real_missing = [m for m in missing if "shortcut" not in m]
    assert not real_missing, f"missing keys: {real_missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    model.eval()

    (ROOT / "weights").mkdir(exist_ok=True)
    save_file(state_dict_to_nhwc(model.state_dict()),
              str(ROOT / "weights" / f"matched_{args.tag}.safetensors"))

    img = Image.open(args.image).convert("RGB") if args.image else synth_scene()
    (ROOT / "fixtures").mkdir(exist_ok=True)
    img.save(ROOT / "fixtures" / f"{args.tag}_input.png")

    x_np = preprocess(img)                        # NHWC
    x = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous()  # NCHW for torch
    with torch.no_grad():
        side, thin = model(x)

    fixture = {"input": x_np.astype(np.float32),
               "fused": nhwc(side[-1]), "thin": nhwc(thin)}
    for i, s in enumerate(side):
        fixture[f"side{i}"] = nhwc(s)
    save_file(fixture, str(ROOT / "fixtures" / f"parity_{args.tag}.safetensors"))

    # Also save the reference edge maps as PNGs for a visual oracle.
    for name, t in [("fused", side[-1]), ("thin", thin)]:
        m = (t[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(m, "L").save(ROOT / "fixtures" / f"{args.tag}_{name}_ref.png")

    print(f"wrote weights/matched_{args.tag}.safetensors + fixtures/parity_{args.tag}.safetensors")
    print(f"  input {tuple(x.shape)}  fused∈[{side[-1].min():.3f},{side[-1].max():.3f}]  "
          f"thin∈[{thin.min():.3f},{thin.max():.3f}]")


if __name__ == "__main__":
    main()
