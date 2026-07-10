# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""
Validate the PDC -> vanilla-conv weight conversion against the *original
authors'* runtime op code (`models/ops.py::createConvFunc`).

For each op (cd, ad, rd) we check that
    conv2d(x, convert_pdc(op, W))  ==  pdc_runtime(op)(x, W)
i.e. folding the kernel (what Swift consumes) is numerically identical to the
paper's on-the-fly pixel-difference convolution. This closes the loop on the
only non-trivial transform in the port.

Loads ops.py directly (importlib) to avoid the repo package __init__ pulling in
CUDA-bound modules.
"""
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from matched_ref import convert_pdc

REPO_OPS = Path("../../../python/MatchED/models/ops.py").resolve()

spec = importlib.util.spec_from_file_location("pidinet_ops", REPO_OPS)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def check(op, ksize=3):
    torch.manual_seed(0)
    C = 4
    W = torch.randn(C, C, ksize, ksize)
    x = torch.randn(1, C, 12, 12)

    runtime = ops.createConvFunc(op)
    if op == "rd":
        # rd sets its own padding=2*dilation internally.
        y_raw = runtime(x, W, None, 1, 2, 1, 1)
        y_conv = F.conv2d(x, convert_pdc(op, W.clone()), padding=2)
    else:  # cd, ad: padding == dilation == 1
        y_raw = runtime(x, W, None, 1, 1, 1, 1)
        y_conv = F.conv2d(x, convert_pdc(op, W.clone()), padding=1)

    diff = (y_raw - y_conv).abs().max().item()
    print(f"  {op}: max abs diff = {diff:.3e}  {'OK' if diff < 1e-5 else 'FAIL'}")
    return diff < 1e-5


if __name__ == "__main__":
    print(f"verifying convert_pdc against {REPO_OPS}")
    ok = all([check("cd"), check("ad"), check("rd")])
    print("ALL OK" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)
