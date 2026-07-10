# ``MatchEdKit``

Crisp edge detection on Apple Silicon — an MLX port of MatchEd / PiDiNet.

## Overview

`MatchEdKit` runs the MatchEd edge-detection network (PiDiNet trunk with CDCM /
CSAM / MapReduce side heads, plus a `SmallUNet` "thinner" for the crisp output)
on Apple Silicon via [MLX](https://github.com/ml-explore/mlx-swift). Tensors are
NHWC throughout and the model is plain grouped ``PiDiNet`` convolutions — the
pixel-difference kernels are folded into vanilla convs once, at weight-load
time, so there is no custom Metal.

``MatchEd`` is the high-level pipeline: load a converted checkpoint once, then
run detection on images. It is the shared driver behind both the `matched` CLI
and the `Examples/MatchEdDemo` SwiftUI app.

```swift
import MatchEdKit

let matched = try MatchEd(weightsURL: weightsURL)
let out = try matched.detect(imageURL: imageURL)
try ImageIOHelper.saveGray(out.thin, url: crispURL)   // refined crisp edges
try ImageIOHelper.saveGray(out.fused, url: edgeURL)   // fused side map
```

Weights come from a raw PiDiNet `.pth` checkpoint converted to NHWC safetensors
by `Scripts/convert_weights.py` (or `make_real_fixture.py`). The Swift loader
(``PiDiNet/loadWeights(url:dtype:)``) needs no key remapping.

## Topics

### Running detection

- ``MatchEd``
- ``PiDiNetOutput``
- ``MatchEdError``

### The model

- ``PiDiNet``
- ``PiDiNetConfig``
- ``PDCConfig``
- ``PDCType``

### Building blocks

- ``PDCBlock``
- ``CDCM``
- ``CSAM``
- ``MapReduce``
- ``SmallUNet``
- ``InstanceStatBatchNorm``

### Images and ops

- ``ImageIOHelper``
- ``bilinearResize(_:_:_:)``
