import Foundation
import MLX
import MLXNN

/// Bilinear resize an NHWC tensor to an **exact** `(targetH, targetW)`.
///
/// Mirrors PyTorch `F.interpolate(..., mode="bilinear", align_corners=False)`,
/// which takes an explicit output *size*. `MLXNN.Upsample` only accepts a scale
/// *factor* and derives the size as `Int(scale × dim)` (truncation) — for a
/// non-integer scale (e.g. a 87→700 side map) that silently drops a pixel, so
/// side maps that should all be `W` wide come back `W` and `W-1` and fail to
/// concatenate. We instead gather with the exact `align_corners=False` sample
/// coordinates, one axis at a time, guaranteeing the requested size.
public func bilinearResize(_ x: MLXArray, _ targetH: Int, _ targetW: Int) -> MLXArray {
    var out = x
    out = resizeAxis(out, axis: 1, target: targetH)
    out = resizeAxis(out, axis: 2, target: targetW)
    return out
}

private func resizeAxis(_ x: MLXArray, axis: Int, target: Int) -> MLXArray {
    let inN = x.dim(axis)
    if inN == target { return x }
    let ratio = Float(inN) / Float(target)
    // align_corners=False: output index o maps to input coord (o+0.5)·ratio−0.5.
    let o = MLXArray(0..<target).asType(.float32)
    var src = (o + 0.5) * ratio - 0.5
    src = MLX.minimum(MLX.maximum(src, MLXArray(Float(0))), MLXArray(Float(inN - 1)))
    let i0f = MLX.floor(src)
    let i0 = i0f.asType(.int32)
    let i1 = MLX.minimum(i0 + Int32(1), MLXArray(Int32(inN - 1)))
    let frac = src - i0f  // interpolation weight toward i1

    let x0 = MLX.take(x, i0, axis: axis)
    let x1 = MLX.take(x, i1, axis: axis)

    var wshape = [Int](repeating: 1, count: x.ndim)
    wshape[axis] = target
    let w = frac.reshaped(wshape)
    return x0 * (1 - w) + x1 * w
}

/// BatchNorm that always normalizes using the *current input's* statistics
/// (per channel, over N·H·W), i.e. PyTorch `nn.BatchNorm2d(..., track_running_stats=False)`.
///
/// With `track_running_stats=False` PyTorch stores no running mean/var and uses
/// batch statistics even in `eval()` — for MatchED's `SmallUNet` thinner, which
/// always runs at batch size 1, this is a spatial mean/var over the single
/// image. `MLXNN.BatchNorm` cannot express this (it uses running stats in
/// eval), so we implement it directly on NHWC tensors.
public final class InstanceStatBatchNorm: Module, UnaryLayer {
    public let weight: MLXArray   // affine gamma, shape [C]
    public let bias: MLXArray     // affine beta,  shape [C]
    let eps: Float

    public init(channels: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([channels])
        self.bias = MLXArray.zeros([channels])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // NHWC → reduce over N, H, W (axes 0,1,2), keep channel (axis 3).
        let axes = [0, 1, 2]
        let mean = MLX.mean(x, axes: axes, keepDims: true)
        let variance = MLX.variance(x, axes: axes, keepDims: true)
        let normed = (x - mean) / MLX.sqrt(variance + eps)
        return normed * weight + bias
    }
}
