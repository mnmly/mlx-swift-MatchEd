import Foundation
import MLX
import MLXNN

/// The MatchED "thinner" (`models/pidinet.py::SmallUNet`).
///
/// A tiny 1-channel → 1-channel refinement net (enc1 → enc2 → dec1, no skip
/// connections despite the name) applied to the fused edge map to produce the
/// *crisp* output. Every `BatchNorm2d` uses `track_running_stats=False`, so it
/// normalizes with batch statistics — see `InstanceStatBatchNorm`. Output is
/// sigmoid-activated.
///
/// Weight keys mirror the PyTorch `nn.Sequential`s: `enc1.0` (conv), `enc1.1`
/// (bn), `enc1.3` (conv), `enc1.4` (bn), etc. ReLUs occupy the param-less
/// indices 2 and 5 so the numbering stays aligned with the checkpoint.
public final class SmallUNet: Module {
    @ModuleInfo(key: "enc1") var enc1: [UnaryLayer]
    @ModuleInfo(key: "enc2") var enc2: [UnaryLayer]
    @ModuleInfo(key: "dec1") var dec1: [UnaryLayer]

    public override init() {
        func conv(_ cin: Int, _ cout: Int) -> Conv2d {
            Conv2d(inputChannels: cin, outputChannels: cout, kernelSize: 3, padding: 1, bias: true)
        }
        self._enc1 = ModuleInfo(wrappedValue: [
            conv(1, 16), InstanceStatBatchNorm(channels: 16), ReLU(),
            conv(16, 16), InstanceStatBatchNorm(channels: 16), ReLU(),
        ], key: "enc1")
        self._enc2 = ModuleInfo(wrappedValue: [
            conv(16, 32), InstanceStatBatchNorm(channels: 32), ReLU(),
            conv(32, 32), InstanceStatBatchNorm(channels: 32), ReLU(),
        ], key: "enc2")
        self._dec1 = ModuleInfo(wrappedValue: [
            conv(32, 16), InstanceStatBatchNorm(channels: 16), ReLU(),
            conv(16, 1),
        ], key: "dec1")
        super.init()
    }

    private func run(_ x: MLXArray, _ layers: [UnaryLayer]) -> MLXArray {
        var out = x
        for layer in layers { out = layer(out) }
        return out
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = run(run(run(x, enc1), enc2), dec1)
        return MLX.sigmoid(out)
    }
}
