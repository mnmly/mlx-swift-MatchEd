import Foundation
import MLX
import MLXNN

extension PiDiNet {
    /// Load a converted MatchEd checkpoint (see `Scripts/convert_weights.py`).
    ///
    /// The safetensors file must already be:
    ///   * PDC-converted (pixel-difference kernels folded into vanilla convs), and
    ///   * NHWC — conv weights transposed from PyTorch `(O,I,kH,kW)` to
    ///     `(O,kH,kW,I)`.
    ///
    /// Keys line up 1:1 with the PyTorch `state_dict`, so no remapping is
    /// needed. `verify: [.noUnusedKeys]` catches any stray/misnamed key.
    /// (Stride-1 blocks legitimately carry no `shortcut.*` params; their
    /// always-allocated shortcut conv stays at init and is never called.)
    public func loadWeights(url: URL, dtype: DType = .float32) throws {
        let raw = try MLX.loadArrays(url: url)
        var params: [(String, MLXArray)] = []
        params.reserveCapacity(raw.count)
        for (key, value) in raw {
            params.append((key, value.asType(dtype)))
        }
        try update(parameters: ModuleParameters.unflattened(params),
                   verify: [.noUnusedKeys])
        eval(self)
    }
}
