import Foundation
import MLX

/// High-level MatchEd edge-detection pipeline: load model once, run on images.
///
/// This is the shared driver (see swift-cli-gui-shared-driver) consumed by the
/// `matched` CLI and reusable from a SwiftUI app.
///
/// Deliberately **actor-agnostic and not `Sendable`**: the synchronous CLI
/// drives it on one thread, and the SwiftUI app confines one instance inside an
/// actor (`ModelHost`) so it is never touched from two isolation domains at
/// once. Keeping the driver isolation-neutral is what lets both frontends share
/// it (swift-cli-gui-shared-driver); the frontend owns the isolation decision.
public final class MatchEd {
    public let model: PiDiNet
    let dtype: DType

    /// ImageNet normalization used by the MatchEd dataloaders
    /// (`edge_dataloader.py`: ToTensor + Normalize).
    public static let mean: [Float] = [0.485, 0.456, 0.406]
    public static let std: [Float] = [0.229, 0.224, 0.225]

    public init(weightsURL: URL, config: PiDiNetConfig = .matched, dtype: DType = .float32) throws {
        self.dtype = dtype
        self.model = PiDiNet(config)
        try model.loadWeights(url: weightsURL, dtype: dtype)
    }

    /// Construct without weights (random init) — for shape tests / benchmarking.
    public init(config: PiDiNetConfig = .matched, dtype: DType = .float32) {
        self.dtype = dtype
        self.model = PiDiNet(config)
        eval(self.model)
    }

    /// Normalize an NHWC `[B,H,W,3]` tensor in `[0,1]` with ImageNet stats.
    public func normalize(_ rgb01: MLXArray) -> MLXArray {
        let meanA = MLXArray(Self.mean, [1, 1, 1, 3]).asType(dtype)
        let stdA = MLXArray(Self.std, [1, 1, 1, 3]).asType(dtype)
        return (rgb01.asType(dtype) - meanA) / stdA
    }

    /// Run the model on an already-normalized NHWC `[B,H,W,3]` tensor.
    public func callAsFunction(_ normalized: MLXArray) -> PiDiNetOutput {
        model(normalized)
    }

    /// Full pipeline: normalize `[0,1]` RGB → forward → materialized output.
    public func detect(rgb01: MLXArray) -> PiDiNetOutput {
        let out = model(normalize(rgb01))
        eval(out.thin, out.fused)
        return out
    }

    #if canImport(CoreGraphics)
    /// Load an image file, run detection, return the output (edge maps in `[0,1]`).
    public func detect(imageURL: URL) throws -> PiDiNetOutput {
        let rgb = try ImageIOHelper.loadRGB(url: imageURL)
        return detect(rgb01: rgb)
    }
    #endif
}
