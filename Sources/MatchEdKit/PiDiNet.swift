import Foundation
import MLX
import MLXNN

/// Full inference output of the MatchEd model.
public struct PiDiNetOutput {
    /// Five sigmoid-activated side edge maps, NHWC `[B,H,W,1]`:
    /// `e1…e4` (per-stage) followed by the fused `output`.
    public let sideOutputs: [MLXArray]
    /// The crisp refined edge map from the `SmallUNet` thinner, `[B,H,W,1]`.
    public let thin: MLXArray

    /// The primary fused edge map (`sideOutputs.last`).
    public var fused: MLXArray { sideOutputs[4] }
}

/// MatchEd / PiDiNet edge-detection network (converted vanilla-conv form).
///
/// Ported 1:1 from `models/pidinet.py::PiDiNet` for the shipped config
/// (`carv4`, `--sa --dil`, base width 60, `dil=24`). All pixel-difference ops
/// are pre-converted to vanilla convolutions at weight-load time, so every
/// layer here is a plain `Conv2d`. Tensors are NHWC throughout.
public final class PiDiNet: Module {
    let config: PiDiNetConfig

    @ModuleInfo(key: "init_block") var initBlock: Conv2d

    @ModuleInfo(key: "block1_1") var block1_1: PDCBlock
    @ModuleInfo(key: "block1_2") var block1_2: PDCBlock
    @ModuleInfo(key: "block1_3") var block1_3: PDCBlock

    @ModuleInfo(key: "block2_1") var block2_1: PDCBlock
    @ModuleInfo(key: "block2_2") var block2_2: PDCBlock
    @ModuleInfo(key: "block2_3") var block2_3: PDCBlock
    @ModuleInfo(key: "block2_4") var block2_4: PDCBlock

    @ModuleInfo(key: "block3_1") var block3_1: PDCBlock
    @ModuleInfo(key: "block3_2") var block3_2: PDCBlock
    @ModuleInfo(key: "block3_3") var block3_3: PDCBlock
    @ModuleInfo(key: "block3_4") var block3_4: PDCBlock

    @ModuleInfo(key: "block4_1") var block4_1: PDCBlock
    @ModuleInfo(key: "block4_2") var block4_2: PDCBlock
    @ModuleInfo(key: "block4_3") var block4_3: PDCBlock
    @ModuleInfo(key: "block4_4") var block4_4: PDCBlock

    @ModuleInfo(key: "dilations") var dilations: [CDCM]
    @ModuleInfo(key: "attentions") var attentions: [CSAM]
    @ModuleInfo(key: "conv_reduces") var convReduces: [MapReduce]

    @ModuleInfo(key: "classifier") var classifier: Conv2d
    @ModuleInfo(key: "thinner") var thinner: SmallUNet

    public init(_ config: PiDiNetConfig = .matched) {
        precondition(config.sa && config.dil != nil,
                     "This port targets the shipped MatchEd config (--sa --dil).")
        self.config = config
        let pdcs = config.pdc.pdcs
        let c = config.inplane
        let dil = config.dil!

        // init_block: 3 → C, kernel from pdcs[0] (cd → 3×3), no bias.
        self._initBlock = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: 3, outputChannels: c,
                kernelSize: .init(pdcs[0].kernelSize), padding: .init(pdcs[0].padding),
                bias: false),
            key: "init_block")

        func blk(_ i: Int, _ cin: Int, _ cout: Int, stride: Int = 1) -> PDCBlock {
            PDCBlock(pdc: pdcs[i], inplane: cin, ouplane: cout, stride: stride)
        }
        // Stage 1: C → C.
        self._block1_1 = ModuleInfo(wrappedValue: blk(1, c, c), key: "block1_1")
        self._block1_2 = ModuleInfo(wrappedValue: blk(2, c, c), key: "block1_2")
        self._block1_3 = ModuleInfo(wrappedValue: blk(3, c, c), key: "block1_3")
        // Stage 2: C → 2C (stride-2 entry).
        self._block2_1 = ModuleInfo(wrappedValue: blk(4, c, 2 * c, stride: 2), key: "block2_1")
        self._block2_2 = ModuleInfo(wrappedValue: blk(5, 2 * c, 2 * c), key: "block2_2")
        self._block2_3 = ModuleInfo(wrappedValue: blk(6, 2 * c, 2 * c), key: "block2_3")
        self._block2_4 = ModuleInfo(wrappedValue: blk(7, 2 * c, 2 * c), key: "block2_4")
        // Stage 3: 2C → 4C (stride-2 entry).
        self._block3_1 = ModuleInfo(wrappedValue: blk(8, 2 * c, 4 * c, stride: 2), key: "block3_1")
        self._block3_2 = ModuleInfo(wrappedValue: blk(9, 4 * c, 4 * c), key: "block3_2")
        self._block3_3 = ModuleInfo(wrappedValue: blk(10, 4 * c, 4 * c), key: "block3_3")
        self._block3_4 = ModuleInfo(wrappedValue: blk(11, 4 * c, 4 * c), key: "block3_4")
        // Stage 4: 4C → 4C (stride-2 entry, width unchanged).
        self._block4_1 = ModuleInfo(wrappedValue: blk(12, 4 * c, 4 * c, stride: 2), key: "block4_1")
        self._block4_2 = ModuleInfo(wrappedValue: blk(13, 4 * c, 4 * c), key: "block4_2")
        self._block4_3 = ModuleInfo(wrappedValue: blk(14, 4 * c, 4 * c), key: "block4_3")
        self._block4_4 = ModuleInfo(wrappedValue: blk(15, 4 * c, 4 * c), key: "block4_4")

        let fuse = config.fusePlanes
        self._dilations = ModuleInfo(
            wrappedValue: (0..<4).map { CDCM(inChannels: fuse[$0], outChannels: dil) },
            key: "dilations")
        self._attentions = ModuleInfo(
            wrappedValue: (0..<4).map { _ in CSAM(channels: dil) },
            key: "attentions")
        self._convReduces = ModuleInfo(
            wrappedValue: (0..<4).map { _ in MapReduce(channels: dil) },
            key: "conv_reduces")

        self._classifier = ModuleInfo(
            wrappedValue: Conv2d(inputChannels: 4, outputChannels: 1, kernelSize: 1, bias: true),
            key: "classifier")
        self._thinner = ModuleInfo(wrappedValue: SmallUNet(), key: "thinner")
        super.init()
    }

    /// Forward pass. `x` is NHWC `[B,H,W,3]`, already ImageNet-normalized.
    public func callAsFunction(_ x: MLXArray) -> PiDiNetOutput {
        let h = x.dim(1)
        let w = x.dim(2)

        let x0 = initBlock(x)

        var x1 = block1_1(x0); x1 = block1_2(x1); x1 = block1_3(x1)
        var x2 = block2_1(x1); x2 = block2_2(x2); x2 = block2_3(x2); x2 = block2_4(x2)
        var x3 = block3_1(x2); x3 = block3_2(x3); x3 = block3_3(x3); x3 = block3_4(x3)
        var x4 = block4_1(x3); x4 = block4_2(x4); x4 = block4_3(x4); x4 = block4_4(x4)

        let stages = [x1, x2, x3, x4]
        var edges: [MLXArray] = []
        for i in 0..<4 {
            let fused = attentions[i](dilations[i](stages[i]))
            var e = convReduces[i](fused)
            e = bilinearResize(e, h, w)
            edges.append(e)
        }

        // classifier over the 4 stacked side maps (channel axis = 3 in NHWC).
        let stacked = MLX.concatenated(edges, axis: 3)
        let output = classifier(stacked)

        var sideOutputs = edges
        sideOutputs.append(output)
        sideOutputs = sideOutputs.map { MLX.sigmoid($0) }

        let thin = thinner(sideOutputs[4])
        return PiDiNetOutput(sideOutputs: sideOutputs, thin: thin)
    }
}
