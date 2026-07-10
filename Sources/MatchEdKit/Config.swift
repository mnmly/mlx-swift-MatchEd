import Foundation

/// Pixel-difference convolution op type for one layer.
///
/// In the *converted* model (what this port runs) the op type only affects the
/// spatial kernel size of the depthwise `conv1`: `rd` becomes a 5×5 kernel,
/// every other op a 3×3 kernel. The value-level difference (which entries of
/// the kernel are summed/subtracted) is baked into the weights at conversion
/// time in Python — see `Scripts/convert_weights.py`.
public enum PDCType: String, Sendable {
    case cv  // vanilla
    case cd  // central difference
    case ad  // angular difference
    case rd  // radial difference (5×5 after conversion)

    /// Converted kernel size for a `conv1` using this op.
    public var kernelSize: Int { self == .rd ? 5 : 3 }
    /// Padding that keeps spatial size fixed for `kernelSize`.
    public var padding: Int { self == .rd ? 2 : 1 }
}

/// A named PiDiNet PDC configuration (the 16 per-layer op types).
///
/// Mirrors `models/config.py::nets`. Only the shipped `carv4` config is needed
/// for the MatchEd checkpoints, but the table is trivial to extend.
public struct PDCConfig: Sendable {
    /// 16 op types: index 0 is `init_block`, indices 1…15 are the block
    /// `conv1`s in trunk order (block1_1 … block4_4).
    public let pdcs: [PDCType]

    public init(pdcs: [PDCType]) {
        precondition(pdcs.count == 16, "PiDiNet needs exactly 16 PDC ops")
        self.pdcs = pdcs
    }

    /// `carv4`: cd, ad, rd, cv repeated across the four stages.
    public static let carv4 = PDCConfig(pdcs: [
        .cd, .ad, .rd, .cv,
        .cd, .ad, .rd, .cv,
        .cd, .ad, .rd, .cv,
        .cd, .ad, .rd, .cv,
    ])

    public static func named(_ name: String) -> PDCConfig {
        switch name {
        case "carv4": return .carv4
        default: fatalError("unknown PDC config \(name)")
        }
    }
}

/// Top-level PiDiNet architecture parameters.
///
/// Defaults match the MatchEd release command
/// (`--config carv4 --sa --dil`, base width 60, `dil=24`).
public struct PiDiNetConfig: Sendable {
    public let inplane: Int
    public let dil: Int?          // CDCM channels; nil disables dilation module
    public let sa: Bool           // CSAM spatial attention
    public let pdc: PDCConfig

    public init(inplane: Int = 60, dil: Int? = 24, sa: Bool = true, pdc: PDCConfig = .carv4) {
        self.inplane = inplane
        self.dil = dil
        self.sa = sa
        self.pdc = pdc
    }

    public static let matched = PiDiNetConfig()

    /// The four fuse-stage channel counts: [C, 2C, 4C, 4C].
    public var fusePlanes: [Int] {
        [inplane, inplane * 2, inplane * 4, inplane * 4]
    }
}
