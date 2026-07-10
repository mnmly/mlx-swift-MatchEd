import Foundation
import MLX
import MLXNN

/// Converted PiDiNet residual block (`models/pidinet.py::PDCBlock_converted`).
///
/// `conv1` is a depthwise (`groups = inplane`) spatial conv whose kernel is 5×5
/// for an `rd` op and 3×3 otherwise; `conv2` is a 1×1 channel mixer. For a
/// stride-2 block the input is max-pooled first and a 1×1 `shortcut` projects
/// the residual to the output width.
public final class PDCBlock: Module, UnaryLayer {
    let stride: Int
    let hasShortcut: Bool

    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "shortcut") var shortcut: Conv2d

    let pool: MaxPool2d
    let relu = ReLU()

    public init(pdc: PDCType, inplane: Int, ouplane: Int, stride: Int = 1) {
        self.stride = stride
        self.hasShortcut = stride > 1
        self.pool = MaxPool2d(kernelSize: 2, stride: 2)

        self._conv1 = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inplane, outputChannels: inplane,
                kernelSize: .init(pdc.kernelSize), padding: .init(pdc.padding),
                groups: inplane, bias: false),
            key: "conv1")
        self._conv2 = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inplane, outputChannels: ouplane,
                kernelSize: 1, bias: false),
            key: "conv2")
        // Always allocated; only consulted when `hasShortcut`. When stride == 1
        // the checkpoint carries no `shortcut.*` keys so nothing is left unused.
        self._shortcut = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inplane, outputChannels: ouplane,
                kernelSize: 1, bias: true),
            key: "shortcut")
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = input
        if stride > 1 { x = pool(x) }
        var y = conv1(x)
        y = relu(y)
        y = conv2(y)
        let residual = hasShortcut ? shortcut(x) : x
        return y + residual
    }
}

/// Compact Dilation Convolution Module (`CDCM`).
///
/// 1×1 reduce → four parallel dilated 3×3 convs (dilations 5,7,9,11) summed.
public final class CDCM: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2_1") var conv2_1: Conv2d
    @ModuleInfo(key: "conv2_2") var conv2_2: Conv2d
    @ModuleInfo(key: "conv2_3") var conv2_3: Conv2d
    @ModuleInfo(key: "conv2_4") var conv2_4: Conv2d
    let relu = ReLU()

    public init(inChannels: Int, outChannels: Int) {
        self._conv1 = ModuleInfo(
            wrappedValue: Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                 kernelSize: 1, bias: true),
            key: "conv1")
        func dilated(_ d: Int) -> Conv2d {
            Conv2d(inputChannels: outChannels, outputChannels: outChannels,
                   kernelSize: 3, padding: .init(d), dilation: .init(d), bias: false)
        }
        self._conv2_1 = ModuleInfo(wrappedValue: dilated(5), key: "conv2_1")
        self._conv2_2 = ModuleInfo(wrappedValue: dilated(7), key: "conv2_2")
        self._conv2_3 = ModuleInfo(wrappedValue: dilated(9), key: "conv2_3")
        self._conv2_4 = ModuleInfo(wrappedValue: dilated(11), key: "conv2_4")
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let x = conv1(relu(input))
        return conv2_1(x) + conv2_2(x) + conv2_3(x) + conv2_4(x)
    }
}

/// Compact Spatial Attention Module (`CSAM`).
///
/// relu → 1×1 (→4) → 3×3 (→1) → sigmoid, then gates the input.
public final class CSAM: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    let relu = ReLU()

    public init(channels: Int) {
        let mid = 4
        self._conv1 = ModuleInfo(
            wrappedValue: Conv2d(inputChannels: channels, outputChannels: mid,
                                 kernelSize: 1, bias: true),
            key: "conv1")
        self._conv2 = ModuleInfo(
            wrappedValue: Conv2d(inputChannels: mid, outputChannels: 1,
                                 kernelSize: 3, padding: 1, bias: false),
            key: "conv2")
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        var y = relu(input)
        y = conv1(y)
        y = conv2(y)
        y = MLX.sigmoid(y)
        return input * y
    }
}

/// Reduce a feature map to a single-channel edge map with a 1×1 conv.
public final class MapReduce: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d

    public init(channels: Int) {
        self._conv = ModuleInfo(
            wrappedValue: Conv2d(inputChannels: channels, outputChannels: 1,
                                 kernelSize: 1, bias: true),
            key: "conv")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}
