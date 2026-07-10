import Foundation
import MLX
#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
#endif

/// Minimal ImageIO-backed image load/save for the pipeline. Apple-native, no
/// external image dependency.
public enum ImageIOHelper {
    #if canImport(CoreGraphics)

    /// Load an RGB image as an NHWC `[1,H,W,3]` float tensor in `[0,1]`.
    public static func loadRGB(url: URL) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw MatchEdError.imageLoad(url) }

        let w = cg.width
        let h = cg.height
        var buffer = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &buffer, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw MatchEdError.imageLoad(url) }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        // Drop alpha, scale to [0,1].
        var rgb = [Float](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[i * 3 + 0] = Float(buffer[i * 4 + 0]) / 255.0
            rgb[i * 3 + 1] = Float(buffer[i * 4 + 1]) / 255.0
            rgb[i * 3 + 2] = Float(buffer[i * 4 + 2]) / 255.0
        }
        return MLXArray(rgb, [1, h, w, 3])
    }

    /// Convert a single-channel `[H,W]` / `[1,H,W,1]` float map in `[0,1]` to a
    /// grayscale `CGImage` (for on-screen display; `CGImage` is presentation-
    /// neutral and safe to expose from the library).
    public static func cgImageGray(_ map: MLXArray) -> CGImage? {
        let shape = map.shape
        let (h, w): (Int, Int)
        if shape.count == 4 { (h, w) = (shape[1], shape[2]) }
        else if shape.count == 2 { (h, w) = (shape[0], shape[1]) }
        else { return nil }

        let m = map.reshaped([-1])
        eval(m)
        let values = m.asArray(Float.self)
        var bytes = [UInt8](repeating: 0, count: h * w)
        for i in 0..<(h * w) {
            bytes[i] = UInt8(max(0, min(255, (values[i] * 255.0).rounded())))
        }
        let cs = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(
            data: &bytes, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w, space: cs, bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { return nil }
        return ctx.makeImage()
    }

    /// Save a single-channel `[H,W]` (or `[1,H,W,1]`) float map in `[0,1]` as a
    /// grayscale PNG.
    public static func saveGray(_ map: MLXArray, url: URL) throws {
        let m = map.reshaped([-1])
        eval(m)
        let values = m.asArray(Float.self)
        let total = values.count
        // Infer H×W from the input shape.
        let shape = map.shape
        let (h, w): (Int, Int)
        if shape.count == 4 { (h, w) = (shape[1], shape[2]) }
        else if shape.count == 2 { (h, w) = (shape[0], shape[1]) }
        else { throw MatchEdError.imageSave(url) }
        precondition(h * w == total)

        var bytes = [UInt8](repeating: 0, count: total)
        for i in 0..<total {
            bytes[i] = UInt8(max(0, min(255, (values[i] * 255.0).rounded())))
        }
        let cs = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(
            data: &bytes, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w, space: cs,
            bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let cg = ctx.makeImage()
        else { throw MatchEdError.imageSave(url) }

        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw MatchEdError.imageSave(url) }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw MatchEdError.imageSave(url) }
    }

    #endif
}

public enum MatchEdError: Error, CustomStringConvertible {
    case imageLoad(URL)
    case imageSave(URL)

    public var description: String {
        switch self {
        case .imageLoad(let u): return "failed to load image at \(u.path)"
        case .imageSave(let u): return "failed to save image at \(u.path)"
        }
    }
}
