import XCTest
import MLX
@testable import MatchEdKit

/// End-to-end numerical parity against the CPU PyTorch reference
/// (`Scripts/matched_ref.py`). Fixture produced by
/// `Scripts/make_parity_fixture.py`.
final class ParityTests: XCTestCase {
    /// Package root, derived from this file's path.
    static var root: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MatchEdKitTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
    }

    func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.max(MLX.abs(a - b)).item(Float.self)
    }

    func testEndToEndParity() throws {
        let weights = Self.root.appendingPathComponent("weights/matched_random.safetensors")
        let fixture = Self.root.appendingPathComponent("fixtures/parity.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: weights.path),
                          "run Scripts/make_parity_fixture.py first")

        let model = PiDiNet()
        try model.loadWeights(url: weights)

        let golden = try MLX.loadArrays(url: fixture)
        let input = golden["input"]!
        let out = model(input)
        eval(out.thin, out.fused)

        // fp32 conv/interpolate accumulation tolerance.
        let tol: Float = 2e-4
        for i in 0..<5 {
            let d = maxAbsDiff(out.sideOutputs[i], golden["side\(i)"]!)
            XCTAssertLessThan(d, tol, "side\(i) max abs diff \(d)")
        }
        let df = maxAbsDiff(out.fused, golden["fused"]!)
        let dt = maxAbsDiff(out.thin, golden["thin"]!)
        print("parity: fused Δ=\(df)  thin Δ=\(dt)")
        XCTAssertLessThan(df, tol, "fused max abs diff \(df)")
        XCTAssertLessThan(dt, tol, "thin max abs diff \(dt)")
    }

    func testShapes() throws {
        let model = PiDiNet()
        eval(model)
        let x = MLXArray.zeros([1, 48, 64, 3])
        let out = model(x)
        eval(out.thin, out.fused)
        XCTAssertEqual(out.sideOutputs.count, 5)
        for s in out.sideOutputs { XCTAssertEqual(s.shape, [1, 48, 64, 1]) }
        XCTAssertEqual(out.thin.shape, [1, 48, 64, 1])
    }

    /// Regression: an odd input size whose stage-4 map does NOT rescale by an
    /// integer factor. The old `Upsample(scaleFactor:)` path produced a
    /// W-vs-(W-1) mismatch that crashed the side-map concatenation.
    func testNonIntegerScaleSize() throws {
        let model = PiDiNet()
        eval(model)
        let x = MLXArray.zeros([1, 97, 175, 3])
        let out = model(x)
        eval(out.thin, out.fused)          // would crash in `concatenated` if mismatched
        for s in out.sideOutputs { XCTAssertEqual(s.shape, [1, 97, 175, 1]) }
        XCTAssertEqual(out.thin.shape, [1, 97, 175, 1])
    }
}
