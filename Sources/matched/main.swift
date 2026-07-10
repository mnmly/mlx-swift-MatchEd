import Foundation
import ArgumentParser
import MLX
import MatchEDKit

@main
struct Matched: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "matched",
        abstract: "MatchED crisp edge detection on Apple Silicon (MLX).",
        subcommands: [Run.self, Parity.self, Bench.self],
        defaultSubcommand: Run.self
    )
}

// MARK: - run

struct Run: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Detect edges in an image and write edge / crisp PNGs.")

    @Option(name: .shortAndLong, help: "Converted NHWC safetensors weights.")
    var weights: String

    @Argument(help: "Input image.")
    var image: String

    @Option(name: .shortAndLong, help: "Output directory.")
    var outdir: String = "."

    func run() throws {
        let pipeline = try MatchED(weightsURL: URL(fileURLWithPath: weights))
        let out = try pipeline.detect(imageURL: URL(fileURLWithPath: image))

        let stem = URL(fileURLWithPath: image).deletingPathExtension().lastPathComponent
        let dir = URL(fileURLWithPath: outdir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let edgePath = dir.appendingPathComponent("\(stem)_edge.png")
        let thinPath = dir.appendingPathComponent("\(stem)_thin.png")
        try ImageIOHelper.saveGray(out.fused, url: edgePath)
        try ImageIOHelper.saveGray(out.thin, url: thinPath)
        print("wrote \(edgePath.path)")
        print("wrote \(thinPath.path)")
    }
}

// MARK: - parity

struct Parity: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Compare Swift output against the Python fixture.")

    @Option(name: .shortAndLong, help: "Converted NHWC safetensors weights.")
    var weights: String

    @Option(name: .shortAndLong, help: "Parity fixture safetensors.")
    var fixture: String

    func run() throws {
        let model = PiDiNet()
        try model.loadWeights(url: URL(fileURLWithPath: weights))
        let golden = try MLX.loadArrays(url: URL(fileURLWithPath: fixture))
        let out = model(golden["input"]!)
        eval(out.thin, out.fused)

        func diff(_ a: MLXArray, _ b: MLXArray) -> Float {
            MLX.max(MLX.abs(a - b)).item(Float.self)
        }
        for i in 0..<5 {
            print(String(format: "side%d  Δ = %.3e", i, diff(out.sideOutputs[i], golden["side\(i)"]!)))
        }
        print(String(format: "fused  Δ = %.3e", diff(out.fused, golden["fused"]!)))
        print(String(format: "thin   Δ = %.3e", diff(out.thin, golden["thin"]!)))
    }
}

// MARK: - bench

struct Bench: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Benchmark throughput and watch active memory for leaks.")

    @Option(name: .shortAndLong, help: "Converted NHWC safetensors weights (optional; random if omitted).")
    var weights: String?

    @Option(help: "Input height.") var height: Int = 320
    @Option(help: "Input width.") var width: Int = 480
    @Option(help: "Iterations.") var iters: Int = 50
    @Option(help: "Warmup iterations.") var warmup: Int = 5

    func run() throws {
        let pipeline: MatchED
        if let w = weights {
            pipeline = try MatchED(weightsURL: URL(fileURLWithPath: w))
        } else {
            pipeline = MatchED()  // random init
        }
        // Reuse one input tensor across runs; new randomness each iter is
        // unnecessary for timing and would pollute the leak signal.
        let x = MLXArray.zeros([1, height, width, 3])

        func mb(_ bytes: Int) -> String { String(format: "%.1f MB", Double(bytes) / 1_048_576) }

        for _ in 0..<warmup {
            let out = pipeline.model(pipeline.normalize(x))
            eval(out.thin, out.fused)
        }

        let activeStart = MLX.GPU.activeMemory
        let clock = ContinuousClock()
        var times: [Double] = []
        for i in 0..<iters {
            let start = clock.now
            let out = pipeline.model(pipeline.normalize(x))
            eval(out.thin, out.fused)
            let elapsed = clock.now - start
            times.append(Double(elapsed.components.seconds)
                         + Double(elapsed.components.attoseconds) / 1e18)
            if i % 10 == 0 {
                print("iter \(i): active \(mb(MLX.GPU.activeMemory))")
            }
        }
        let activeEnd = MLX.GPU.activeMemory
        let avg = times.reduce(0, +) / Double(times.count)
        let sorted = times.sorted()
        let median = sorted[sorted.count / 2]
        print(String(format: "\n%dx%d  iters=%d", height, width, iters))
        print(String(format: "avg %.2f ms  median %.2f ms  (%.1f fps)",
                     avg * 1000, median * 1000, 1.0 / median))
        print("active memory: start \(mb(activeStart)) -> end \(mb(activeEnd)) "
              + "(Δ \(mb(activeEnd - activeStart)))")
        print("peak \(mb(MLX.GPU.peakMemory)), cache \(mb(MLX.GPU.cacheMemory))")
        print("flat active Δ ⇒ no leak; large peak is MLX's reusable buffer cache.")
    }
}
