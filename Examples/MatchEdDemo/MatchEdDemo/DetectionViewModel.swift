// DetectionViewModel — the SwiftUI frontend's only stateful piece.
//
// All compute goes through MatchEdKit.MatchEd via a `ModelHost` actor; there is
// no model / conv / weight code here, by design (swift-cli-gui-shared-driver).
//
// Concurrency posture (this target: default MainActor isolation + approachable
// concurrency): the view model is `@MainActor` (implicitly), so it may only be
// touched on the main actor. The heavy MLX inference is confined to `ModelHost`
// — an `actor` — so a plain structured `Task` suffices: `await host.detect(…)`
// hops off the main actor for the compute and resumes back on it to publish the
// result. No `Task.detached`, no manual `MainActor.run`, no `@unchecked
// Sendable` on the library type — the actor expresses the "one detection at a
// time, off the main thread" contract directly.

import SwiftUI
import Observation
import CoreGraphics
import ImageIO
import MatchEdKit

/// Rendered result crossing the actor → main-actor boundary. `CGImage` is
/// immutable and thread-safe but not `Sendable`, so box it (the one place a
/// Sendable escape hatch is warranted, and it's a value type of read-only data).
struct DetectionImages: @unchecked Sendable {
    let edge: CGImage?
    let thin: CGImage?
    let elapsedMs: Double
}

/// Owns the loaded model **off the main actor**. Serializes detections by
/// construction (actor reentrancy is not hit — each call runs to completion
/// before the next) and reuses the model across images so weights load once.
actor ModelHost {
    private var cached: (url: URL, model: MatchEd)?

    func detect(weightsURL: URL, imageURL: URL) throws -> DetectionImages {
        let model: MatchEd
        if let c = cached, c.url == weightsURL {
            model = c.model
        } else {
            model = try MatchEd(weightsURL: weightsURL)
            cached = (weightsURL, model)
        }
        return try autoreleasepool {
            let clock = ContinuousClock()
            let t0 = clock.now
            let out = try model.detect(imageURL: imageURL)
            let dt = clock.now - t0
            let ms = Double(dt.components.seconds) * 1000
                + Double(dt.components.attoseconds) / 1e15
            return DetectionImages(
                edge: ImageIOHelper.cgImageGray(out.fused),
                thin: ImageIOHelper.cgImageGray(out.thin),
                elapsedMs: ms)
        }
    }
}

@Observable
final class DetectionViewModel {
    var inputImage: CGImage?
    var edgeImage: CGImage?
    var thinImage: CGImage?
    var status: String = "Pick weights (.safetensors) and an image."
    var isRunning = false

    @ObservationIgnored var weightsURL: URL?
    @ObservationIgnored var imageURL: URL?
    @ObservationIgnored private let host = ModelHost()
    @ObservationIgnored private var task: Task<Void, Never>?

    var canRun: Bool { weightsURL != nil && imageURL != nil && !isRunning }

    func detect() {
        guard let weightsURL, let imageURL, !isRunning else { return }
        isRunning = true
        status = "Running…"
        edgeImage = nil
        thinImage = nil
        inputImage = Self.loadCGImage(imageURL)

        task = Task { [weak self, host] in
            do {
                let imgs = try await host.detect(weightsURL: weightsURL, imageURL: imageURL)
                guard let self else { return }
                self.isRunning = false
                self.edgeImage = imgs.edge
                self.thinImage = imgs.thin
                self.status = String(format: "Done in %.0f ms.", imgs.elapsedMs)
            } catch {
                guard let self else { return }
                self.isRunning = false
                self.status = "Error: \(error)"
            }
        }
    }

    private static func loadCGImage(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}
