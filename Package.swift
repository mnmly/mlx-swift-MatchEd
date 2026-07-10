// swift-tools-version: 6.0
//
// mlx-swift-MatchED — Apple-Silicon port of MatchED / PiDiNet crisp edge
// detection (CVPR 2026, "MatchED: Crisp Edge Detection Using End-to-End,
// Matching-based Supervision"; edge-detection trunk is PiDiNet, ICCV 2021).
//
// The PiDiNet trunk is a pixel-difference-convolution (PDC) network. Every
// PDC op (cd / ad / rd) is *algebraically identical* to a vanilla convolution
// once its kernel weights are transformed (this is the paper's own
// `--evaluate-converted` path). We do that transform once, in Python, at
// weight-conversion time — so the Swift model is plain grouped `Conv2d`
// everywhere and needs no custom Metal.
//
// The `MatchEDKit` library holds the model + a single `MatchED` pipeline
// driver consumed identically by the `matched` CLI (see
// swift-cli-gui-shared-driver).

import PackageDescription

let package = Package(
    name: "mlx-swift-MatchED",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MatchEDKit", targets: ["MatchEDKit"]),
        .executable(name: "matched", targets: ["matched"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.25.6"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MatchEDKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MatchEDKitTests",
            dependencies: [
                "MatchEDKit",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "matched",
            dependencies: [
                "MatchEDKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ]
)

// Pull in swift-docc-plugin only when generating documentation, so normal
// builds and downstream consumers don't have to resolve an extra dependency.
// Scripts/build_docs.sh exports BUILD_DOC=1; the Swift Package Index sets
// SPI_GENERATE_DOCS automatically.
if Context.environment["SPI_GENERATE_DOCS"] == "1"
    || Context.environment["BUILD_DOC"] == "1"
{
    package.dependencies.append(
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3")
    )
}
