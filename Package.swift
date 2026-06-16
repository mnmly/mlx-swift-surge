// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "mlx-swift-surge",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
        .tvOS(.v17),
    ],
    products: [
        .library(
            name: "MLXSurGe",
            targets: ["MLXSurGe"]
        ),
        .executable(name: "surge-bench", targets: ["surge-bench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),
    ],
    targets: [
        .target(
            name: "MLXSurGe",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ],
            path: "Sources/MLXSurGe"
        ),
        .executableTarget(
            name: "surge-bench",
            dependencies: [
                "MLXSurGe",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Examples/surge-bench"
        ),
        .testTarget(
            name: "MLXSurGeTests",
            dependencies: ["MLXSurGe"],
            path: "Tests/MLXSurGeTests"
        ),
    ]
)
