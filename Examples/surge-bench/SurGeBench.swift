#if os(macOS)
import AppKit
import ArgumentParser
import CoreGraphics
import Foundation
import MLX
import MLXSurGe

@main
struct SurGeBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "surge-bench",
        abstract: "Benchmark MLX Swift SurGe inference and check for leaks."
    )

    @Option(name: .shortAndLong, help: "HuggingFace snapshot dir (config.json + model.safetensors). Defaults to the on-device cache.")
    var weights: String?

    @Option(name: .shortAndLong, help: "Input image path (optional; synthetic image used if omitted).")
    var input: String? = nil

    @Option(name: .long, help: "Synthetic image size when no --input is given.")
    var size: Int = 448

    @Option(name: .long, help: "Weight dtype: float16 or float32.")
    var dtype: String = "float32"

    @Option(name: .long, help: "Token budget: min, max, or an integer.")
    var tokens: String = "min"

    @Flag(name: .long, inversion: .prefixedNo, help: "Recompute points from depth + intrinsics.")
    var forceProjection: Bool = true

    @Option(name: .long, help: "Warmup iterations.")
    var warmup: Int = 2

    @Option(name: .long, help: "Measured iterations.")
    var iterations: Int = 10

    @Option(name: .long, help: "Bound the MLX GPU buffer cache (MB). 0 = unbounded.")
    var cacheLimitMB: Int = 0

    @Flag(name: .long, inversion: .prefixedNo, help: "Use the fused neighborhood-attention Metal kernel (default). --no-fused for the gather reference.")
    var fused: Bool = true

    @Flag(name: .long, help: "Download the model into the cache first if it's missing.")
    var download: Bool = false

    func parseTokens() -> SurGeTokens {
        switch tokens.lowercased() {
        case "min": return .min
        case "max": return .max
        default: return .count(Int(tokens) ?? 1024)
        }
    }

    func run() async throws {
        if cacheLimitMB > 0 {
            Memory.cacheLimit = cacheLimitMB * 1024 * 1024
        }

        // Resolve weights: explicit path, else the shared on-device cache.
        let cacheDir = SurGeModelDownloader.defaultCacheDirectory()
        let weightsURL = weights.map { URL(fileURLWithPath: $0) } ?? cacheDir
        if download && !SurGeModelDownloader.isDownloaded(at: weightsURL) {
            print("downloading model to \(weightsURL.path) ...")
            try await SurGeModelDownloader().download(to: weightsURL) { frac in
                print(String(format: "  %.0f%%", frac * 100))
            }
        }

        let cfg = SurGeSessionConfig(
            weightsPath: weightsURL.path,
            dtype: dtype == "float16" ? .float16 : .float32,
            useFusedKernel: fused,
            tokens: parseTokens(),
            forceProjection: forceProjection)

        let loadStart = CFAbsoluteTimeGetCurrent()
        let session = try SurGeSession.load(cfg)
        let loadSeconds = CFAbsoluteTimeGetCurrent() - loadStart

        let image: MLXArray
        let H: Int
        let W: Int
        if let input, let (loaded, h, w) = loadImageAsNHWC(path: input) {
            image = loaded; H = h; W = w
        } else {
            H = size; W = size
            image = MLXRandom.uniform(low: 0.0, high: 1.0, [1, H, W, 3]).asType(.float32)
        }
        eval(image)

        let r = session.benchmark(image: image, warmup: warmup, iterations: iterations)

        print("backend=mlx-swift")
        print("model=karimknaebel/surge-large")
        print("dtype=\(dtype)")
        print("source_size=\(W)x\(H)")
        print("tokens=\(tokens)")
        print("fused_kernel=\(fused)")
        print("force_projection=\(forceProjection)")
        print(String(format: "load_s=%.6f", loadSeconds))
        print("warmup=\(warmup)")
        print("iterations=\(r.iterations)")
        print(String(format: "mean_s=%.6f", r.meanSeconds))
        print(String(format: "median_s=%.6f", r.medianSeconds))
        print(String(format: "min_s=%.6f", r.minSeconds))
        print(String(format: "max_s=%.6f", r.maxSeconds))
        print(String(format: "active_mem_delta_mb=%.1f", Double(r.activeMemoryDeltaBytes) / 1e6))
        print(String(format: "peak_mem_mb=%.1f", Double(r.peakMemoryBytes) / 1e6))
    }
}

private func loadImageAsNHWC(path: String) -> (MLXArray, Int, Int)? {
    let url = URL(fileURLWithPath: path)
    let cgImage: CGImage? = {
        if let provider = CGDataProvider(url: url as CFURL),
           let png = CGImage(
               pngDataProviderSource: provider,
               decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        {
            return png
        }
        return NSImage(contentsOf: url)?.cgImage(forProposedRect: nil, context: nil, hints: nil)
    }()
    guard let cg = cgImage else { return nil }
    return (cgImageToNHWC(cg).expandedDimensions(axis: 0), cg.height, cg.width)
}

#else
@main
struct SurGeBench {
    static func main() {
        fatalError("surge-bench is only available on macOS")
    }
}
#endif
