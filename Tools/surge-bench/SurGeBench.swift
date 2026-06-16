#if os(macOS)
import AppKit
import ArgumentParser
import CoreGraphics
import Foundation
import MLX
import MLXSurGe

@main
struct SurGeBench: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "surge-bench",
        abstract: "Benchmark MLX Swift SurGe inference and check for leaks."
    )

    @Option(name: .shortAndLong, help: "Path to HuggingFace snapshot (config.json + model.safetensors).")
    var weights: String

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

    func parseTokens() -> SurGeTokens {
        switch tokens.lowercased() {
        case "min": return .min
        case "max": return .max
        default: return .count(Int(tokens) ?? 1024)
        }
    }

    func run() throws {
        let targetDtype: DType = (dtype == "float16") ? .float16 : .float32

        if cacheLimitMB > 0 {
            GPU.set(cacheLimit: cacheLimitMB * 1024 * 1024)
        }

        let loadStart = CFAbsoluteTimeGetCurrent()
        let model = try SurGeModel.fromPretrained(
            path: weights, dtype: targetDtype, useFusedKernel: fused)
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

        let tok = parseTokens()
        let runOnce: () -> Void = {
            let out = model.infer(
                image: image, tokens: tok, resizeOutput: true,
                forceProjection: forceProjection)
            eval(Array(out.values))
        }

        for _ in 0..<max(0, warmup) { runOnce() }

        let memBefore = GPU.activeMemory
        var times: [Double] = []
        for _ in 0..<max(1, iterations) {
            let start = CFAbsoluteTimeGetCurrent()
            runOnce()
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }
        let memAfter = GPU.activeMemory

        let mean = times.reduce(0, +) / Double(times.count)
        let sorted = times.sorted()
        let median = sorted[sorted.count / 2]

        print("backend=mlx-swift")
        print("model=karimknaebel/surge-large")
        print("dtype=\(dtype)")
        print("source_size=\(W)x\(H)")
        print("tokens=\(tokens)")
        print("fused_kernel=\(fused)")
        print("force_projection=\(forceProjection)")
        print(String(format: "load_s=%.6f", loadSeconds))
        print("warmup=\(warmup)")
        print("iterations=\(times.count)")
        print(String(format: "mean_s=%.6f", mean))
        print(String(format: "median_s=%.6f", median))
        print(String(format: "min_s=%.6f", sorted.first ?? 0))
        print(String(format: "max_s=%.6f", sorted.last ?? 0))
        print(String(format: "active_mem_before_mb=%.1f", Double(memBefore) / 1e6))
        print(String(format: "active_mem_after_mb=%.1f", Double(memAfter) / 1e6))
        print(String(format: "active_mem_delta_mb=%.1f", Double(Int(memAfter) - Int(memBefore)) / 1e6))
        print(String(format: "peak_mem_mb=%.1f", Double(GPU.peakMemory) / 1e6))
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
    return (cgImageToNHWC(cg), cg.height, cg.width)
}

#else
@main
struct SurGeBench {
    static func main() {
        fatalError("surge-bench is only available on macOS")
    }
}
#endif
