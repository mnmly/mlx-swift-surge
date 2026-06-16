// Shared inference driver for SurGe.
//
// All non-presentation work — model load, per-image inference, defensive
// `MLX.eval`, benchmarking — lives here so the CLI (`surge-bench`) and the
// SwiftUI app (`SURGEDemo`) drive the *same* engine. Each frontend owns only
// its loop, cadence, and presentation surface.
//
// `SurGeSession` is `@unchecked Sendable`: the contract is "one caller drives it
// at a time." The CLI is single-threaded; the app runs it inside a single
// detached Task while the UI reads only the Sendable `SurGeFrame` snapshots it
// hops back to the main actor. It is deliberately NOT `@MainActor` so the
// synchronous CLI can use it directly.

import CoreGraphics
import Foundation
import MLX

/// All knobs for a session. Defaults work for both frontends.
public struct SurGeSessionConfig: Sendable {
    public var weightsPath: String
    public var dtype: DType
    public var useFusedKernel: Bool
    public var tokens: SurGeTokens
    public var forceProjection: Bool
    public var fovX: Float?

    public init(
        weightsPath: String,
        dtype: DType = .float32,
        useFusedKernel: Bool = true,
        tokens: SurGeTokens = .min,
        forceProjection: Bool = true,
        fovX: Float? = nil
    ) {
        self.weightsPath = weightsPath
        self.dtype = dtype
        self.useFusedKernel = useFusedKernel
        self.tokens = tokens
        self.forceProjection = forceProjection
        self.fovX = fovX
    }
}

/// A Sendable snapshot of one inference, safe to hand to the main actor. Holds
/// host-side `[Float]` depth (no `MLXArray`) plus a normalized colormap helper.
public struct SurGeFrame: Sendable {
    public let width: Int
    public let height: Int
    /// Row-major depth, `width * height`; non-finite where masked.
    public let depth: [Float]
    public let depthMin: Float
    public let depthMax: Float
    public let fx: Float
    public let fy: Float
    public let inferenceSeconds: Double

    /// Normalized inverse-depth grayscale RGBA (`width * height * 4`), nearer =
    /// brighter; masked pixels transparent. Built off the main actor so the UI
    /// only wraps the bytes in a `CGImage`.
    public func depthRGBA() -> [UInt8] {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        // Normalize on inverse depth for a perceptually nicer ramp.
        var lo = Float.greatestFiniteMagnitude
        var hi = -Float.greatestFiniteMagnitude
        for d in depth where d.isFinite && d > 0 {
            let inv = 1 / d
            lo = min(lo, inv); hi = max(hi, inv)
        }
        let span = (hi > lo) ? (hi - lo) : 1
        for i in 0..<(width * height) {
            let d = depth[i]
            if d.isFinite && d > 0 {
                let t = ((1 / d) - lo) / span
                let g = UInt8(max(0, min(255, t * 255)))
                rgba[i * 4 + 0] = g
                rgba[i * 4 + 1] = g
                rgba[i * 4 + 2] = g
                rgba[i * 4 + 3] = 255
            }
        }
        return rgba
    }
}

/// Timing summary from ``SurGeSession/benchmark(image:warmup:iterations:)``.
public struct SurGeBenchResult: Sendable {
    public let iterations: Int
    public let meanSeconds: Double
    public let medianSeconds: Double
    public let minSeconds: Double
    public let maxSeconds: Double
    public let activeMemoryDeltaBytes: Int
    public let peakMemoryBytes: Int
}

public final class SurGeSession: @unchecked Sendable {
    public let config: SurGeSessionConfig
    public let model: SurGeModel

    /// Wrap an already-constructed model (used by tests).
    public init(model: SurGeModel, config: SurGeSessionConfig) {
        self.model = model
        self.config = config
    }

    /// Load weights from `config.weightsPath` (a HuggingFace snapshot dir with
    /// `config.json` + `model.safetensors`). Heavy; call once per run.
    public static func load(_ config: SurGeSessionConfig) throws -> SurGeSession {
        let model = try SurGeModel.fromPretrained(
            path: config.weightsPath, dtype: config.dtype, useFusedKernel: config.useFusedKernel)
        MLX.eval(model)
        return SurGeSession(model: model, config: config)
    }

    /// Run inference and materialize the outputs. The shared compute path —
    /// both frontends and the benchmark funnel through it.
    ///
    /// - `fovX`: known horizontal field of view in **radians** (e.g. derived
    ///   from EXIF). Overrides `config.fovX`; when both are nil the focal is
    ///   estimated. A known FoV fixes the focal and solves shift-only — more
    ///   accurate perspective + intrinsics.
    @discardableResult
    public func inferArrays(_ image: MLXArray, fovX: Float? = nil) -> [String: MLXArray] {
        let out = model.infer(
            image: image, tokens: config.tokens,
            resizeOutput: true, forceProjection: config.forceProjection,
            fovX: fovX ?? config.fovX)
        MLX.eval(Array(out.values))
        return out
    }

    /// Run inference and build a Sendable ``SurGeFrame`` snapshot (for the GUI).
    public func infer(_ image: MLXArray, fovX: Float? = nil) -> SurGeFrame {
        let start = CFAbsoluteTimeGetCurrent()
        let out = inferArrays(image, fovX: fovX)
        let seconds = CFAbsoluteTimeGetCurrent() - start

        let depthArr = out["depth"]!.asType(.float32)
        let h = depthArr.dim(depthArr.ndim - 2)
        let w = depthArr.dim(depthArr.ndim - 1)
        let depth: [Float] = depthArr.asArray(Float.self)

        var lo = Float.greatestFiniteMagnitude
        var hi = -Float.greatestFiniteMagnitude
        for d in depth where d.isFinite {
            lo = min(lo, d); hi = max(hi, d)
        }

        let intr = out["intrinsics"]!.asType(.float32)
        let intrFlat: [Float] = intr.asArray(Float.self)  // [.., 3, 3], take first
        let fx = intrFlat.count >= 1 ? intrFlat[0] : 0
        let fy = intrFlat.count >= 5 ? intrFlat[4] : 0

        return SurGeFrame(
            width: w, height: h, depth: depth,
            depthMin: lo.isFinite ? lo : 0, depthMax: hi.isFinite ? hi : 0,
            fx: fx, fy: fy, inferenceSeconds: seconds)
    }

    /// Convenience: decode a `CGImage` to NHWC and infer.
    public func infer(_ cgImage: CGImage) -> SurGeFrame {
        infer(cgImageToNHWC(cgImage).expandedDimensions(axis: 0))
    }

    /// Time `iterations` runs after `warmup`, watching MLX GPU memory. Used by
    /// the CLI; identical compute path as the GUI's `infer`.
    public func benchmark(image: MLXArray, warmup: Int, iterations: Int) -> SurGeBenchResult {
        for _ in 0..<max(0, warmup) { inferArrays(image) }
        let memBefore = Memory.activeMemory
        var times: [Double] = []
        for _ in 0..<max(1, iterations) {
            let s = CFAbsoluteTimeGetCurrent()
            inferArrays(image)
            times.append(CFAbsoluteTimeGetCurrent() - s)
        }
        let memAfter = Memory.activeMemory
        let sorted = times.sorted()
        return SurGeBenchResult(
            iterations: times.count,
            meanSeconds: times.reduce(0, +) / Double(times.count),
            medianSeconds: sorted[sorted.count / 2],
            minSeconds: sorted.first ?? 0,
            maxSeconds: sorted.last ?? 0,
            activeMemoryDeltaBytes: Int(memAfter) - Int(memBefore),
            peakMemoryBytes: Memory.peakMemory)
    }
}
