// High-level pipeline API: CGImage in, geometry out.
//
// Wraps `SurGeModel` so callers don't deal with manual NHWC conversion or
// `infer` flags. Mirrors the shape of `MoGePipeline`.

import CoreGraphics
import Foundation
import MLX

public struct SurGePrediction {
    public let outputs: [String: MLXArray]
    public let sourceHeight: Int
    public let sourceWidth: Int

    public init(outputs: [String: MLXArray], sourceHeight: Int, sourceWidth: Int) {
        self.outputs = outputs
        self.sourceHeight = sourceHeight
        self.sourceWidth = sourceWidth
    }

    public var points: MLXArray? { outputs["points"] }
    public var depth: MLXArray? { outputs["depth"] }
    public var intrinsics: MLXArray? { outputs["intrinsics"] }
}

public struct SurGePipeline {
    public let model: SurGeModel
    public let dtype: DType

    public init(model: SurGeModel, dtype: DType = .float32) {
        self.model = model
        self.dtype = dtype
    }

    /// Convert a `CGImage` to row-major NHWC float32 `[0, 1]`.
    public func preprocess(_ image: CGImage) -> MLXArray {
        cgImageToNHWC(image)
    }

    public func predict(
        _ input: MLXArray,
        tokens: SurGeTokens = .max,
        resizeOutput: Bool = true,
        forceProjection: Bool = true,
        fovX: Float? = nil
    ) -> [String: MLXArray] {
        let outputs = model.infer(
            image: input,
            tokens: tokens,
            resizeOutput: resizeOutput,
            forceProjection: forceProjection,
            fovX: fovX)
        eval(Array(outputs.values))
        return outputs
    }

    public func callAsFunction(
        _ image: CGImage,
        tokens: SurGeTokens = .max,
        resizeOutput: Bool = true,
        forceProjection: Bool = true,
        fovX: Float? = nil
    ) -> SurGePrediction {
        let input = preprocess(image)
        let outputs = predict(
            input, tokens: tokens, resizeOutput: resizeOutput,
            forceProjection: forceProjection, fovX: fovX)
        return SurGePrediction(
            outputs: outputs, sourceHeight: image.height, sourceWidth: image.width)
    }
}

public extension SurGePipeline {
    static func fromPretrained(
        _ path: String, dtype: DType = .float32, useFusedKernel: Bool = true
    ) throws -> SurGePipeline {
        let model = try SurGeModel.fromPretrained(
            path: path, dtype: dtype, useFusedKernel: useFusedKernel)
        return SurGePipeline(model: model, dtype: dtype)
    }

    static func fromPretrained(
        url: URL, dtype: DType = .float32, useFusedKernel: Bool = true
    ) throws -> SurGePipeline {
        try fromPretrained(url.path, dtype: dtype, useFusedKernel: useFusedKernel)
    }
}

public enum MLXSurGe {
    public static func fromPretrained(
        _ path: String, dtype: DType = .float32, useFusedKernel: Bool = true
    ) throws -> SurGePipeline {
        try SurGePipeline.fromPretrained(path, dtype: dtype, useFusedKernel: useFusedKernel)
    }
}

// MARK: - CGImage → NHWC

/// Decode a `CGImage` to a row-major NHWC float32 tensor in `[0, 1]` (sRGB,
/// premultiplied-last RGBA) so the byte layout matches PIL + numpy.
public func cgImageToNHWC(_ image: CGImage) -> MLXArray {
    let W = image.width
    let H = image.height
    let bytesPerRow = W * 4
    var rgba = [UInt8](repeating: 0, count: H * bytesPerRow)
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let bitmapInfo =
        CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

    rgba.withUnsafeMutableBytes { buf in
        guard let ctx = CGContext(
            data: buf.baseAddress, width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace, bitmapInfo: bitmapInfo
        ) else { return }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: W, height: H))
    }

    var floats = [Float](repeating: 0, count: H * W * 3)
    for i in 0..<(H * W) {
        floats[i * 3 + 0] = Float(rgba[i * 4 + 0]) / 255.0
        floats[i * 3 + 1] = Float(rgba[i * 4 + 1]) / 255.0
        floats[i * 3 + 2] = Float(rgba[i * 4 + 2]) / 255.0
    }
    return MLXArray(floats, [H, W, 3])
}
