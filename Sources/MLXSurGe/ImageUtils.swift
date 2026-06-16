// Image processing utilities for the SurGe Swift port.
//
// `bilinearResize` / `padReplicate` mirror the MoGe port. `bicubicResize`
// reproduces PyTorch's `F.interpolate(mode="bicubic", align_corners=False,
// antialias=False)` exactly (separable cubic-convolution, a = -0.75, half-pixel
// source mapping using the *passed* scale factor). DINOv2 uses bicubic for
// positional-embedding interpolation, which is parity-critical for the encoder.

import MLX
import MLXNN

/// Replicate (edge) padding for NHWC tensors.
public func padReplicate(_ x: MLXArray, _ pad: Int) -> MLXArray {
    if pad == 0 { return x }
    // Pad only spatial axes (1, 2); leave batch and channel unchanged.
    let widths: [IntOrPair] = [
        IntOrPair(0), IntOrPair(pad), IntOrPair(pad), IntOrPair(0),
    ]
    return padded(x, widths: widths, mode: .edge)
}

/// Bilinear interpolation resize for NHWC tensors. Matches PyTorch's
/// `F.interpolate(..., mode='bilinear', align_corners=False)`.
public func bilinearResize(_ x: MLXArray, _ targetH: Int, _ targetW: Int) -> MLXArray {
    let H = x.dim(1)
    let W = x.dim(2)
    if H == targetH && W == targetW { return x }

    // `nextUp` nudges the scale past the truncation boundary so a Float
    // division landing a hair below the exact ratio doesn't drop a row/col.
    let sh = (Float(targetH) / Float(H)).nextUp
    let sw = (Float(targetW) / Float(W)).nextUp
    let up = Upsample(scaleFactor: .array([sh, sw]), mode: .linear(alignCorners: false))
    var out = up(x)
    let outH = out.dim(1)
    let outW = out.dim(2)
    if outH != targetH || outW != targetW {
        out = out[0..., 0..<targetH, 0..<targetW, 0...]
    }
    return out
}

// MARK: - Bicubic (exact PyTorch parity)

private let cubicA: Float = -0.75

@inline(__always)
private func cubicConv1(_ x: Float) -> Float {
    // |x| in [0, 1]
    ((cubicA + 2) * x - (cubicA + 3)) * x * x + 1
}

@inline(__always)
private func cubicConv2(_ x: Float) -> Float {
    // |x| in [1, 2]
    ((cubicA * x - 5 * cubicA) * x + 8 * cubicA) * x - 4 * cubicA
}

/// Cubic-convolution interpolation along a single NHWC spatial axis.
///
/// - `axis`: 1 (height) or 2 (width).
/// - `outSize`: target length along `axis`.
/// - `scale`: the *scale factor* passed to PyTorch's `interpolate`
///   (output coordinate `o` maps to source `(o + 0.5)/scale - 0.5`).
private func cubicInterpolate1D(
    _ x: MLXArray, axis: Int, outSize: Int, scale: Float
) -> MLXArray {
    let inSize = x.dim(axis)
    if inSize == outSize && abs(scale - 1) < 1e-12 { return x }

    let step = 1.0 / scale
    // 4 taps: offsets -1, 0, +1, +2 relative to floor(src).
    var idxTaps: [[Int32]] = Array(repeating: [Int32](repeating: 0, count: outSize), count: 4)
    var wTaps: [[Float]] = Array(repeating: [Float](repeating: 0, count: outSize), count: 4)

    for o in 0..<outSize {
        let src = (Float(o) + 0.5) * step - 0.5
        let x0 = src.rounded(.down)
        let t = src - x0
        let base = Int(x0)
        // PyTorch get_cubic_upsample_coefficients ordering.
        let w0 = cubicConv2(t + 1)
        let w1 = cubicConv1(t)
        let w2 = cubicConv1(1 - t)
        let w3 = cubicConv2((1 - t) + 1)
        let ws = [w0, w1, w2, w3]
        let offs = [base - 1, base, base + 1, base + 2]
        for k in 0..<4 {
            let clamped = min(max(offs[k], 0), inSize - 1)
            idxTaps[k][o] = Int32(clamped)
            wTaps[k][o] = ws[k]
        }
    }

    // Weight broadcast shape: 1 everywhere except `outSize` on `axis`.
    var wShape = [Int](repeating: 1, count: x.ndim)
    wShape[axis] = outSize

    var out: MLXArray? = nil
    for k in 0..<4 {
        let gatherIdx = MLXArray(idxTaps[k])
        let sampled = take(x, gatherIdx, axis: axis)
        let w = MLXArray(wTaps[k]).reshaped(wShape).asType(x.dtype)
        let term = sampled * w
        out = (out == nil) ? term : out! + term
    }
    return out!
}

/// Bicubic resize of an NHWC tensor matching
/// `F.interpolate(x, scale_factor=(scaleH, scaleW), mode="bicubic",
/// align_corners=False, antialias=False)`.
///
/// Output size is `floor(inSize * scale)` per axis, exactly as PyTorch
/// computes it when `recompute_scale_factor` is false.
public func bicubicResize(
    _ x: MLXArray, scaleH: Float, scaleW: Float
) -> MLXArray {
    let inH = x.dim(1)
    let inW = x.dim(2)
    let outH = Int((Float(inH) * scaleH))
    let outW = Int((Float(inW) * scaleW))
    var out = cubicInterpolate1D(x, axis: 1, outSize: outH, scale: scaleH)
    out = cubicInterpolate1D(out, axis: 2, outSize: outW, scale: scaleW)
    return out
}
