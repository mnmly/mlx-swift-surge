// User-friendly inference for SurGe. Mirrors `SurGe.infer` in
// `surge/models/surge.py`: forward pass -> recover focal/shift from the affine
// point map -> camera intrinsics + depth -> optional reprojection.
//
// `recover_focal_shift` is the one host-side step: Python solves
// `min_shift |focal(shift)·xy/(z+shift) - uv|` with `scipy.optimize.least_squares`
// (focal closed-form per shift). The joint (focal, shift) Levenberg-Marquardt
// here reaches the same optimum (∂/∂focal = 0 gives the closed form). Points are
// downsampled to 64×64 with nearest sampling, matching the reference.

import Foundation
import MLX

public enum SurGeTokens {
    case min
    case max
    case count(Int)

    func resolve(_ range: [Int]) -> Int {
        switch self {
        case .min: return range[0]
        case .max: return range[1]
        case .count(let n): return n
        }
    }
}

extension SurGeModel {

    /// Full inference with focal/shift recovery.
    ///
    /// - image: NHWC `(H, W, 3)` or `(B, H, W, 3)` float in [0, 1].
    /// - Returns: `points` `(B?, H, W, 3)`, `depth` `(B?, H, W)`,
    ///   `intrinsics` `(B?, 3, 3)`.
    public func infer(
        image: MLXArray,
        tokens: SurGeTokens = .max,
        resizeOutput: Bool = true,
        forceProjection: Bool = true,
        fovX: Float? = nil
    ) -> [String: MLXArray] {
        let omitBatch = image.ndim == 3
        let batched = omitBatch ? image.expandedDimensions(axis: 0) : image

        let B = batched.dim(0)
        let imgH = batched.dim(1)
        let imgW = batched.dim(2)
        let aspectRatio = Float(imgW) / Float(imgH)

        let numTokens = tokens.resolve(numTokensRange)

        // Forward pass -> affine point map (B, H, W, 3).
        let pointsOut = self(batched, numTokens: numTokens, resizeOutput: resizeOutput)
        let outH = pointsOut.dim(1)
        let outW = pointsOut.dim(2)
        MLX.eval(pointsOut)
        let pointsFlat: [Float] = pointsOut.asType(.float32).asArray(Float.self)

        var focals = [Float](repeating: 1, count: B)
        var shifts = [Float](repeating: 0, count: B)
        let stride = outH * outW * 3

        for b in 0..<B {
            let ptsSlice = Array(pointsFlat[(b * stride)..<((b + 1) * stride)])
            let (uv, xyz) = Self.gatherUVXYZ(
                points: ptsSlice, H: outH, W: outW, aspectRatio: aspectRatio)

            if let fx = fovX {
                let knownFocal =
                    aspectRatio / (1 + aspectRatio * aspectRatio).squareRoot() / tanf(fx / 2)
                shifts[b] = Self.solveOptimalShift(uv: uv, xyz: xyz, focal: knownFocal)
                focals[b] = knownFocal
            } else {
                let (f, s) = Self.solveOptimalFocalShift(uv: uv, xyz: xyz)
                focals[b] = f
                shifts[b] = s
            }
        }

        // Depth = points_z + shift.
        var depthFlat = [Float](repeating: 0, count: B * outH * outW)
        for b in 0..<B {
            let s = shifts[b]
            for i in 0..<(outH * outW) {
                depthFlat[b * outH * outW + i] = pointsFlat[b * stride + i * 3 + 2] + s
            }
        }

        // Intrinsics.
        var fxArr = [Float](repeating: 0, count: B)
        var fyArr = [Float](repeating: 0, count: B)
        let sqrtTerm = (1 + aspectRatio * aspectRatio).squareRoot()
        for b in 0..<B {
            fxArr[b] = focals[b] / 2 * sqrtTerm / aspectRatio
            fyArr[b] = focals[b] / 2 * sqrtTerm
        }
        let intrinsics = intrinsicsFromFocalCenter(
            fx: fxArr, fy: fyArr,
            cx: [Float](repeating: 0.5, count: B),
            cy: [Float](repeating: 0.5, count: B))

        let depthMLX = MLXArray(depthFlat, [B, outH, outW])

        var pointsMLX: MLXArray
        if forceProjection {
            pointsMLX = depthMapToPointMap(
                depth: depthMLX, intrinsics: intrinsics, height: outH, width: outW)
        } else {
            // Replace z with shifted depth.
            pointsMLX = MLXArray(pointsFlat, [B, outH, outW, 3])
            let xy = pointsMLX[.ellipsis, ..<2]
            let z = depthMLX.expandedDimensions(axis: -1)
            pointsMLX = concatenated([xy, z], axis: -1)
        }

        var result: [String: MLXArray] = [
            "points": pointsMLX,
            "depth": depthMLX,
            "intrinsics": intrinsics,
        ]
        if omitBatch {
            for (k, v) in result { result[k] = v[0] }
        }
        return result
    }

    // MARK: - Focal/shift recovery

    /// Downsample a single point map to 64×64 (nearest) and pair each sample
    /// with its normalized-view-plane UV. No mask (SurGe.infer uses none).
    private static func gatherUVXYZ(
        points: [Float], H: Int, W: Int, aspectRatio: Float, downsample: Int = 64
    ) -> (uv: [Float], xyz: [Float]) {
        let spanX = aspectRatio / (1 + aspectRatio * aspectRatio).squareRoot()
        let spanY = 1.0 / (1 + aspectRatio * aspectRatio).squareRoot()
        let uEnd = spanX * Float(W - 1) / Float(W)
        let vEnd = spanY * Float(H - 1) / Float(H)

        @inline(__always) func uAt(_ x: Int) -> Float {
            W <= 1 ? -uEnd : -uEnd + (2 * uEnd) * Float(x) / Float(W - 1)
        }
        @inline(__always) func vAt(_ y: Int) -> Float {
            H <= 1 ? -vEnd : -vEnd + (2 * vEnd) * Float(y) / Float(H - 1)
        }

        let n = downsample * downsample
        var uv = [Float](repeating: 0, count: n * 2)
        var xyz = [Float](repeating: 0, count: n * 3)

        var k = 0
        for oy in 0..<downsample {
            let sy = min((oy * H) / downsample, H - 1) // nearest
            let vv = vAt(sy)
            for ox in 0..<downsample {
                let sx = min((ox * W) / downsample, W - 1)
                let idx = sy * W + sx
                uv[k * 2 + 0] = uAt(sx)
                uv[k * 2 + 1] = vv
                xyz[k * 3 + 0] = points[idx * 3 + 0]
                xyz[k * 3 + 1] = points[idx * 3 + 1]
                xyz[k * 3 + 2] = points[idx * 3 + 2]
                k += 1
            }
        }
        return (uv, xyz)
    }

    /// Joint `(focal, shift)` Levenberg-Marquardt for
    /// `min |focal·xy/(z+shift) - uv|`. Same optimum as the reference's
    /// shift-only solve with closed-form focal.
    fileprivate static func solveOptimalFocalShift(uv: [Float], xyz: [Float]) -> (Float, Float) {
        let n = uv.count / 2
        if n < 2 { return (1.0, 0.0) }

        var focal: Float = 1.0
        var shift: Float = 0.0
        var lambda: Float = 1e-3

        func cost(_ f: Float, _ s: Float) -> Float {
            var sum: Float = 0
            for i in 0..<n {
                let x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2]
                let denom = z + s
                let ru = f * x / denom - uv[i * 2 + 0]
                let rv = f * y / denom - uv[i * 2 + 1]
                sum += ru * ru + rv * rv
            }
            return sum
        }

        var prevCost = cost(focal, shift)
        for _ in 0..<100 {
            var a00: Float = 0, a01: Float = 0, a11: Float = 0, b0: Float = 0, b1: Float = 0
            for i in 0..<n {
                let x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2]
                let invD = 1 / (z + shift)
                let ru = focal * x * invD - uv[i * 2 + 0]
                let rv = focal * y * invD - uv[i * 2 + 1]
                let dRuDf = x * invD, dRuDs = -focal * x * invD * invD
                let dRvDf = y * invD, dRvDs = -focal * y * invD * invD
                a00 += dRuDf * dRuDf + dRvDf * dRvDf
                a01 += dRuDf * dRuDs + dRvDf * dRvDs
                a11 += dRuDs * dRuDs + dRvDs * dRvDs
                b0 += dRuDf * ru + dRvDf * rv
                b1 += dRuDs * ru + dRvDs * rv
            }
            let m00 = a00 * (1 + lambda)
            let m11 = a11 * (1 + lambda)
            let det = m00 * m11 - a01 * a01
            if abs(det) < 1e-20 { break }
            let df = (-b0 * m11 + b1 * a01) / det
            let ds = (m00 * -b1 + a01 * b0) / det
            let newFocal = focal + df
            let newShift = shift + ds
            let newCost = cost(newFocal, newShift)
            if newCost < prevCost {
                focal = newFocal
                shift = newShift
                lambda = max(lambda / 3, 1e-10)
                if abs(prevCost - newCost) < 1e-9 * max(prevCost, 1e-10) { break }
                prevCost = newCost
            } else {
                lambda = min(lambda * 5, 1e10)
            }
        }
        return (focal, shift)
    }

    /// Shift-only solve for a known `focal`.
    fileprivate static func solveOptimalShift(uv: [Float], xyz: [Float], focal: Float) -> Float {
        let n = uv.count / 2
        if n < 2 { return 0 }
        var shift: Float = 0
        var lambda: Float = 1e-3

        func cost(_ s: Float) -> Float {
            var sum: Float = 0
            for i in 0..<n {
                let x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2]
                let denom = z + s
                let ru = focal * x / denom - uv[i * 2 + 0]
                let rv = focal * y / denom - uv[i * 2 + 1]
                sum += ru * ru + rv * rv
            }
            return sum
        }

        var prevCost = cost(shift)
        for _ in 0..<100 {
            var a: Float = 0, b: Float = 0
            for i in 0..<n {
                let x = xyz[i * 3 + 0], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2]
                let invD = 1 / (z + shift)
                let ru = focal * x * invD - uv[i * 2 + 0]
                let rv = focal * y * invD - uv[i * 2 + 1]
                let dRuDs = -focal * x * invD * invD
                let dRvDs = -focal * y * invD * invD
                a += dRuDs * dRuDs + dRvDs * dRvDs
                b += dRuDs * ru + dRvDs * rv
            }
            let m = a * (1 + lambda)
            if m < 1e-20 { break }
            let ds = -b / m
            let newShift = shift + ds
            let newCost = cost(newShift)
            if newCost < prevCost {
                shift = newShift
                lambda = max(lambda / 3, 1e-10)
                if abs(prevCost - newCost) < 1e-9 * max(prevCost, 1e-10) { break }
                prevCost = newCost
            } else {
                lambda = min(lambda * 5, 1e10)
            }
        }
        return shift
    }
}
