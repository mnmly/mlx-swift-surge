// Point-cloud extraction from a SurGe inference.
//
// Library-side (no RealityKit — the library stays headless-usable). Produces a
// Sendable `SurGePointCloud` of camera-space positions + per-pixel RGB that a
// frontend turns into a RealityKit `LowLevelMesh`.

import Foundation
import MLX

/// A Sendable point cloud: flat `positions` (3·count, RUB camera frame so it
/// displays upright/facing the viewer) + `colors` (4·count RGBA) + a bounding
/// sphere for camera framing.
public struct SurGePointCloud: Sendable {
    public let positions: [Float]
    public let colors: [UInt8]
    public let count: Int
    public let center: SIMD3<Float>
    public let radius: Float

    public static let empty = SurGePointCloud(
        positions: [], colors: [], count: 0, center: .zero, radius: 1)
}

extension SurGeSession {
    /// Run inference and project the point map to a colored point cloud.
    ///
    /// - Colors come from `image` (same H×W as the output point map).
    /// - Invalid pixels (non-finite or non-positive depth) are dropped.
    /// - Points are converted RDF → RUB (`x, -y, -z`) so the cloud sits upright
    ///   and faces the camera in RealityKit's Y-up / -Z-forward space.
    /// - Downsampled by striding when the valid count exceeds `maxPoints`.
    public func inferPointCloud(
        _ image: MLXArray, maxPoints: Int = 300_000, fovX: Float? = nil
    ) -> SurGePointCloud {
        let out = inferArrays(image, fovX: fovX)
        let pointsArr = out["points"]!.asType(.float32)   // (1, H, W, 3)
        let depthArr = out["depth"]!.asType(.float32)      // (1, H, W)
        let H = pointsArr.dim(pointsArr.ndim - 3)
        let W = pointsArr.dim(pointsArr.ndim - 2)

        let p: [Float] = pointsArr.asArray(Float.self)     // 3·H·W
        let d: [Float] = depthArr.asArray(Float.self)      // H·W
        // Colors from the (already output-sized) input image.
        let colorImage = bilinearResize(image.asType(.float32), H, W)
        let c: [Float] = colorImage.asArray(Float.self)    // 3·H·W in [0, 1]

        let total = H * W
        let stride = max(1, Int((Double(total) / Double(max(1, maxPoints))).rounded(.up)))

        var positions: [Float] = []
        var colors: [UInt8] = []
        positions.reserveCapacity((total / stride) * 3)
        colors.reserveCapacity((total / stride) * 4)

        var minX = Float.greatestFiniteMagnitude, minY = minX, minZ = minX
        var maxX = -Float.greatestFiniteMagnitude, maxY = maxX, maxZ = maxX

        @inline(__always) func byte(_ v: Float) -> UInt8 { UInt8(max(0, min(255, v * 255))) }

        var i = 0
        while i < total {
            let z = d[i]
            let x = p[i * 3 + 0], y = p[i * 3 + 1], zz = p[i * 3 + 2]
            if z.isFinite && z > 0 && x.isFinite && y.isFinite && zz.isFinite {
                let px = x, py = -y, pz = -zz   // RDF → RUB
                positions.append(px); positions.append(py); positions.append(pz)
                colors.append(byte(c[i * 3 + 0]))
                colors.append(byte(c[i * 3 + 1]))
                colors.append(byte(c[i * 3 + 2]))
                colors.append(255)
                minX = min(minX, px); maxX = max(maxX, px)
                minY = min(minY, py); maxY = max(maxY, py)
                minZ = min(minZ, pz); maxZ = max(maxZ, pz)
            }
            i += stride
        }

        let count = colors.count / 4
        guard count > 0 else { return .empty }
        let center = SIMD3<Float>((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2)
        let dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ
        let radius = max(1e-3, 0.5 * (dx * dx + dy * dy + dz * dz).squareRoot())
        return SurGePointCloud(
            positions: positions, colors: colors, count: count, center: center, radius: radius)
    }
}
