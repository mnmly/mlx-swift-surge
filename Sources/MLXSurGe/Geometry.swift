// Full geometry extraction from a SurGe inference: a colored point cloud, a
// connected surface mesh (UV-mapped), and a normal-color texture — enough to
// render the point cloud, a textured mesh, and a normal-shaded mesh.
//
// Library-side (no RealityKit). Everything a frontend needs is Sendable:
// positions are camera-space RUB (`x, -y, -z`, upright/facing the viewer);
// per-vertex/per-point COLOR is delivered via UVs + an H×W texture (photo for
// the cloud/textured mesh, colorize_normal for the normal mesh), so the frontend
// uses a plain textured `UnlitMaterial` with no custom shader.

import Foundation
import MLX

public struct SurGeGeometry: Sendable {
    // Point cloud (every valid pixel, strided to maxPoints).
    public let pointPositions: [Float]   // 3·pointCount (RUB)
    public let pointUVs: [Float]         // 2·pointCount, in [0,1]
    public let pointCount: Int

    // Connected surface mesh (strided grid, valid quads).
    public let meshPositions: [Float]    // 3·vertexCount (RUB)
    public let meshUVs: [Float]          // 2·vertexCount
    public let meshIndices: [UInt32]     // 3·faceCount
    public let vertexCount: Int
    public let faceCount: Int

    // Textures sampled by UV (row-major RGBA, texWidth × texHeight).
    public let texWidth: Int
    public let texHeight: Int
    public let photoRGBA: [UInt8]        // input image
    public let normalRGBA: [UInt8]       // colorize_normal(per-pixel normals)

    public let center: SIMD3<Float>
    public let radius: Float

    public static let empty = SurGeGeometry(
        pointPositions: [], pointUVs: [], pointCount: 0,
        meshPositions: [], meshUVs: [], meshIndices: [], vertexCount: 0, faceCount: 0,
        texWidth: 0, texHeight: 0, photoRGBA: [], normalRGBA: [],
        center: .zero, radius: 1)
}

extension SurGeSession {
    /// Run inference and build point-cloud + mesh + normal-color geometry.
    ///
    /// - `maxPoints`: cap on point-cloud points (strided).
    /// - `meshStride`: grid stride for the surface mesh (1 = full res).
    public func inferGeometry(
        _ image: MLXArray, maxPoints: Int = 250_000, meshStride: Int = 2, fovX: Float? = nil
    ) -> SurGeGeometry {
        let out = inferArrays(image, fovX: fovX)
        let pointsArr = out["points"]!.asType(.float32)   // (1, H, W, 3)
        let depthArr = out["depth"]!.asType(.float32)      // (1, H, W)
        let H = pointsArr.dim(pointsArr.ndim - 3)
        let W = pointsArr.dim(pointsArr.ndim - 2)

        let p: [Float] = pointsArr.asArray(Float.self)     // 3·H·W (RDF)
        let d: [Float] = depthArr.asArray(Float.self)      // H·W
        let colorImage = bilinearResize(image.asType(.float32), H, W)
        let c: [Float] = colorImage.asArray(Float.self)    // 3·H·W in [0,1]

        // Validity mask.
        var valid = [Bool](repeating: false, count: H * W)
        for i in 0..<(H * W) {
            let z = d[i]
            valid[i] = z.isFinite && z > 0
                && p[i * 3].isFinite && p[i * 3 + 1].isFinite && p[i * 3 + 2].isFinite
        }

        // Per-pixel normals (utils3d points_to_normals: average of the 4 valid
        // neighbor cross-products), then colorize_normal → RGBA.
        let normal = Self.pointsToNormals(p, valid: valid, H: H, W: W) // 3·H·W (RDF)
        var photoRGBA = [UInt8](repeating: 0, count: H * W * 4)
        var normalRGBA = [UInt8](repeating: 0, count: H * W * 4)
        @inline(__always) func byte(_ v: Float) -> UInt8 { UInt8(max(0, min(255, v * 255))) }
        for i in 0..<(H * W) {
            photoRGBA[i * 4 + 0] = byte(c[i * 3 + 0])
            photoRGBA[i * 4 + 1] = byte(c[i * 3 + 1])
            photoRGBA[i * 4 + 2] = byte(c[i * 3 + 2])
            photoRGBA[i * 4 + 3] = 255
            // colorize_normal: normal * [0.5, -0.5, -0.5] + 0.5
            let nx = normal[i * 3 + 0], ny = normal[i * 3 + 1], nz = normal[i * 3 + 2]
            normalRGBA[i * 4 + 0] = byte(nx * 0.5 + 0.5)
            normalRGBA[i * 4 + 1] = byte(ny * -0.5 + 0.5)
            normalRGBA[i * 4 + 2] = byte(nz * -0.5 + 0.5)
            normalRGBA[i * 4 + 3] = 255
        }

        // RealityKit samples textures with v=0 at the top, but pixel row 0 is
        // also the top — empirically the result was vertically flipped, so flip v.
        @inline(__always) func uv(_ row: Int, _ col: Int) -> (Float, Float) {
            ((Float(col) + 0.5) / Float(W), 1 - (Float(row) + 0.5) / Float(H))
        }
        @inline(__always) func rub(_ idx: Int) -> (Float, Float, Float) {
            (p[idx * 3 + 0], -p[idx * 3 + 1], -p[idx * 3 + 2])
        }

        var minX = Float.greatestFiniteMagnitude, minY = minX, minZ = minX
        var maxX = -Float.greatestFiniteMagnitude, maxY = maxX, maxZ = maxX
        @inline(__always) func expand(_ x: Float, _ y: Float, _ z: Float) {
            minX = min(minX, x); maxX = max(maxX, x)
            minY = min(minY, y); maxY = max(maxY, y)
            minZ = min(minZ, z); maxZ = max(maxZ, z)
        }

        // --- Point cloud (strided) ---
        let total = H * W
        let pStride = max(1, Int((Double(total) / Double(max(1, maxPoints))).rounded(.up)))
        var pointPositions: [Float] = []
        var pointUVs: [Float] = []
        var i = 0
        while i < total {
            if valid[i] {
                let (x, y, z) = rub(i)
                pointPositions.append(x); pointPositions.append(y); pointPositions.append(z)
                let (u, v) = uv(i / W, i % W)
                pointUVs.append(u); pointUVs.append(v)
                expand(x, y, z)
            }
            i += pStride
        }

        // --- Surface mesh (strided grid, valid quads) ---
        let s = max(1, meshStride)
        let gh = (H + s - 1) / s, gw = (W + s - 1) / s
        var vIndex = [Int32](repeating: -1, count: gh * gw)
        var meshPositions: [Float] = []
        var meshUVs: [Float] = []
        for gi in 0..<gh {
            let row = min(gi * s, H - 1)
            for gj in 0..<gw {
                let col = min(gj * s, W - 1)
                let idx = row * W + col
                guard valid[idx] else { continue }
                vIndex[gi * gw + gj] = Int32(meshPositions.count / 3)
                let (x, y, z) = rub(idx)
                meshPositions.append(x); meshPositions.append(y); meshPositions.append(z)
                let (u, v) = uv(row, col)
                meshUVs.append(u); meshUVs.append(v)
            }
        }
        var meshIndices: [UInt32] = []
        for gi in 0..<(gh - 1) {
            for gj in 0..<(gw - 1) {
                let a = vIndex[gi * gw + gj]
                let b = vIndex[gi * gw + gj + 1]
                let cIdx = vIndex[(gi + 1) * gw + gj]
                let e = vIndex[(gi + 1) * gw + gj + 1]
                guard a >= 0, b >= 0, cIdx >= 0, e >= 0 else { continue }
                // two triangles per quad (CCW)
                meshIndices.append(UInt32(a)); meshIndices.append(UInt32(cIdx)); meshIndices.append(UInt32(b))
                meshIndices.append(UInt32(b)); meshIndices.append(UInt32(cIdx)); meshIndices.append(UInt32(e))
            }
        }

        let pointCount = pointPositions.count / 3
        guard pointCount > 0 else { return .empty }
        let center = SIMD3<Float>((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2)
        let dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ
        let radius = max(1e-3, 0.5 * (dx * dx + dy * dy + dz * dz).squareRoot())

        return SurGeGeometry(
            pointPositions: pointPositions, pointUVs: pointUVs, pointCount: pointCount,
            meshPositions: meshPositions, meshUVs: meshUVs, meshIndices: meshIndices,
            vertexCount: meshPositions.count / 3, faceCount: meshIndices.count / 3,
            texWidth: W, texHeight: H, photoRGBA: photoRGBA, normalRGBA: normalRGBA,
            center: center, radius: radius)
    }

    /// Port of `utils3d.numpy.points_to_normals`: per pixel, average the unit
    /// cross-products of the (up,left),(left,down),(down,right),(right,up)
    /// neighbor pairs that are both valid, then normalize. Output (3·H·W, RDF).
    static func pointsToNormals(_ p: [Float], valid: [Bool], H: Int, W: Int) -> [Float] {
        var normal = [Float](repeating: 0, count: H * W * 3)

        @inline(__always) func pt(_ r: Int, _ c: Int) -> (Float, Float, Float) {
            let i = r * W + c
            return (p[i * 3], p[i * 3 + 1], p[i * 3 + 2])
        }
        @inline(__always) func ok(_ r: Int, _ c: Int) -> Bool {
            r >= 0 && r < H && c >= 0 && c < W && valid[r * W + c]
        }
        @inline(__always) func cross(_ a: (Float, Float, Float), _ b: (Float, Float, Float))
            -> (Float, Float, Float)
        {
            (a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0)
        }

        for r in 0..<H {
            for c in 0..<W {
                let i = r * W + c
                guard valid[i] else { continue }
                let o = pt(r, c)
                func diff(_ rr: Int, _ cc: Int) -> (Float, Float, Float)? {
                    guard ok(rr, cc) else { return nil }
                    let q = pt(rr, cc)
                    return (q.0 - o.0, q.1 - o.1, q.2 - o.2)
                }
                let up = diff(r - 1, c), left = diff(r, c - 1)
                let down = diff(r + 1, c), right = diff(r, c + 1)
                var sx: Float = 0, sy: Float = 0, sz: Float = 0
                var n = 0
                func accum(_ a: (Float, Float, Float)?, _ b: (Float, Float, Float)?) {
                    guard let a, let b else { return }
                    let cr = cross(a, b)
                    let len = (cr.0 * cr.0 + cr.1 * cr.1 + cr.2 * cr.2).squareRoot() + 1e-12
                    sx += cr.0 / len; sy += cr.1 / len; sz += cr.2 / len
                    n += 1
                }
                accum(up, left); accum(left, down); accum(down, right); accum(right, up)
                if n > 0 {
                    let len = (sx * sx + sy * sy + sz * sz).squareRoot() + 1e-12
                    normal[i * 3 + 0] = sx / len
                    normal[i * 3 + 1] = sy / len
                    normal[i * 3 + 2] = sz / len
                }
            }
        }
        return normal
    }
}
