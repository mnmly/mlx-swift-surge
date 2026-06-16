// Geometry utilities for SurGe post-processing.
//
// SurGe's geometry code is adapted from MoGe; these helpers mirror
// `surge/utils/geometry_torch.py` + `utils3d`. The focal/shift recovery that
// Python delegates to `scipy.optimize.least_squares` is ported as a small
// Levenberg-Marquardt solver in SurGeInference.swift.

import Foundation
import MLX

/// Generate normalized UV coordinates for a view plane, matching
/// `normalized_view_plane_uv`. Returns `(H, W, 2)` with
/// `meshgrid(u, v, indexing="xy")` semantics.
///
/// Note: SurGe uses the endpoint convention
/// `linspace(-span*(N-1)/N, span*(N-1)/N, N)`.
public func normalizedViewPlaneUV(
    width: Int,
    height: Int,
    aspectRatio: Float? = nil,
    dtype: DType = .float32
) -> MLXArray {
    let ar = aspectRatio ?? Float(width) / Float(height)
    let spanX = ar / (1.0 + ar * ar).squareRoot()
    let spanY = 1.0 / (1.0 + ar * ar).squareRoot()

    let uEnd = spanX * Float(width - 1) / Float(width)
    let vEnd = spanY * Float(height - 1) / Float(height)

    let u = MLX.linspace(-uEnd, uEnd, count: width).asType(dtype)   // (W,)
    let v = MLX.linspace(-vEnd, vEnd, count: height).asType(dtype)  // (H,)

    // meshgrid(u, v, indexing="xy") -> u broadcast over rows, v over cols.
    let gridU = broadcast(u.expandedDimensions(axis: 0), to: [height, width]) // (H, W)
    let gridV = broadcast(v.expandedDimensions(axis: 1), to: [height, width]) // (H, W)

    return stacked([gridU, gridV], axis: -1) // (H, W, 2)
}

/// Build a `(B, 3, 3)` intrinsics matrix from per-sample focal/principal arrays.
public func intrinsicsFromFocalCenter(
    fx: [Float], fy: [Float], cx: [Float], cy: [Float]
) -> MLXArray {
    let B = fx.count
    precondition(fy.count == B && cx.count == B && cy.count == B)
    var rows: [[Float]] = []
    for i in 0..<B {
        rows.append([fx[i], 0, cx[i],
                     0, fy[i], cy[i],
                     0,     0,    1])
    }
    let flat = rows.flatMap { $0 }
    return MLXArray(flat).reshaped([B, 3, 3])
}

/// Unproject a depth map to a 3D point cloud using normalized intrinsics
/// (cx/cy in [0, 1]). Mirrors `utils3d.torch.depth_to_points`.
public func depthMapToPointMap(
    depth: MLXArray, intrinsics: MLXArray, height: Int, width: Int
) -> MLXArray {
    let isBatched = depth.ndim == 3
    let depthBatched = isBatched ? depth : depth.expandedDimensions(axis: 0)
    let intrinsicsBatched =
        intrinsics.ndim == 3 ? intrinsics : intrinsics.expandedDimensions(axis: 0)

    let B = depthBatched.dim(0)
    let H = height
    let W = width

    // Pixel coordinates in [0, 1] with half-pixel offsets.
    let u = MLX.linspace(0.5 / Float(W), 1 - 0.5 / Float(W), count: W) // (W,)
    let v = MLX.linspace(0.5 / Float(H), 1 - 0.5 / Float(H), count: H) // (H,)

    let gridU = broadcast(u.expandedDimensions(axis: 0), to: [H, W])
    let gridV = broadcast(v.expandedDimensions(axis: 1), to: [H, W])

    let gridU4 = gridU.reshaped([1, H, W, 1])
    let gridV4 = gridV.reshaped([1, H, W, 1])

    let fx = intrinsicsBatched[0..., 0, 0].reshaped([B, 1, 1, 1])
    let fy = intrinsicsBatched[0..., 1, 1].reshaped([B, 1, 1, 1])
    let cx = intrinsicsBatched[0..., 0, 2].reshaped([B, 1, 1, 1])
    let cy = intrinsicsBatched[0..., 1, 2].reshaped([B, 1, 1, 1])

    let depth4 = depthBatched.expandedDimensions(axis: -1)

    let x = (gridU4 - cx) / fx * depth4
    let y = (gridV4 - cy) / fy * depth4
    let z = depth4

    var result = concatenated([x, y, z], axis: -1) // (B, H, W, 3)
    if !isBatched {
        result = result[0]
    }
    return result
}
