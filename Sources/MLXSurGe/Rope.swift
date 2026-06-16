// Rotary position embedding for the NAD head.
//
// Ports timm's `RotaryEmbeddingCat(in_pixels=False)` + `apply_rot_embed_cat`
// (`timm/layers/pos_embed_sincos.py`). NAD constructs it with `feat_shape=None`
// (bands mode), `grid_indexing="ij"`, `grid_offset=0`, and `half=False`
// (interleaved rotation, since `RotaryEmbeddingCat` has no `rotate_half`
// attribute).

import Foundation
import MLX

/// Repeat-interleave the last axis by 2 (numpy `repeat`, torch
/// `repeat_interleave(2, -1)`). `[a, b]` -> `[a, a, b, b]`.
private func interleave2(_ x: MLXArray) -> MLXArray {
    let L = x.dim(-1)
    var bshape = x.shape
    bshape[bshape.count - 1] = L
    bshape.append(2)
    let e = x.expandedDimensions(axis: -1)            // (..., L, 1)
    let b = broadcast(e, to: bshape)                  // (..., L, 2)
    var outShape = x.shape
    outShape[outShape.count - 1] = L * 2
    return b.reshaped(outShape)                       // (..., 2L)
}

/// Build the concatenated sin/cos rotary embedding for a `height × width`
/// grid, matching `RotaryEmbeddingCat.get_embed([h, w]).reshape(h, w, 1, -1)`.
///
/// - `headDim`: per-head channel dimension (must be divisible by 4).
/// - `temperature`: inverse-frequency temperature.
/// - Returns: `(height, width, 1, 2 * headDim)` with `[..., :headDim]` = sin,
///   `[..., headDim:]` = cos (each interleave-doubled across frequency).
public func ropeEmbedding(
    height: Int, width: Int, headDim: Int, temperature: Float, dtype: DType = .float32
) -> MLXArray {
    let numBands = headDim / 4

    // freq_bands(num_bands, temperature, step=1): bands_k = temperature^(-k/num_bands)
    var bandsF = [Float](repeating: 0, count: numBands)
    for k in 0..<numBands {
        bandsF[k] = powf(temperature, -Float(k) / Float(numBands))
    }
    let bands = MLXArray(bandsF) // (numBands,)

    // grid via meshgrid(arange(h), arange(w), indexing="ij"), grid_offset=0.
    let rows = MLXArray(Array(0..<height).map { Float($0) }).reshaped([height, 1])
    let cols = MLXArray(Array(0..<width).map { Float($0) }).reshaped([1, width])
    let gridY = broadcast(rows, to: [height, width])
    let gridX = broadcast(cols, to: [height, width])
    let grid = stacked([gridY, gridX], axis: -1) // (h, w, 2)

    let pos = grid.expandedDimensions(axis: -1) * bands // (h, w, 2, numBands)
    let posSin = MLX.sin(pos)
    let posCos = MLX.cos(pos)

    // reshape (h, w, 2*numBands) then repeat_interleave(2, -1) -> (h, w, headDim)
    let sinFlat = posSin.reshaped([height, width, 2 * numBands])
    let cosFlat = posCos.reshaped([height, width, 2 * numBands])
    let sinRep = interleave2(sinFlat)
    let cosRep = interleave2(cosFlat)

    let emb = concatenated([sinRep, cosRep], axis: -1) // (h, w, 2*headDim)
    return emb.reshaped([height, width, 1, 2 * headDim]).asType(dtype)
}

/// Interleaved rotation `rot(x)`: `[x0, x1, x2, x3] -> [-x1, x0, -x3, x2]`.
private func rotateInterleaved(_ x: MLXArray) -> MLXArray {
    let hd = x.dim(-1)
    var pairShape = Array(x.shape.dropLast())
    pairShape.append(hd / 2)
    pairShape.append(2)
    let xv = x.reshaped(pairShape)            // (..., hd/2, 2)
    let xEven = xv[.ellipsis, 0]              // (..., hd/2)
    let xOdd = xv[.ellipsis, 1]               // (..., hd/2)
    let rotated = stacked([-xOdd, xEven], axis: -1) // (..., hd/2, 2)
    return rotated.reshaped(x.shape)
}

/// Apply rotary embedding to `x` (`half=False`): `x*cos + rot(x)*sin`.
///
/// - `x`: `(B, H, W, heads, headDim)`.
/// - `emb`: `(H, W, 1, 2*headDim)` from `ropeEmbedding`.
public func applyRotaryEmbedding(_ x: MLXArray, _ emb: MLXArray) -> MLXArray {
    let hd = x.dim(-1)
    let e = emb.asType(x.dtype)
    let sinE = e[.ellipsis, ..<hd]
    let cosE = e[.ellipsis, hd...]
    return x * cosE + rotateInterleaved(x) * sinE
}
