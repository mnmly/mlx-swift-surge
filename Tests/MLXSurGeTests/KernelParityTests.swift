// Gather-vs-fused-kernel parity. The fused Metal kernel must reproduce the
// gather reference (the parity oracle) before it can be trusted in the head.
// Run with: xcodebuild -scheme mlx-swift-surge-Package -destination 'platform=macOS' test

import MLX
import MLXNN
import XCTest

@testable import MLXSurGe

final class KernelParityTests: XCTestCase {

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = abs(a.asType(.float32) - b.asType(.float32)).max()
        eval(d)
        return d.item(Float.self)
    }

    /// Raw windowed-attention op: gather vs fused on random q/k/v.
    func testFusedMatchesGatherRaw() {
        let cases: [(b: Int, h: Int, w: Int, heads: Int, hd: Int, k: Int)] = [
            (1, 12, 10, 1, 64, 9),
            (1, 16, 16, 4, 32, 9),
            (2, 11, 13, 2, 16, 9),
            (1, 32, 32, 16, 64, 9),  // stage-0 shape
        ]
        for c in cases {
            let q = MLXRandom.normal([c.b, c.h, c.w, c.heads, c.hd])
            let k = MLXRandom.normal([c.b, c.h, c.w, c.heads, c.hd])
            let v = MLXRandom.normal([c.b, c.h, c.w, c.heads, c.hd])
            let scale: Float = 1.0 / Float(c.hd).squareRoot()

            let ref = neighborhoodAttention(
                q: q, k: k, v: v, kernelH: c.k, kernelW: c.k,
                dilationH: 1, dilationW: 1, scale: scale)
            let fused = neighborhoodAttentionFused(
                q: q, k: k, v: v, kernelH: c.k, kernelW: c.k,
                dilationH: 1, dilationW: 1, scale: scale)

            XCTAssertEqual(fused.shape, ref.shape, "shape \(c)")
            let diff = maxAbsDiff(ref, fused)
            print("[kernel] case \(c): maxAbs=\(diff)")
            XCTAssertLessThan(diff, 1e-4, "fused vs gather diverged for \(c)")
        }
    }

    /// Same weights, through the module, with rope + qk-norm: toggling
    /// `useFusedKernel` must not change the output.
    func testFusedMatchesGatherThroughModule() {
        let dim = 128
        let heads = 2
        let attn = NeighborhoodAttention2d(
            embedDim: dim, numHeads: heads, kernelSize: (9, 9), dilation: (1, 1),
            qkvBias: true, qkNorm: true, useFusedKernel: false)
        let x = MLXRandom.normal([1, 14, 12, dim])
        let rope = ropeEmbedding(height: 14, width: 12, headDim: dim / heads, temperature: 2.8648)

        attn.useFusedKernel = false
        let ref = attn(x, ropeEmbed: rope)
        attn.useFusedKernel = true
        let fused = attn(x, ropeEmbed: rope)

        XCTAssertEqual(fused.shape, ref.shape)
        let diff = maxAbsDiff(ref, fused)
        print("[kernel] through-module: maxAbs=\(diff)")
        XCTAssertLessThan(diff, 1e-4)
    }
}
