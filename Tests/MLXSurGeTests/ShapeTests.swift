// Weight-free shape / smoke tests for the SurGe building blocks.
// Run with: xcodebuild -scheme mlx-swift-surge -destination 'platform=macOS' test

import MLX
import MLXNN
import XCTest

@testable import MLXSurGe

final class ShapeTests: XCTestCase {

    func testBicubicResizeShapeAndIdentity() {
        // Identity when scale = 1.
        let x = MLXRandom.uniform(low: 0, high: 1, [1, 8, 8, 4])
        let same = bicubicResize(x, scaleH: 1.0, scaleW: 1.0)
        eval(same)
        XCTAssertEqual(same.shape, [1, 8, 8, 4])

        // Downsample 37 -> 32 like the DINOv2 pos-embed interpolation.
        let pe = MLXRandom.uniform(low: 0, high: 1, [1, 37, 37, 16])
        let scale = (Float(32) + 0.1) / Float(37)
        let out = bicubicResize(pe, scaleH: scale, scaleW: scale)
        eval(out)
        XCTAssertEqual(out.shape, [1, 32, 32, 16])
    }

    func testRopeEmbeddingShape() {
        let emb = ropeEmbedding(height: 5, width: 7, headDim: 64, temperature: 2.8648)
        eval(emb)
        XCTAssertEqual(emb.shape, [5, 7, 1, 128])
    }

    func testNeighborIndicesInnerWindow() {
        // Inner-neighborhood rule: every query attends to exactly `kernel` keys,
        // sliding inward at the borders.
        let idx = neighborIndices(length: 6, kernel: 3, dilation: 1)
        XCTAssertEqual(idx.count, 6)
        XCTAssertEqual(idx[0], [0, 1, 2])   // left border slides right
        XCTAssertEqual(idx[2], [1, 2, 3])   // centered
        XCTAssertEqual(idx[5], [3, 4, 5])   // right border slides left
    }

    func testNeighborhoodAttentionShape() {
        let dim = 64
        let attn = NeighborhoodAttention2d(
            embedDim: dim, numHeads: 1, kernelSize: (9, 9), dilation: (1, 1), qkNorm: true)
        let x = MLXRandom.uniform(low: 0, high: 1, [1, 12, 10, dim])
        let rope = ropeEmbedding(height: 12, width: 10, headDim: dim, temperature: 2.8648)
        let out = attn(x, ropeEmbed: rope)
        eval(out)
        XCTAssertEqual(out.shape, [1, 12, 10, dim])
    }

    func testNADBlockResidualShape() {
        let dim = 128
        let block = NADBlock(
            dim: dim, numHeads: 2, mlpRatio: 4.0, qkNorm: true,
            kernelSize: (9, 9), dilation: (1, 1))
        let x = MLXRandom.uniform(low: 0, high: 1, [1, 8, 8, dim])
        let rope = ropeEmbedding(height: 8, width: 8, headDim: 64, temperature: 2.8648)
        let out = block(x, ropeEmbed: rope)
        eval(out)
        XCTAssertEqual(out.shape, [1, 8, 8, dim])
    }

    func testUpsampleBlockDoublesResolution() {
        let block = UpsampleBlock(inChannels: 32, outChannels: 16, scaleFactor: 2, refineKernelSize: 3)
        let x = MLXRandom.uniform(low: 0, high: 1, [1, 8, 8, 32])
        let out = block(x)
        eval(out)
        XCTAssertEqual(out.shape, [1, 16, 16, 16])
    }

    func testNADHeadEndToEndShape() {
        // Tiny config that mirrors the real stage topology (5 stages, 2x upsample).
        let head = NAD(
            dimIn: [10, 2, 2, 2, 2],
            dimOut: [nil, nil, nil, nil, 3],
            embedDim: [64, 64, 64, 64, 64],
            depth: [1, 1, 1, 1, 1],
            headDim: 64, mlpRatio: 4.0, kernelSize: 9, dilation: 1,
            qkNorm: true, ropeTemperatureScale: 1.0 / Float.pi,
            upsampleRefineKernelSize: 3)
        // level 0: encoder feature + UV (channels 10); levels 1..4: UV (2).
        let f0 = MLXRandom.uniform(low: 0, high: 1, [1, 4, 4, 10])
        let levels: [MLXArray?] = [
            f0,
            MLXRandom.uniform(low: 0, high: 1, [1, 8, 8, 2]),
            MLXRandom.uniform(low: 0, high: 1, [1, 16, 16, 2]),
            MLXRandom.uniform(low: 0, high: 1, [1, 32, 32, 2]),
            MLXRandom.uniform(low: 0, high: 1, [1, 64, 64, 2]),
        ]
        let out = head(levels)
        eval(out)
        XCTAssertEqual(out.shape, [1, 64, 64, 3])
    }
}
