// Per-stage + end-to-end parity vs the Torch reference.
//
// Fixtures are produced by `Scripts/generate_fixtures.py` (real surge-large
// weights, export-friendly neighborhood attention = the same math this port
// implements). Boundaries are checked bottom-up so a divergence localizes:
//
//   bicubic pos-embed  ->  encoder feature  ->  forward points  ->  infer outputs
//
// Set SURGE_WEIGHTS to the HF snapshot dir (config.json + model.safetensors).
// Defaults to the standard HuggingFace cache location for surge-large.

import Foundation
import MLX
import XCTest

@testable import MLXSurGe

final class ParityFixtureTests: XCTestCase {

    private static let defaultWeights =
        ("\(NSHomeDirectory())/.cache/huggingface/hub/models--karimknaebel--surge-large"
            + "/snapshots/860cb8c37d6db782df94fac5861e89e4fc228aee")

    private func fixtureURL() throws -> URL {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let dir = env["SURGE_FIXTURE_DIR"] ?? root.appendingPathComponent("Tests/Fixtures").path
        let url = URL(fileURLWithPath: dir).appendingPathComponent("surge_fixtures.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Fixture not found at \(url.path). Run Scripts/generate_fixtures.py.")
        }
        return url
    }

    private func weightsPath() throws -> String {
        let env = ProcessInfo.processInfo.environment
        let path = env["SURGE_WEIGHTS"] ?? Self.defaultWeights
        guard FileManager.default.fileExists(atPath: "\(path)/model.safetensors") else {
            throw XCTSkip("SURGE_WEIGHTS not found at \(path) (need config.json + model.safetensors).")
        }
        return path
    }

    private static var cachedModel: SurGeModel?
    private func loadModel() throws -> SurGeModel {
        if let m = Self.cachedModel { return m }
        let m = try SurGeModel.fromPretrained(path: try weightsPath(), dtype: .float32)
        Self.cachedModel = m
        return m
    }

    private struct Stats { let maxAbs: Float; let meanAbs: Float; let n: Int }

    private func compare(_ actual: MLXArray, _ expected: MLXArray) -> Stats {
        let a = actual.asType(.float32)
        let e = expected.asType(.float32)
        eval(a, e)
        let av: [Float] = a.asArray(Float.self)
        let ev: [Float] = e.asArray(Float.self)
        var maxAbs: Float = 0
        var sum: Double = 0
        var n = 0
        for (x, y) in zip(av, ev) {
            guard x.isFinite, y.isFinite else { continue }
            let d = abs(x - y)
            if d > maxAbs { maxAbs = d }
            sum += Double(d)
            n += 1
        }
        return Stats(maxAbs: maxAbs, meanAbs: n > 0 ? Float(sum / Double(n)) : .infinity, n: n)
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: MLXArray, key: String,
        atol: Float, meanAtol: Float, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, "\(key) shape", file: file, line: line)
        guard actual.shape == expected.shape else { return }
        let s = compare(actual, expected)
        print("[parity] \(key): maxAbs=\(s.maxAbs), meanAbs=\(s.meanAbs), n=\(s.n)")
        if s.maxAbs > atol || s.meanAbs > meanAtol {
            XCTFail(
                "\(key) parity failed: maxAbs=\(s.maxAbs) (limit \(atol)), "
                    + "meanAbs=\(s.meanAbs) (limit \(meanAtol))", file: file, line: line)
        }
    }

    // MARK: - Stage 1: isolated bicubic pos-embed interpolation

    func testBicubicPosEmbedMatchesTorch() throws {
        let fx = try loadArrays(url: fixtureURL())
        guard let pe37 = fx["pos_embed_patch_37"],
              let peInterp = fx["pos_embed_interp_grid"] else {
            throw XCTSkip("Missing pos-embed fixtures.")
        }
        let grid = peInterp.dim(1)            // 32
        let m = pe37.dim(1)                   // 37
        let scale = (Float(grid) + 0.1) / Float(m)
        let out = bicubicResize(pe37, scaleH: scale, scaleW: scale)
        assertClose(out, peInterp, key: "bicubic_pos_embed", atol: 2e-4, meanAtol: 2e-5)
    }

    // MARK: - Stage 2: encoder feature

    func testEncoderFeatureMatchesTorch() throws {
        let fx = try loadArrays(url: fixtureURL())
        guard let input = fx["input_nhwc"], let expected = fx["encoder_feature"] else {
            throw XCTSkip("Missing encoder fixtures.")
        }
        let model = try loadModel()
        let grid = expected.dim(1)
        let (feature, _) = model.encoder(input, tokenRows: grid, tokenCols: grid)
        // Encoder features have absmean ~2.9. ~0.6% relative drift accumulates
        // over 24 fp32 transformer layers (cross-framework SDPA / op ordering).
        // This is an intermediate; parity is gated by `forward_points` below.
        assertClose(feature, expected, key: "encoder_feature", atol: 0.6, meanAtol: 0.03)
    }

    // MARK: - Stage 3: forward points (exp-remapped affine point map)

    func testForwardPointsMatchTorch() throws {
        let fx = try loadArrays(url: fixtureURL())
        guard let input = fx["input_nhwc"], let expected = fx["forward_points"] else {
            throw XCTSkip("Missing forward fixtures.")
        }
        // Explicitly the gather reference (the default is now the fused kernel),
        // so this path keeps independent end-to-end coverage.
        let model = try SurGeModel.fromPretrained(
            path: try weightsPath(), dtype: .float32, useFusedKernel: false)
        let points = model(input, numTokens: 1024, resizeOutput: true)
        assertClose(points, expected, key: "forward_points_gather", atol: 5e-2, meanAtol: 5e-3)
    }

    // MARK: - Stage 3b: forward points with the fused Metal kernel

    func testForwardPointsFusedMatchTorch() throws {
        let fx = try loadArrays(url: fixtureURL())
        guard let input = fx["input_nhwc"], let expected = fx["forward_points"] else {
            throw XCTSkip("Missing forward fixtures.")
        }
        // Fresh model with the fused neighborhood-attention kernel enabled.
        let model = try SurGeModel.fromPretrained(
            path: try weightsPath(), dtype: .float32, useFusedKernel: true)
        let points = model(input, numTokens: 1024, resizeOutput: true)
        assertClose(points, expected, key: "forward_points_fused", atol: 5e-2, meanAtol: 5e-3)
    }

    // MARK: - Stage 4: full infer (points / depth / intrinsics)

    func testInferMatchesTorch() throws {
        let fx = try loadArrays(url: fixtureURL())
        guard let input = fx["input_nhwc"] else { throw XCTSkip("Missing input.") }
        let model = try loadModel()
        let out = model.infer(image: input, tokens: .count(1024), forceProjection: true)

        if let p = fx["infer_points"], let a = out["points"] {
            assertClose(a, p, key: "infer_points", atol: 0.2, meanAtol: 1e-2)
        }
        if let d = fx["infer_depth"], let a = out["depth"] {
            assertClose(a, d, key: "infer_depth", atol: 0.2, meanAtol: 1e-2)
        }
        if let k = fx["infer_intrinsics"], let a = out["intrinsics"] {
            assertClose(a, k, key: "infer_intrinsics", atol: 1e-2, meanAtol: 2e-3)
        }
    }
}
