// Contract tests for the shared driver (SurGeSession) and the model downloader.
// Both frontends (surge-bench, SURGEDemo) inherit correctness from these.

import Foundation
import MLX
import XCTest

@testable import MLXSurGe

final class SessionTests: XCTestCase {

    // Weight-free: cache path + presence detection.
    func testDownloaderPathsAndPresence() throws {
        let cache = SurGeModelDownloader.defaultCacheDirectory()
        XCTAssertTrue(cache.path.contains("MLXSurGe"))
        XCTAssertTrue(cache.path.contains("karimknaebel--surge-large"))

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        XCTAssertFalse(SurGeModelDownloader.isDownloaded(at: tmp))
        for f in SurGeModelDownloader.files {
            try Data().write(to: tmp.appendingPathComponent(f))
        }
        XCTAssertTrue(SurGeModelDownloader.isDownloaded(at: tmp))
    }

    private func weightsPath() throws -> String {
        let env = ProcessInfo.processInfo.environment
        let def = "\(NSHomeDirectory())/.cache/huggingface/hub/models--karimknaebel--surge-large"
            + "/snapshots/860cb8c37d6db782df94fac5861e89e4fc228aee"
        let path = env["SURGE_WEIGHTS"] ?? def
        guard FileManager.default.fileExists(atPath: "\(path)/model.safetensors") else {
            throw XCTSkip("SURGE_WEIGHTS not set and default snapshot missing.")
        }
        return path
    }

    // Weight-gated: load a Session and run one inference end to end.
    func testSessionInferProducesFrame() throws {
        let cfg = SurGeSessionConfig(weightsPath: try weightsPath(), tokens: .min)
        let session = try SurGeSession.load(cfg)

        let image = MLXRandom.uniform(low: 0, high: 1, [1, 448, 448, 3]).asType(.float32)
        let frame = session.infer(image)

        XCTAssertEqual(frame.width, 448)
        XCTAssertEqual(frame.height, 448)
        XCTAssertEqual(frame.depth.count, 448 * 448)
        XCTAssertEqual(frame.depthRGBA().count, 448 * 448 * 4)
        XCTAssertGreaterThan(frame.inferenceSeconds, 0)

        // Benchmark path (the CLI's entry) returns sane timing.
        let r = session.benchmark(image: image, warmup: 0, iterations: 2)
        XCTAssertEqual(r.iterations, 2)
        XCTAssertGreaterThan(r.medianSeconds, 0)

        // Point-cloud path.
        let cloud = session.inferPointCloud(image, maxPoints: 50_000)
        XCTAssertGreaterThan(cloud.count, 0)
        XCTAssertLessThanOrEqual(cloud.count, 50_000)
        XCTAssertEqual(cloud.positions.count, cloud.count * 3)
        XCTAssertEqual(cloud.colors.count, cloud.count * 4)
        XCTAssertGreaterThan(cloud.radius, 0)

        // Full geometry path (point cloud + mesh + normal texture for the app).
        let geo = session.inferGeometry(image, maxPoints: 50_000, meshStride: 2)
        XCTAssertGreaterThan(geo.pointCount, 0)
        XCTAssertEqual(geo.pointUVs.count, geo.pointCount * 2)
        XCTAssertGreaterThan(geo.vertexCount, 0)
        XCTAssertGreaterThan(geo.faceCount, 0)
        XCTAssertEqual(geo.meshIndices.count, geo.faceCount * 3)
        XCTAssertEqual(geo.meshUVs.count, geo.vertexCount * 2)
        XCTAssertEqual(geo.photoRGBA.count, geo.texWidth * geo.texHeight * 4)
        XCTAssertEqual(geo.normalRGBA.count, geo.texWidth * geo.texHeight * 4)
    }
}
