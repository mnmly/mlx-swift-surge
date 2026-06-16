import Metal
import MLXSurGe
import RealityKit
import SwiftUI

/// Renders a `SurGePointCloud` in a RealityKit `RealityView` using a
/// `LowLevelMesh` with `.point` topology (position + per-vertex color). Mouse
/// drag orbits, scroll zooms (`.realityViewCameraControls(.orbit)`).
///
/// Points render at the GPU default size; a dense cloud reads well. The cloud is
/// re-centered on its bounding-sphere center and scaled to ~unit so the orbit
/// camera frames it regardless of scene scale.
struct PointCloudView: View {
    let cloud: SurGePointCloud

    var body: some View {
        // Single make-closure: the caller forces a fresh view (and a rebuild)
        // per inference with `.id(...)`, so we never mutate `content.entities`
        // mid-iteration (which crashes RealityKit) or rebuild on spurious
        // SwiftUI updates.
        RealityView { content in
            if let entity = try? Self.makeEntity(cloud) { content.add(entity) }
        }
        .realityViewCameraControls(.orbit)
    }

    @MainActor
    static func makeEntity(_ cloud: SurGePointCloud) throws -> Entity {
        let n = cloud.count
        guard n > 0 else { return Entity() }

        let stride = 16 // float3 position (12) + uchar4 color (4)
        var desc = LowLevelMesh.Descriptor()
        desc.vertexCapacity = n
        desc.vertexAttributes = [
            LowLevelMesh.Attribute(semantic: .position, format: .float3, layoutIndex: 0, offset: 0),
            LowLevelMesh.Attribute(semantic: .color, format: .uchar4Normalized, layoutIndex: 0, offset: 12),
        ]
        desc.vertexLayouts = [
            LowLevelMesh.Layout(bufferIndex: 0, bufferOffset: 0, bufferStride: stride)
        ]
        desc.indexCapacity = n
        desc.indexType = .uint32

        let mesh = try LowLevelMesh(descriptor: desc)

        cloud.positions.withUnsafeBufferPointer { pos in
            cloud.colors.withUnsafeBufferPointer { col in
                mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in
                    let base = raw.baseAddress!
                    for i in 0..<n {
                        let vp = base + i * stride
                        vp.storeBytes(of: pos[i * 3 + 0], toByteOffset: 0, as: Float.self)
                        vp.storeBytes(of: pos[i * 3 + 1], toByteOffset: 4, as: Float.self)
                        vp.storeBytes(of: pos[i * 3 + 2], toByteOffset: 8, as: Float.self)
                        vp.storeBytes(of: col[i * 4 + 0], toByteOffset: 12, as: UInt8.self)
                        vp.storeBytes(of: col[i * 4 + 1], toByteOffset: 13, as: UInt8.self)
                        vp.storeBytes(of: col[i * 4 + 2], toByteOffset: 14, as: UInt8.self)
                        vp.storeBytes(of: col[i * 4 + 3], toByteOffset: 15, as: UInt8.self)
                    }
                }
            }
        }

        mesh.withUnsafeMutableIndices { raw in
            let idx = raw.bindMemory(to: UInt32.self)
            for i in 0..<n { idx[i] = UInt32(i) }
        }

        let bounds = BoundingBox(
            min: cloud.center - SIMD3<Float>(repeating: cloud.radius),
            max: cloud.center + SIMD3<Float>(repeating: cloud.radius))
        mesh.parts.replaceAll([
            LowLevelMesh.Part(indexOffset: 0, indexCount: n, topology: .point, materialIndex: 0, bounds: bounds)
        ])

        let resource = try MeshResource(from: mesh)
        var material = UnlitMaterial()
        material.faceCulling = .none
        let model = ModelEntity(mesh: resource, materials: [material])

        // Re-center on the cloud center, then scale the parent to ~unit size.
        let root = Entity()
        model.position = -cloud.center
        root.scale = SIMD3<Float>(repeating: 1 / cloud.radius)
        root.addChild(model)
        return root
    }
}
