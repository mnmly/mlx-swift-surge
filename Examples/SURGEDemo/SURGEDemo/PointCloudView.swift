import CoreGraphics
import Metal
import MLXSurGe
import RealityKit
import SwiftUI

/// What a `SurGeRealityView` renders from one `SurGeGeometry`.
enum SurGeRenderKind {
    case pointCloud   // LowLevelMesh points, photo-textured
    case texturedMesh // surface mesh, photo-textured
    case normalMesh   // surface mesh, normal-color-textured
}

/// A RealityKit view of a `SurGeGeometry`. Color comes from UVs + a texture
/// (photo or normal-color) sampled by a plain `UnlitMaterial` — no custom
/// shader. Drag orbits, scroll zooms. The caller forces a fresh build per
/// inference via `.id(...)`, so there's no mid-iteration mutation of content.
struct SurGeRealityView: View {
    let geometry: SurGeGeometry
    let kind: SurGeRenderKind

    var body: some View {
        RealityView { content in
            if let entity = try? await Self.makeEntity(geometry, kind: kind) {
                content.add(entity)
            }
        }
        .realityViewCameraControls(.orbit)
    }

    @MainActor
    static func makeEntity(_ g: SurGeGeometry, kind: SurGeRenderKind) async throws -> Entity {
        let root = Entity()
        guard g.pointCount > 0 else { return root }

        let texBytes = (kind == .normalMesh) ? g.normalRGBA : g.photoRGBA
        let material = makeTexturedMaterial(texBytes, g.texWidth, g.texHeight)

        let model: ModelEntity
        switch kind {
        case .pointCloud:
            model = try pointModel(g, material: material)
        case .texturedMesh, .normalMesh:
            model = try meshModel(g, material: material)
        }

        model.position = -g.center
        root.scale = SIMD3<Float>(repeating: 1 / g.radius)
        root.addChild(model)
        return root
    }

    // MARK: - Material

    static func makeTexturedMaterial(_ rgba: [UInt8], _ w: Int, _ h: Int) -> RealityKit.Material {
        if w > 0, h > 0, let tex = makeTexture(rgba, w, h) {
            var mat = UnlitMaterial()
            mat.color = .init(tint: .white, texture: .init(tex))
            mat.faceCulling = .none
            return mat
        }
        var fallback = UnlitMaterial(color: .gray)
        fallback.faceCulling = .none
        return fallback
    }

    static func makeTexture(_ rgba: [UInt8], _ w: Int, _ h: Int) -> TextureResource? {
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let cg = CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
            space: cs, bitmapInfo: info, provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent)
        else { return nil }
        return try? TextureResource(image: cg, options: .init(semantic: .color))
    }

    // MARK: - Point cloud (LowLevelMesh, points, position + uv)

    static func pointModel(_ g: SurGeGeometry, material: RealityKit.Material) throws -> ModelEntity {
        let n = g.pointCount
        let stride = 20 // float3 position (12) + float2 uv (8)
        var desc = LowLevelMesh.Descriptor()
        desc.vertexCapacity = n
        desc.vertexAttributes = [
            LowLevelMesh.Attribute(semantic: .position, format: .float3, layoutIndex: 0, offset: 0),
            LowLevelMesh.Attribute(semantic: .uv0, format: .float2, layoutIndex: 0, offset: 12),
        ]
        desc.vertexLayouts = [LowLevelMesh.Layout(bufferIndex: 0, bufferOffset: 0, bufferStride: stride)]
        desc.indexCapacity = n
        desc.indexType = .uint32

        let mesh = try LowLevelMesh(descriptor: desc)
        g.pointPositions.withUnsafeBufferPointer { pos in
            g.pointUVs.withUnsafeBufferPointer { uvs in
                mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in
                    let base = raw.baseAddress!
                    for i in 0..<n {
                        let vp = base + i * stride
                        vp.storeBytes(of: pos[i * 3 + 0], toByteOffset: 0, as: Float.self)
                        vp.storeBytes(of: pos[i * 3 + 1], toByteOffset: 4, as: Float.self)
                        vp.storeBytes(of: pos[i * 3 + 2], toByteOffset: 8, as: Float.self)
                        vp.storeBytes(of: uvs[i * 2 + 0], toByteOffset: 12, as: Float.self)
                        vp.storeBytes(of: uvs[i * 2 + 1], toByteOffset: 16, as: Float.self)
                    }
                }
            }
        }
        mesh.withUnsafeMutableIndices { raw in
            let idx = raw.bindMemory(to: UInt32.self)
            for i in 0..<n { idx[i] = UInt32(i) }
        }
        let bounds = BoundingBox(
            min: g.center - SIMD3<Float>(repeating: g.radius),
            max: g.center + SIMD3<Float>(repeating: g.radius))
        mesh.parts.replaceAll([
            LowLevelMesh.Part(indexOffset: 0, indexCount: n, topology: .point, materialIndex: 0, bounds: bounds)
        ])
        return ModelEntity(mesh: try MeshResource(from: mesh), materials: [material])
    }

    // MARK: - Surface mesh (MeshDescriptor triangles + uv)

    static func meshModel(_ g: SurGeGeometry, material: RealityKit.Material) throws -> ModelEntity {
        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(g.vertexCount)
        for v in 0..<g.vertexCount {
            positions.append(SIMD3(g.meshPositions[v * 3], g.meshPositions[v * 3 + 1], g.meshPositions[v * 3 + 2]))
        }
        var uvs: [SIMD2<Float>] = []
        uvs.reserveCapacity(g.vertexCount)
        for v in 0..<g.vertexCount {
            uvs.append(SIMD2(g.meshUVs[v * 2], g.meshUVs[v * 2 + 1]))
        }

        var desc = MeshDescriptor(name: "surge")
        desc.positions = MeshBuffer(positions)
        desc.textureCoordinates = MeshBuffer(uvs)
        desc.primitives = .triangles(g.meshIndices)
        let mesh = try MeshResource.generate(from: [desc])
        return ModelEntity(mesh: mesh, materials: [material])
    }
}
