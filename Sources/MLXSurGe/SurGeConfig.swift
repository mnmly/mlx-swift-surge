// Configuration for the SurGe model, decoded from the HuggingFace `config.json`
// shipped with `karimknaebel/surge-large`.

import Foundation

public struct SurGeConfig: Codable, Sendable {
    public struct Encoder: Codable, Sendable {
        public var backbone: String
        public var intermediateLayers: [Int]
        public var dimOut: Int
        private let _reduction: String?
        public var reduction: String { _reduction ?? "sum" }

        enum CodingKeys: String, CodingKey {
            case backbone
            case intermediateLayers = "intermediate_layers"
            case dimOut = "dim_out"
            case _reduction = "reduction"
        }
    }

    public struct PointsHead: Codable, Sendable {
        public var dimIn: [Int?]
        public var dimOut: [Int?]
        public var embedDim: [Int]
        public var depth: [Int]
        public var headDim: Int
        public var kernelSize: Int
        private let _mlpRatio: Float?
        public var mlpRatio: Float { _mlpRatio ?? 4.0 }
        private let _qkNorm: Bool?
        public var qkNorm: Bool { _qkNorm ?? true }
        private let _rope: Bool?
        public var rope: Bool { _rope ?? true }
        public var ropeTemperatureScale: Float
        private let _upsampleRefineKernelSize: Int?
        public var upsampleRefineKernelSize: Int { _upsampleRefineKernelSize ?? 3 }

        enum CodingKeys: String, CodingKey {
            case dimIn = "dim_in"
            case dimOut = "dim_out"
            case embedDim = "embed_dim"
            case depth
            case headDim = "head_dim"
            case kernelSize = "kernel_size"
            case _mlpRatio = "mlp_ratio"
            case _qkNorm = "qk_norm"
            case _rope = "rope"
            case ropeTemperatureScale = "rope_temperature_scale"
            case _upsampleRefineKernelSize = "upsample_refine_kernel_size"
        }
    }

    public var encoder: Encoder
    public var pointsHead: PointsHead
    public var remapOutput: String
    private let _concatUv: Bool?
    public var concatUv: Bool { _concatUv ?? true }
    public var numTokensRange: [Int]

    enum CodingKeys: String, CodingKey {
        case encoder
        case pointsHead = "points_head"
        case remapOutput = "remap_output"
        case _concatUv = "concat_uv"
        case numTokensRange = "num_tokens_range"
    }

    public static func load(from url: URL) throws -> SurGeConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(SurGeConfig.self, from: data)
    }
}
