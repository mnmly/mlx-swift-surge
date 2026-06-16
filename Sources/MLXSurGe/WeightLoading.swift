// Weight loading for SurGe. Loads the HuggingFace checkpoint directly
// (`config.json` + `model.safetensors` from `karimknaebel/surge-large`) — no
// offline conversion step. PyTorch convolution weights are transposed to MLX's
// NHWC kernel layout at load time; everything else (Linear, LayerNorm,
// LayerScale, rope-free params) keeps its key and layout.

import Foundation
import MLX
import MLXNN

public enum SurGeWeightLoadingError: Error, CustomStringConvertible {
    case missingFile(String)

    public var description: String {
        switch self {
        case .missingFile(let p): return "Missing file: \(p)"
        }
    }
}

extension SurGeModel {

    /// Load a `SurGeModel` from a directory containing `config.json` and
    /// `model.safetensors` (the standard HuggingFace snapshot layout). Uses the
    /// fused neighborhood-attention kernel by default; pass
    /// `useFusedKernel: false` for the gather reference.
    public static func fromPretrained(
        path: String, dtype: DType = .float32, useFusedKernel: Bool = true
    ) throws -> SurGeModel {
        let dir = URL(fileURLWithPath: path)
        return try fromPretrained(
            configURL: dir.appendingPathComponent("config.json"),
            weightsURL: dir.appendingPathComponent("model.safetensors"),
            dtype: dtype, useFusedKernel: useFusedKernel)
    }

    public static func fromPretrained(
        configURL: URL, weightsURL: URL, dtype: DType = .float32, useFusedKernel: Bool = true
    ) throws -> SurGeModel {
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw SurGeWeightLoadingError.missingFile(configURL.path)
        }
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw SurGeWeightLoadingError.missingFile(weightsURL.path)
        }

        let config = try SurGeConfig.load(from: configURL)
        let model = SurGeModel(config: config, useFusedKernel: useFusedKernel)

        let weights = try MLX.loadArrays(url: weightsURL)
        var remapped: [(String, MLXArray)] = []
        remapped.reserveCapacity(weights.count)
        for (key, value) in weights {
            remapped.append((key, transformWeight(key: key, value: value).asType(dtype)))
        }

        MLX.eval(remapped.map { $0.1 })
        try model.update(
            parameters: ModuleParameters.unflattened(remapped), verify: [.noUnusedKeys])
        MLX.eval(model)
        return model
    }

    /// Transpose PyTorch conv weights / normalization buffers into MLX layout.
    /// Keys are otherwise identical to the Swift module tree.
    static func transformWeight(key: String, value: MLXArray) -> MLXArray {
        // ImageNet stats: (1, 3, 1, 1) -> (1, 1, 1, 3)
        if key == "encoder.image_mean" || key == "encoder.image_std" {
            return value.reshaped([1, 1, 1, 3])
        }

        // ConvTranspose2d resample: PyTorch (C_in, C_out, kH, kW) -> (C_out, kH, kW, C_in)
        if key.hasSuffix("upsample.resample.weight") {
            return value.transposed(1, 2, 3, 0)
        }

        // Standard Conv2d weights: PyTorch (C_out, C_in, kH, kW) -> (C_out, kH, kW, C_in)
        // Covers: patch_embed.proj, encoder.output_projections.N, upsample.refine
        if key.hasSuffix("patch_embed.proj.weight")
            || key.hasSuffix("upsample.refine.weight")
            || (key.contains("output_projections.") && key.hasSuffix(".weight"))
        {
            return value.transposed(0, 2, 3, 1)
        }

        return value
    }
}
