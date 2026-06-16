// Neighborhood Attention Decoder (NAD) — SurGe's point-map head.
//
// Ports `surge/modules/heads/nad.py` for the shipped `surge-large` config:
//   embed_dim = [1024, 512, 256, 128, 64], depth 3/stage, head_dim 64,
//   kernel 9, dilation 1, qk_norm = true, rope = true, init_values = None
//   (no LayerScale), norm_layer = None (no block LayerNorm), act = relu,
//   upsample = conv_transpose + 3×3 replicate-padded refine.
//
// Everything runs NHWC. The head consumes the encoder feature (level 0) plus
// per-level UV grids (levels 1–4) and produces a `(B, H, W, 3)` affine point
// map at the finest stage.

import MLX
import MLXNN
import Foundation

/// 2× upsample via native transposed convolution. PyTorch `ConvTranspose2d`
/// weight `(C_in, C_out, kH, kW)` is transposed to `(C_out, kH, kW, C_in)` by
/// the weight loader; here we just call `convTransposed2d`.
public class ConvTranspose2d: Module {
    @ParameterInfo(key: "weight") private var weight: MLXArray // (C_out, kH, kW, C_in)
    @ParameterInfo(key: "bias") private var bias: MLXArray?
    let stride: Int

    public init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, bias: Bool = true) {
        let scale = 1.0 / Float(inChannels * kernelSize * kernelSize).squareRoot()
        self._weight = ParameterInfo(
            wrappedValue: MLXRandom.uniform(
                low: -scale, high: scale,
                [outChannels, kernelSize, kernelSize, inChannels]),
            key: "weight")
        self._bias = ParameterInfo(
            wrappedValue: bias ? MLXArray.zeros([outChannels]) : nil, key: "bias")
        self.stride = stride
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convTransposed2d(x, weight, stride: .init(stride), padding: .init(0))
        if let bias { y = y + bias }
        return y
    }
}

/// Upsample block: transposed-conv resample to `next_embed_dim`, then a 3×3
/// replicate-padded refine conv. NHWC throughout.
public class UpsampleBlock: Module {
    @ModuleInfo(key: "resample") private var resample: ConvTranspose2d
    @ModuleInfo(key: "refine") private var refine: Conv2d
    private let refinePad: Int

    public init(
        inChannels: Int, outChannels: Int, scaleFactor: Int = 2,
        refineKernelSize: Int = 3
    ) {
        self._resample.wrappedValue = ConvTranspose2d(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: scaleFactor, stride: scaleFactor)
        self.refinePad = refineKernelSize / 2
        self._refine.wrappedValue = Conv2d(
            inputChannels: outChannels, outputChannels: outChannels,
            kernelSize: .init(refineKernelSize), stride: .init(1),
            padding: 0, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = resample(x)            // (B, 2H, 2W, outC)
        out = padReplicate(out, refinePad)
        out = refine(out)
        return out
    }
}

/// NAD transformer block: `x = x + attn(x); x = x + mlp(x)` (no norm / no
/// LayerScale for this config). The MLP is a 3-element `[UnaryLayer]`
/// `[Linear, ReLU, Linear]` so weight keys `mlp.0` / `mlp.2` map directly.
public class NADBlock: Module {
    @ModuleInfo(key: "attn") private var attn: NeighborhoodAttention2d
    @ModuleInfo(key: "mlp") private var mlp: [UnaryLayer]

    public init(
        dim: Int, numHeads: Int, mlpRatio: Float = 4.0,
        qkNorm: Bool, kernelSize: (Int, Int), dilation: (Int, Int),
        useFusedKernel: Bool = true
    ) {
        self._attn.wrappedValue = NeighborhoodAttention2d(
            embedDim: dim, numHeads: numHeads,
            kernelSize: kernelSize, dilation: dilation,
            qkvBias: true, qkNorm: qkNorm, useFusedKernel: useFusedKernel)
        let hidden = Int(Float(dim) * mlpRatio)
        let layers: [UnaryLayer] = [
            Linear(dim, hidden, bias: true),
            ReLU(),
            Linear(hidden, dim, bias: true),
        ]
        self._mlp.wrappedValue = layers
    }

    public func callAsFunction(_ x: MLXArray, ropeEmbed: MLXArray?) -> MLXArray {
        var h = x + attn(x, ropeEmbed: ropeEmbed)
        var m = h
        for layer in mlp { m = layer(m) }
        h = h + m
        return h
    }
}

/// One NAD stage: input projection, transformer blocks, optional output
/// projection (last stage), optional upsample (non-last stages).
public class NADStage: Module {
    @ModuleInfo(key: "input_projection") private var inputProjection: Linear
    @ModuleInfo(key: "blocks") private var blocks: [NADBlock]
    @ModuleInfo(key: "output_projection") private var outputProjection: Linear?
    @ModuleInfo(key: "upsample") private var upsample: UpsampleBlock?

    private let headDim: Int
    private let ropeTemperature: Float

    public init(
        dimIn: Int, dimOut: Int?, embedDim: Int, nextEmbedDim: Int?,
        depth: Int, headDim: Int, mlpRatio: Float,
        kernelSize: (Int, Int), dilation: (Int, Int),
        qkNorm: Bool, ropeTemperatureScale: Float,
        upsampleRefineKernelSize: Int,
        useFusedKernel: Bool = true
    ) {
        self.headDim = headDim
        // effective_kernel_size = max over dims of dilation*(kernel-1)+1
        let effKH = dilation.0 * (kernelSize.0 - 1) + 1
        let effKW = dilation.1 * (kernelSize.1 - 1) + 1
        let effK = Float(max(effKH, effKW))
        self.ropeTemperature = max(1.0, ropeTemperatureScale * effK)

        self._inputProjection.wrappedValue = Linear(dimIn, embedDim, bias: true)

        let numHeads = embedDim / headDim
        var blockArray: [NADBlock] = []
        for _ in 0..<depth {
            blockArray.append(NADBlock(
                dim: embedDim, numHeads: numHeads, mlpRatio: mlpRatio,
                qkNorm: qkNorm, kernelSize: kernelSize, dilation: dilation,
                useFusedKernel: useFusedKernel))
        }
        self._blocks.wrappedValue = blockArray

        self._outputProjection.wrappedValue = dimOut.map { Linear(embedDim, $0, bias: true) }

        if let nextEmbedDim {
            self._upsample.wrappedValue = UpsampleBlock(
                inChannels: embedDim, outChannels: nextEmbedDim,
                scaleFactor: 2, refineKernelSize: upsampleRefineKernelSize)
        } else {
            self._upsample.wrappedValue = nil
        }
    }

    /// Returns `(output, x)`. `output` is non-nil only on the last stage.
    public func callAsFunction(
        _ feature: MLXArray?, _ x: MLXArray?
    ) -> (MLXArray?, MLXArray) {
        let projected = feature.map { inputProjection($0) } // input_scale = Identity

        var xCur: MLXArray
        if let x {
            xCur = (projected != nil) ? (x + projected!) : x
        } else {
            xCur = projected!
        }

        let rope: MLXArray? = blocks.isEmpty
            ? nil
            : ropeEmbedding(
                height: xCur.dim(1), width: xCur.dim(2),
                headDim: headDim, temperature: ropeTemperature, dtype: .float32)

        for blk in blocks {
            xCur = blk(xCur, ropeEmbed: rope)
        }

        let output = outputProjection.map { $0(xCur) }
        if let upsample {
            xCur = upsample(xCur)
        }
        return (output, xCur)
    }
}

/// NAD head: a stack of `NADStage`s. Consumes NHWC features per level and
/// returns the finest-stage `(B, H, W, dim_out)` output.
public class NAD: Module {
    @ModuleInfo(key: "stages") private var stages: [NADStage]

    public init(
        dimIn: [Int?], dimOut: [Int?], embedDim: [Int],
        depth: [Int], headDim: Int, mlpRatio: Float = 4.0,
        kernelSize: Int = 9, dilation: Int = 1,
        qkNorm: Bool = true, ropeTemperatureScale: Float,
        upsampleRefineKernelSize: Int = 3,
        useFusedKernel: Bool = true
    ) {
        let numStages = embedDim.count
        var stageArray: [NADStage] = []
        for i in 0..<numStages {
            let nextEmbed = i < numStages - 1 ? embedDim[i + 1] : nil
            stageArray.append(NADStage(
                dimIn: dimIn[i] ?? embedDim[i],
                dimOut: dimOut[i],
                embedDim: embedDim[i],
                nextEmbedDim: nextEmbed,
                depth: depth[i],
                headDim: headDim,
                mlpRatio: mlpRatio,
                kernelSize: (kernelSize, kernelSize),
                dilation: (dilation, dilation),
                qkNorm: qkNorm,
                ropeTemperatureScale: ropeTemperatureScale,
                upsampleRefineKernelSize: upsampleRefineKernelSize,
                useFusedKernel: useFusedKernel))
        }
        self._stages.wrappedValue = stageArray
    }

    /// - `inFeatures`: per-level NHWC features (`nil` where absent).
    /// - Returns: finest-stage output `(B, H, W, dim_out)`.
    public func callAsFunction(_ inFeatures: [MLXArray?]) -> MLXArray {
        var x: MLXArray? = nil
        var lastOutput: MLXArray? = nil
        for (stage, feature) in zip(stages, inFeatures) {
            let (output, newX) = stage(feature, x)
            x = newX
            if let output { lastOutput = output }
        }
        return lastOutput!
    }
}
