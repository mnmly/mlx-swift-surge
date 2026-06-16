// DINOv2 Vision Transformer backbone (ViT-L/14) for SurGe.
//
// Ported from the `karimknaebel/dinov2-core` variant that SurGe loads via
// torch.hub. That variant has NO register tokens and NO mask token — only a
// class token + interpolated positional embeddings. Adapted from the MoGe
// Swift port; the one parity-relevant change is using exact bicubic
// (`bicubicResize`) for positional-embedding interpolation.

import MLX
import MLXNN
import MLXFast

/// 2D image to patch embedding: (B, H, W, C) -> (B, N, D)
public class PatchEmbed: Module {
    public let patchSize: Int
    @ModuleInfo(key: "proj")
    private var proj: Conv2d

    public init(patchSize: Int = 14, inChannels: Int = 3, embedDim: Int = 1024) {
        self.patchSize = patchSize
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: .init(patchSize),
            stride: .init(patchSize),
            padding: 0,
            bias: true
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let H = x.dim(1)
        let W = x.dim(2)
        var out = proj(x) // (B, H', W', D)
        let pH = H / patchSize
        let pW = W / patchSize
        out = out.reshaped([B, pH * pW, -1])
        return out
    }
}

/// Standard MLP with GELU activation: FC1 -> GELU -> FC2
public class DinoMlp: Module {
    @ModuleInfo(key: "fc1") private var fc1: Linear
    @ModuleInfo(key: "fc2") private var fc2: Linear

    public init(inFeatures: Int, hiddenFeatures: Int? = nil, outFeatures: Int? = nil, bias: Bool = true) {
        let outF = outFeatures ?? inFeatures
        let hidden = hiddenFeatures ?? inFeatures
        self._fc1.wrappedValue = Linear(inFeatures, hidden, bias: bias)
        self._fc2.wrappedValue = Linear(hidden, outF, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(GELU()(fc1(x)))
    }
}

/// Multi-head self-attention with joint QKV projection.
public class DinoAttention: Module {
    public let numHeads: Int
    private let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "qkv") private var qkv: Linear
    @ModuleInfo(key: "proj") private var proj: Linear

    public init(dim: Int, numHeads: Int = 16, qkvBias: Bool = true, projBias: Bool = true) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / Float(self.headDim).squareRoot()
        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        self._proj.wrappedValue = Linear(dim, dim, bias: projBias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let N = x.dim(1)
        let C = x.dim(2)

        let qkvOut = qkv(x)
        let qkvTransposed = qkvOut
            .reshaped([B, N, 3, numHeads, headDim])
            .transposed(2, 0, 3, 1, 4)

        let q = qkvTransposed[0]
        let k = qkvTransposed[1]
        let v = qkvTransposed[2]

        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil)

        out = out.transposed(0, 2, 1, 3).reshaped([B, N, C])
        return proj(out)
    }
}

/// Per-dimension learnable scaling (LayerScale).
public class DinoLayerScale: Module {
    @ParameterInfo(key: "gamma") private var gamma: MLXArray

    public init(_ dim: Int, _ initValues: Float = 1e-5) {
        self._gamma = ParameterInfo(
            wrappedValue: MLXArray.full([dim], values: MLXArray(initValues)),
            key: "gamma")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

/// Pre-norm transformer block: LN -> Attn -> LS -> Residual -> LN -> FFN -> LS -> Residual
public class DinoBlock: Module {
    @ModuleInfo(key: "norm1") private var norm1: LayerNorm
    @ModuleInfo(key: "attn") private var attn: DinoAttention
    @ModuleInfo(key: "ls1") private var ls1: DinoLayerScale?
    @ModuleInfo(key: "norm2") private var norm2: LayerNorm
    @ModuleInfo(key: "mlp") private var mlp: DinoMlp
    @ModuleInfo(key: "ls2") private var ls2: DinoLayerScale?

    public init(
        dim: Int, numHeads: Int, mlpRatio: Float = 4.0,
        qkvBias: Bool = true, projBias: Bool = true, ffnBias: Bool = true,
        initValues: Float? = nil
    ) {
        let mlpHidden = Int(Float(dim) * mlpRatio)
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._attn.wrappedValue = DinoAttention(
            dim: dim, numHeads: numHeads, qkvBias: qkvBias, projBias: projBias)
        self._ls1.wrappedValue = initValues.map { DinoLayerScale(dim, $0) }
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._mlp.wrappedValue = DinoMlp(
            inFeatures: dim, hiddenFeatures: mlpHidden, outFeatures: dim, bias: ffnBias)
        self._ls2.wrappedValue = initValues.map { DinoLayerScale(dim, $0) }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attnOut = attn(norm1(x))
        let scaledAttn = ls1.map { $0(attnOut) } ?? attnOut
        var h = x + scaledAttn
        let ffnOut = mlp(norm2(h))
        let scaledFFN = ls2.map { $0(ffnOut) } ?? ffnOut
        h = h + scaledFFN
        return h
    }
}

/// DINOv2 ViT with intermediate-layer extraction and positional-embedding
/// interpolation (bicubic, matching the Python reference).
public class DinoVisionTransformer: Module {
    public let patchSize: Int
    public let embedDim: Int
    private let interpolateOffset: Float

    @ModuleInfo(key: "patch_embed") private var patchEmbed: PatchEmbed
    @ParameterInfo(key: "cls_token") private var clsToken: MLXArray   // (1, 1, D)
    @ParameterInfo(key: "pos_embed") private var posEmbed: MLXArray   // (1, N+1, D)
    @ModuleInfo(key: "blocks") private var blocks: [DinoBlock]
    @ModuleInfo(key: "norm") private var norm: LayerNorm

    public init(
        imgSize: Int = 518,
        patchSize: Int = 14,
        inChans: Int = 3,
        embedDim: Int = 1024,
        depth: Int = 24,
        numHeads: Int = 16,
        mlpRatio: Float = 4.0,
        qkvBias: Bool = true,
        ffnBias: Bool = true,
        projBias: Bool = true,
        initValues: Float? = 1.0,
        interpolateOffset: Float = 0.1
    ) {
        self.patchSize = patchSize
        self.embedDim = embedDim
        self.interpolateOffset = interpolateOffset

        let gridSide = imgSize / patchSize
        let numPatches = gridSide * gridSide

        self._patchEmbed.wrappedValue = PatchEmbed(
            patchSize: patchSize, inChannels: inChans, embedDim: embedDim)
        self._clsToken = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, 1, embedDim]), key: "cls_token")
        self._posEmbed = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, numPatches + 1, embedDim]), key: "pos_embed")

        var blockArray: [DinoBlock] = []
        for _ in 0..<depth {
            blockArray.append(DinoBlock(
                dim: embedDim, numHeads: numHeads, mlpRatio: mlpRatio,
                qkvBias: qkvBias, projBias: projBias, ffnBias: ffnBias,
                initValues: initValues))
        }
        self._blocks.wrappedValue = blockArray
        self._norm.wrappedValue = LayerNorm(dimensions: embedDim, eps: 1e-6)
    }

    /// Interpolate position embeddings for an arbitrary token grid using exact
    /// bicubic interpolation, matching dinov2-core's `interpolate_pos_encoding`
    /// (scale_factor = (target + offset) / sqrt(N), antialias=False).
    private func interpolatePosEncoding(_ x: MLXArray, _ H: Int, _ W: Int) -> MLXArray {
        let npatch = x.shape[1] - 1
        let N = posEmbed.shape[1] - 1
        if npatch == N && W == H {
            return posEmbed
        }

        let classPos = posEmbed[0..., ..<1]   // (1, 1, D)
        var patchPos = posEmbed[0..., 1...]   // (1, N, D)

        let dim = patchPos.shape.last!
        let M = Int(Float(N).squareRoot())
        assert(M * M == N, "Original patches must form a square grid")

        let w0 = W / patchSize
        let h0 = H / patchSize

        // (1, M, M, D)
        patchPos = patchPos.reshaped([1, M, M, dim])

        // dinov2-core: scale_factor=((w0+offset)/m, (h0+offset)/m), m = sqrt(N).
        // patchPos is (1, M[rows], M[cols], D); rows map to image height (w0
        // in the reference's swapped naming), cols to width.
        let scaleRows = (Float(w0) + interpolateOffset) / Float(M)
        let scaleCols = (Float(h0) + interpolateOffset) / Float(M)
        patchPos = bicubicResize(patchPos, scaleH: scaleRows, scaleW: scaleCols)

        patchPos = patchPos.reshaped([1, patchPos.dim(1) * patchPos.dim(2), dim])
        return concatenated([classPos, patchPos], axis: 1)
    }

    /// Embed patches, prepend CLS, add position embeddings.
    private func prepareTokens(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let H = x.dim(1)
        let W = x.dim(2)

        var tokens = patchEmbed(x) // (B, N, D)
        let clsTokens = broadcast(clsToken, to: [B, 1, embedDim])
        tokens = concatenated([clsTokens, tokens], axis: 1)
        tokens = tokens + interpolatePosEncoding(tokens, H, W)
        return tokens
    }

    /// Run all blocks, capturing the post-norm output at the requested layer
    /// indices (preserving the requested order). Returns flattened pairs
    /// `[patches_i, cls_i, ...]` when `returnClassToken` is true, else
    /// `[patches_i, ...]`.
    public func getIntermediateLayers(
        _ x: MLXArray, indices: [Int], returnClassToken: Bool = false
    ) -> [MLXArray] {
        let tokens = prepareTokens(x)
        let layersToTake = Set(indices)

        var byIndex: [Int: MLXArray] = [:]
        var current = tokens
        for (i, blk) in blocks.enumerated() {
            current = blk(current)
            if layersToTake.contains(i) {
                byIndex[i] = current
            }
        }

        var result: [MLXArray] = []
        for idx in indices {
            let normalized = norm(byIndex[idx]!)
            let clsTok = normalized[0..., 0]       // (B, D)
            let patchToks = normalized[0..., 1...] // (B, N, D)
            result.append(patchToks)
            if returnClassToken { result.append(clsTok) }
        }
        return result
    }
}
