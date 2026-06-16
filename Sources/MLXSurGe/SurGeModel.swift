// SurGe model: DINOv2 encoder + NAD point-map head.
//
// Mirrors `surge/models/surge.py`'s `forward` (image -> affine point map):
// encode -> sum-reduce projected intermediate layers -> concat per-level UV
// grids -> NAD head -> resize to input size -> exp remap.
//
// Everything is NHWC. The trunk + head run in the model dtype (float32 by
// default, matching the reference macOS quality build, where the Python head's
// `torch.autocast(float32)` is a no-op).

import MLX
import MLXNN
import Foundation

/// DINOv2 backbone wrapper: extracts intermediate layers, projects each with a
/// 1×1 conv, and sum-reduces. Mirrors `DINOv2Encoder` + `BaseEncoder.forward`.
public class SurGeEncoder: Module {
    public let patchSize: Int
    private let intermediateLayerIndices: [Int]
    private let reduction: String

    @ModuleInfo(key: "backbone") private var backbone: DinoVisionTransformer
    @ModuleInfo(key: "output_projections") private var outputProjections: [Conv2d]
    @ParameterInfo(key: "image_mean") private var imageMean: MLXArray // (1,1,1,3)
    @ParameterInfo(key: "image_std") private var imageStd: MLXArray    // (1,1,1,3)

    public init(
        backbone: String = "dinov2_vitl14",
        intermediateLayers: [Int] = [5, 11, 17, 23],
        dimOut: Int = 1024,
        reduction: String = "sum"
    ) {
        let configs: [String: (embedDim: Int, depth: Int, numHeads: Int, patchSize: Int)] = [
            "dinov2_vits14": (384, 12, 6, 14),
            "dinov2_vitb14": (768, 12, 12, 14),
            "dinov2_vitl14": (1024, 24, 16, 14),
            "dinov2_vitg14": (1536, 40, 24, 14),
        ]
        guard let cfg = configs[backbone] else {
            fatalError("Unknown DINOv2 backbone: \(backbone)")
        }
        self.patchSize = cfg.patchSize
        self.intermediateLayerIndices = intermediateLayers
        self.reduction = reduction

        self._backbone.wrappedValue = DinoVisionTransformer(
            imgSize: 518, patchSize: cfg.patchSize, inChans: 3,
            embedDim: cfg.embedDim, depth: cfg.depth, numHeads: cfg.numHeads,
            mlpRatio: 4.0, qkvBias: true, ffnBias: true, projBias: true,
            initValues: 1.0, interpolateOffset: 0.1)

        var projs: [Conv2d] = []
        for _ in 0..<intermediateLayers.count {
            projs.append(Conv2d(
                inputChannels: cfg.embedDim, outputChannels: dimOut,
                kernelSize: .init(1), bias: true))
        }
        self._outputProjections.wrappedValue = projs

        self._imageMean = ParameterInfo(
            wrappedValue: MLXArray([0.485, 0.456, 0.406] as [Float]).reshaped([1, 1, 1, 3]),
            key: "image_mean")
        self._imageStd = ParameterInfo(
            wrappedValue: MLXArray([0.229, 0.224, 0.225] as [Float]).reshaped([1, 1, 1, 3]),
            key: "image_std")
    }

    /// - image: NHWC `(B, H, W, 3)` in [0, 1].
    /// - Returns: `(features (B, tokenRows, tokenCols, dimOut), clsToken (B, D))`.
    public func callAsFunction(
        _ image: MLXArray, tokenRows: Int, tokenCols: Int
    ) -> (MLXArray, MLXArray) {
        let targetH = tokenRows * patchSize
        let targetW = tokenCols * patchSize
        var x = bilinearResize(image, targetH, targetW)
        x = (x - imageMean) / imageStd

        let feats = backbone.getIntermediateLayers(
            x, indices: intermediateLayerIndices, returnClassToken: true)

        var projectedSum: MLXArray? = nil
        var lastCls: MLXArray? = nil
        for i in stride(from: 0, to: feats.count, by: 2) {
            let feat = feats[i]        // (B, N, D)
            lastCls = feats[i + 1]     // (B, D)
            let B = feat.dim(0)
            let feat2D = feat.reshaped([B, tokenRows, tokenCols, -1])
            let projected = outputProjections[i / 2](feat2D)
            projectedSum = (projectedSum == nil) ? projected : projectedSum! + projected
        }
        // reduction "sum" already accumulated above; "mean" divides by count.
        var out = projectedSum!
        if reduction == "mean" {
            out = out / Float(intermediateLayerIndices.count)
        }
        return (out, lastCls!)
    }
}

/// SurGe: monocular surface-geometry / point-map model.
public class SurGeModel: Module {
    @ModuleInfo(key: "encoder") public var encoder: SurGeEncoder
    @ModuleInfo(key: "points_head") public var pointsHead: NAD

    public let remapOutput: String
    public let concatUv: Bool
    public let numTokensRange: [Int]

    public init(config: SurGeConfig, useFusedKernel: Bool = true) {
        self.remapOutput = config.remapOutput
        self.concatUv = config.concatUv
        self.numTokensRange = config.numTokensRange

        self._encoder.wrappedValue = SurGeEncoder(
            backbone: config.encoder.backbone,
            intermediateLayers: config.encoder.intermediateLayers,
            dimOut: config.encoder.dimOut,
            reduction: config.encoder.reduction)

        let ph = config.pointsHead
        self._pointsHead.wrappedValue = NAD(
            dimIn: ph.dimIn, dimOut: ph.dimOut, embedDim: ph.embedDim,
            depth: ph.depth, headDim: ph.headDim, mlpRatio: ph.mlpRatio,
            kernelSize: ph.kernelSize, dilation: 1,
            qkNorm: ph.qkNorm, ropeTemperatureScale: ph.ropeTemperatureScale,
            upsampleRefineKernelSize: ph.upsampleRefineKernelSize,
            useFusedKernel: useFusedKernel)
    }

    private func remapPoints(_ points: MLXArray) -> MLXArray {
        switch remapOutput {
        case "linear":
            return points
        case "exp":
            let xy = points[.ellipsis, ..<2]
            let z = points[.ellipsis, 2...] // (..., 1)
            let ze = MLX.exp(z)
            return concatenated([xy * ze, ze], axis: -1)
        default:
            fatalError("Invalid remap output type: \(remapOutput)")
        }
    }

    /// Forward pass: NHWC image `(B, H, W, 3)` in [0, 1] -> affine point map
    /// `(B, H, W, 3)`.
    public func callAsFunction(
        _ image: MLXArray, numTokens: Int, resizeOutput: Bool = true
    ) -> MLXArray {
        let B = image.dim(0)
        let imgH = image.dim(1)
        let imgW = image.dim(2)
        let aspectRatio = Float(imgW) / Float(imgH)

        let baseH = Int((Float(numTokens) / aspectRatio).squareRoot())
        let baseW = Int((Float(numTokens) * aspectRatio).squareRoot())

        // Backbone -> single reduced feature (B, baseH, baseW, dimOut).
        let (feature, _) = encoder(image, tokenRows: baseH, tokenCols: baseW)

        // Per-level features: encoder output at level 0, UV-only at levels 1..4.
        var levels: [MLXArray?] = [feature, nil, nil, nil, nil]
        if concatUv {
            for level in 0..<levels.count {
                let scale = 1 << level
                let uvH = baseH * scale
                let uvW = baseW * scale
                var uv = normalizedViewPlaneUV(
                    width: uvW, height: uvH, aspectRatio: aspectRatio, dtype: image.dtype)
                uv = broadcast(uv.expandedDimensions(axis: 0), to: [B, uvH, uvW, 2])
                if let existing = levels[level] {
                    levels[level] = concatenated([existing, uv], axis: -1)
                } else {
                    levels[level] = uv
                }
            }
        }

        var points = pointsHead(levels) // (B, h4, w4, 3) NHWC
        if resizeOutput {
            points = bilinearResize(points, imgH, imgW)
        }
        points = remapPoints(points)
        return points
    }
}
