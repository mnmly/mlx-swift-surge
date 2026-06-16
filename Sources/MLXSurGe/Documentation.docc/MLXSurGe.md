# ``MLXSurGe``

Monocular surface-geometry prediction on Apple Silicon — a Swift / MLX port of
SurGe (DINOv2 ViT-L/14 encoder + Neighborhood Attention Decoder).

## Overview

`MLXSurGe` turns a single RGB image into an affine point map and recovers a
depth map and camera intrinsics, running entirely on-device via
[mlx-swift](https://github.com/ml-explore/mlx-swift) — no Python or PyTorch at
runtime.

The pipeline is: a **DINOv2 ViT-L/14** encoder (with bicubic positional-embedding
interpolation), a **Neighborhood Attention Decoder** (`NAD`) head that refines a
coarse feature map up to full resolution with windowed attention + rotary
position embeddings, and a host-side **Levenberg–Marquardt** focal/shift recovery
that yields metric-style depth and intrinsics.

The highest-level entry point is ``SurGePipeline`` (CGImage in, geometry out).
For tensor-level control use ``SurGeModel`` directly. Everything runs NHWC in
float32 by default, matching SurGe's macOS quality build.

```swift
import MLXSurGe

let pipeline = try SurGePipeline.fromPretrained(snapshotDir, dtype: .float32)
let prediction = pipeline(cgImage, tokens: .min)   // .min / .max / .count(n)
let points = prediction.points!        // (H, W, 3)
let depth = prediction.depth!          // (H, W)
let intrinsics = prediction.intrinsics! // (3, 3)
```

## Topics

### Running inference

- ``SurGePipeline``
- ``SurGePrediction``
- ``SurGeTokens``
- ``MLXSurGe``

### Shared driver (CLI + app)

- ``SurGeSession``
- ``SurGeSessionConfig``
- ``SurGeFrame``
- ``SurGeBenchResult``
- ``SurGeModelDownloader``

### Model and configuration

- ``SurGeModel``
- ``SurGeEncoder``
- ``SurGeConfig``

### Decoder building blocks

- ``NAD``
- ``NADStage``
- ``NADBlock``
- ``NeighborhoodAttention2d``
- ``UpsampleBlock``
- ``ConvTranspose2d``

### Encoder building blocks

- ``DinoVisionTransformer``
- ``PatchEmbed``
- ``DinoBlock``
- ``DinoAttention``

### Geometry and image utilities

- ``normalizedViewPlaneUV(width:height:aspectRatio:dtype:)``
- ``depthMapToPointMap(depth:intrinsics:height:width:)``
- ``intrinsicsFromFocalCenter(fx:fy:cx:cy:)``
- ``bicubicResize(_:scaleH:scaleW:)``
- ``bilinearResize(_:_:_:)``
- ``ropeEmbedding(height:width:headDim:temperature:dtype:)``
- ``applyRotaryEmbedding(_:_:)``

<!--
Topics are grouped by user task. Add new public symbols to the matching
group when extending the package; keep doc-comment cross-references in
signature-sensitive double-backticks.
-->
