# mlx-swift-surge

📖 [API documentation](https://mnmly.github.io/mlx-swift-surge/)

A Swift / [mlx-swift](https://github.com/ml-explore/mlx-swift) port of
[**SurGe**](https://github.com/karimknaebel/surge) — *Improved Surface Geometry
in Point Maps*. SurGe predicts an affine point map (and recovers depth + camera
intrinsics) from a single RGB image using a **DINOv2 ViT-L/14** encoder and a
**Neighborhood Attention Decoder (NAD)** head.

This package runs the full `image → points / depth / intrinsics` pipeline
natively on Apple Silicon (macOS / iOS / visionOS / tvOS) with no Python or
PyTorch at runtime.

## Architecture

| Stage | Python (`surge`) | Swift (`MLXSurGe`) |
|-------|------------------|--------------------|
| Encoder | DINOv2 ViT-L/14, 4 intermediate layers, 1×1 proj, sum-reduce | `DINOv2.swift` + `SurGeEncoder` |
| Pos-embed interp | `F.interpolate(bicubic)` | exact bicubic (`bicubicResize`) — bit-exact |
| Decoder | NAD: 5 stages, neighborhood attention + rope, conv-transpose upsample | `NAD.swift`, `NeighborhoodAttention.swift`, `Rope.swift` |
| Neighborhood attn | NATTEN fused kernel | gather + softmax reference **or** a fused Metal kernel (opt-in, ~5× faster) |
| Post-process | `recover_focal_shift` (scipy LM) → intrinsics, depth | Levenberg–Marquardt in `SurGeInference.swift` |

### Fused neighborhood-attention kernel

NATTEN's fused kernel has no MLX equivalent. Two implementations are provided:

- **fused Metal kernel** (`useFusedKernel: true`, **the default**) — a single
  `MLXFast.metalKernel` launch with one thread per output and an online softmax
  over the window. **~5× faster** and several GB lower peak memory because it
  never materializes the per-window gathers.
- **gather reference** (`useFusedKernel: false`) — `kernel²` gathers + softmax;
  validated ~1e-6 vs NATTEN. The parity oracle; gather-vs-fused agree to ~1e-6 and
  end-to-end forward parity is identical.

```swift
let pipeline = try SurGePipeline.fromPretrained(dir, dtype: .float32)                       // fused (default)
let reference = try SurGePipeline.fromPretrained(dir, dtype: .float32, useFusedKernel: false) // gather oracle
```

Everything runs **NHWC** in **float32** by default (matching SurGe's macOS
quality build, where the head's `torch.autocast(float32)` is a no-op).

## Installation

```swift
.package(url: "https://github.com/<you>/mlx-swift-surge", from: "0.1.0")
```

Then add the `MLXSurGe` product to your target.

## Usage

```swift
import MLXSurGe
import MLX

// Load from a HuggingFace snapshot dir (config.json + model.safetensors).
let pipeline = try SurGePipeline.fromPretrained(
    "~/.cache/huggingface/hub/models--karimknaebel--surge-large/snapshots/<rev>",
    dtype: .float32)

let prediction = pipeline(cgImage, tokens: .min)   // .min (1024) / .max (2802) / .count(n)
let points = prediction.points!        // (H, W, 3)
let depth = prediction.depth!          // (H, W)
let intrinsics = prediction.intrinsics! // (3, 3)
```

Or drive the model directly with NHWC tensors:

```swift
let model = try SurGeModel.fromPretrained(path: weightsDir, dtype: .float32)
let image = MLXArray(/* NHWC [0,1] */, [1, 448, 448, 3])
let out = model.infer(image: image, tokens: .min)   // ["points", "depth", "intrinsics"]
```

> The model expects the image already at (or near) the model input resolution
> — the host resizes to `tokenGrid × 14`. This mirrors SurGe's on-device
> contract and avoids antialiased-resize discrepancies. For the common
> deployment path (resize-then-infer) the internal resize is identity.

## Weights

Download the official checkpoint (no conversion needed — the loader transposes
conv weights to NHWC at load time):

```
hf download karimknaebel/surge-large
```

## Numerical parity

`Scripts/generate_fixtures.py` dumps per-stage tensors from the Torch reference;
`Tests/MLXSurGeTests/ParityFixtureTests.swift` checks each boundary bottom-up.
On `01_HouseIndoor.jpg` (448², `num_tokens=1024`, float32):

| Boundary | mean abs err | note |
|----------|-------------|------|
| bicubic pos-embed | 1.2e-9 | bit-exact |
| encoder feature | 1.2e-2 | ~0.4% relative — benign 24-layer fp32 drift |
| **forward points** | **1.8e-3** | **0.4% relative — the binding deliverable** |
| infer points / depth | 1.7e-3 / 3.6e-3 | LM solver vs scipy |
| infer intrinsics (fₓ) | <0.7% | |

## Benchmark

`surge-bench` (Swift, MLX GPU, M-series, fp32) median per inference + peak memory:

| tokens | impl | median/infer | peak mem |
|--------|------|-------------|----------|
| 1024 (`min`) | gather | ~1.00 s | 4.8 GB |
| 1024 (`min`) | **fused** | **~0.18 s** | 3.6 GB |
| 2802 (`max`) | gather | ~2.70 s | 8.7 GB |
| 2802 (`max`) | **fused** | **~0.51 s** | 4.8 GB |

For reference, `Benchmarks/torch_surge_bench.py` (Torch **CPU** — the only available
reference, since NATTEN OOMs on MPS) is ~8.5 s at 1024 tokens.

```bash
# Build the CLI (use xcodebuild, not `swift run` — that can't load MLX's metallib):
xcodebuild -scheme surge-bench -configuration Release -destination 'platform=macOS' -derivedDataPath .xcdd build
# Leak check (active memory stays flat across iterations); --no-fused for the gather path:
.xcdd/Build/Products/Release/surge-bench --weights <snapshot-dir> --tokens min --iterations 30
```

`active_mem_delta_mb=0.0` across iterations confirms **no leak** — the multi-GB
`peak_mem_mb` is MLX's reusable buffer cache, not a leak. Bound it in long-lived
consumers with `--cache-limit-mb` (or `GPU.set(cacheLimit:)`).

## Project layout

The CLI and the SwiftUI app drive the **same engine** via a single library-side
``SurGeSession`` (load → infer → benchmark); each frontend owns only its loop and
presentation. See `Sources/MLXSurGe/SurGeSession.swift`.

| Path | What |
|------|------|
| `Sources/MLXSurGe/` | Library: model, `SurGeSession`, `SurGeModelDownloader` |
| `Examples/surge-bench/` | CLI benchmark (drives `SurGeSession`) |
| `Examples/SURGEDemo/` | SwiftUI macOS app: pick an image → **point cloud + textured mesh + normal mesh** (RealityKit, orbit camera), with a **Download Model** button |

The app downloads `karimknaebel/surge-large` (~1.4 GB) into
`~/Library/Caches/MLXSurGe/` on first use via `SurGeModelDownloader`. Because the
app is sandboxed, enable the **Outgoing Connections (Client)** capability for the
download to work.

## Documentation

API reference: **https://mnmly.github.io/mlx-swift-surge/** (built from DocC and
deployed by `.github/workflows/docs.yml`). Build locally with
`Scripts/build_docs.sh` (or `EMIT_LLMS_TXT=1 Scripts/build_docs.sh` for an
`llms.txt` export). It uses `xcodebuild docbuild` + `docc
transform-for-static-hosting` because the SwiftPM-CLI doc build can't locate the
Metal toolchain mlx-swift needs.

## Testing

```bash
xcodebuild -scheme mlx-swift-surge-Package -destination 'platform=macOS' test
```

Shape tests run without weights. Parity tests skip unless `SURGE_WEIGHTS` (or the
default HF cache path) and a generated fixture are present.

## Acknowledgments

Ports [SurGe](https://github.com/karimknaebel/surge) (MIT code; CC BY-NC 4.0
weights). The DINOv2 encoder and MoGe-derived geometry follow the
[`mlx-swift-MoGe`](https://github.com/mnmly/mlx-swift-MoGe) port. SurGe builds on
[MoGe](https://github.com/microsoft/MoGe).
