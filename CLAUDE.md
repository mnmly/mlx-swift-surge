# mlx-swift-surge — agent notes

Swift / mlx-swift port of SurGe (DINOv2 ViT-L/14 encoder + Neighborhood
Attention Decoder). Predicts an affine point map, depth, and camera intrinsics
from a single image. Everything runs **NHWC** in **float32** by default.

## Building & testing

- Use **`xcodebuild`**, not `swift build`/`swift test` — `swift test` can't load
  mlx-swift's `default.metallib` at runtime and crashes on the first MLX op.
  ```bash
  xcodebuild -scheme mlx-swift-surge-Package -destination 'platform=macOS' \
      -derivedDataPath .xcdd build
  xcodebuild -scheme mlx-swift-surge-Package -destination 'platform=macOS' \
      -derivedDataPath .xcdd test
  ```
- Shape tests run without weights. Parity tests skip unless `SURGE_WEIGHTS`
  (default: the HF cache snapshot for `karimknaebel/surge-large`) and a fixture
  from `Scripts/generate_fixtures.py` are present.
- Benchmark / leak check in **Release** (`surge-bench`); `active_mem_delta_mb`
  should stay ~0 across iterations.

## Parity invariants

- Numerical correctness is gated on **`forward_points`** (the affine point map,
  meanAbs ~1.8e-3 vs Torch). The encoder feature is an *intermediate* with ~0.4%
  benign cross-framework fp32 drift — don't chase it.
- Bicubic pos-embed interpolation is bit-exact (`bicubicResize`); preserve its
  PyTorch half-pixel `scale_factor` formula if you touch it.
- Neighborhood attention defaults to a fused `MLXFast.metalKernel`
  (`useFusedKernel: true`, ~5× faster). The gather formulation
  (`useFusedKernel: false`, matches NATTEN ~1e-6) is the parity oracle; keep
  both paths under test (`KernelParityTests` + both `forward_points_*` fixtures).
  If you change the kernel, re-run gather-vs-fused parity before trusting it.

## Documentation

`MLXSurGe` ships DocC-generated reference docs (see
`Sources/MLXSurGe/Documentation.docc/` and `Scripts/build_docs.sh`).
**`///` doc comments on public/`open` symbols are published.**

When you add or modify a `public` or `open` declaration:

- Write a `///` doc comment: one-sentence summary, then a paragraph if the *why*
  is non-obvious. Don't restate the signature.
- Document each parameter with `- Parameter name:` using the **internal** name
  when there's an external label (DocC warns otherwise).
- Cross-reference related symbols with signature-sensitive double-backtick
  links, e.g. `` ``SurGeModel/infer(image:tokens:resizeOutput:forceProjection:fovX:)`` ``.
- Add new top-level public symbols under the matching `## Topics` group in
  `Sources/MLXSurGe/Documentation.docc/MLXSurGe.md` (organized by *user task*).

Verify before declaring documentation work done:

```bash
Scripts/build_docs.sh
```

Expect exit 0 and no new "doesn't exist at" / "external name used to document
parameter" warnings attributable to your changes.
