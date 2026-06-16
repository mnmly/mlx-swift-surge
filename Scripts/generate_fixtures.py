"""Generate per-stage + end-to-end parity fixtures for the SurGe Swift port.

Loads the real `karimknaebel/surge-large` model with the export-friendly
neighborhood attention (the same math the Swift port implements; real NATTEN
OOMs at stage-4 on Mac) and the antialias-off encoder resize, then dumps:

  - input_nhwc            (1, S, S, 3)        deterministic [0,1] image
  - pos_embed_patch_37    (1, 37, 37, 1024)   raw patch pos-embed grid
  - pos_embed_interp_32   (1, G, G, 1024)     bicubic-interpolated grid (isolates bicubic)
  - encoder_feature       (1, G, G, 1024)     summed projected intermediate layers (NHWC)
  - nad_level{0..4}       (1, G*2^l, *, C)    NAD input features after concat_uv (NHWC)
  - forward_points        (1, S, S, 3)        SurGe.forward output (exp-remapped)
  - infer_points/depth/intrinsics             SurGe.infer output

Run from the surge repo with its venv:

    cd /path/to/python/surge
    PYTHONPATH=src:coreai .venv/bin/python \
        /path/to/mlx-swift-surge/Scripts/generate_fixtures.py \
        --out /path/to/mlx-swift-surge/Tests/Fixtures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from _common import GRID, NUM_TOKENS, SIZE, fixed_input, load_model
from surge.utils.geometry_torch import normalized_view_plane_uv


def build_nad_levels(model, image, num_tokens):
    """Replicate SurGe.forward's feature assembly (encoder + concat_uv)."""
    b, _, img_h, img_w = image.shape
    aspect_ratio = img_w / img_h
    base_h = int((num_tokens / aspect_ratio) ** 0.5)
    base_w = int((num_tokens * aspect_ratio) ** 0.5)

    feat, _ = model.encoder(image, base_h, base_w, return_class_token=True)
    features = [feat, None, None, None, None]

    for level in range(len(features)):
        uv = normalized_view_plane_uv(
            width=base_w * 2**level,
            height=base_h * 2**level,
            aspect_ratio=aspect_ratio,
            dtype=image.dtype,
            device=image.device,
        )
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        if features[level] is None:
            features[level] = uv
        else:
            features[level] = torch.concat([features[level], uv], dim=1)
    # features are NCHW; return NHWC for the Swift side.
    return feat, [f.permute(0, 2, 3, 1).contiguous() for f in features], base_h, base_w


def bicubic_pos_embed(model, grid):
    """Isolated bicubic pos-embed interpolation, 37x37 -> grid x grid."""
    backbone = model.encoder.backbone
    pe = backbone.pos_embed.float()           # (1, 1370, 1024)
    n = pe.shape[1] - 1
    m = int(n**0.5)
    patch = pe[:, 1:].reshape(1, m, m, -1)     # (1, 37, 37, 1024) NHWC
    offset = backbone.interpolate_offset
    scale = (grid + offset) / m
    patch_nchw = patch.permute(0, 3, 1, 2)
    interp = F.interpolate(
        patch_nchw, mode="bicubic", antialias=False, scale_factor=(scale, scale)
    ).permute(0, 2, 3, 1).contiguous()         # (1, grid, grid, 1024)
    return patch.contiguous(), interp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output fixture directory.")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--num-tokens", type=int, default=NUM_TOKENS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--image",
        default=None,
        help="Optional real image. Resized to size×size (square) so the encoder "
        "resize stays identity; gives well-conditioned focal/shift geometry. "
        "Falls back to a deterministic noise image when omitted.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model = load_model("cpu")  # patched export-friendly attention + antialias-off resize

    if args.image:
        import cv2

        bgr = cv2.imread(args.image)
        if bgr is None:
            raise FileNotFoundError(args.image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_AREA)
        arr = (rgb.astype("float32") / 255.0)  # (S, S, 3)
        image = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    else:
        image = fixed_input(args.size, args.seed)  # (1, 3, S, S)
    image_nhwc = image.permute(0, 2, 3, 1).contiguous()

    tensors: dict[str, torch.Tensor] = {
        "input_nhwc": image_nhwc.float(),
    }

    with torch.inference_mode():
        grid = int((args.num_tokens) ** 0.5)
        pe37, pe_interp = bicubic_pos_embed(model, grid)
        tensors["pos_embed_patch_37"] = pe37.float()
        tensors["pos_embed_interp_grid"] = pe_interp.float()

        feat_nchw, levels, base_h, base_w = build_nad_levels(model, image, args.num_tokens)
        tensors["encoder_feature"] = feat_nchw.permute(0, 2, 3, 1).contiguous().float()
        for i, lv in enumerate(levels):
            tensors[f"nad_level{i}"] = lv.float()

        forward_out = model(image, num_tokens=args.num_tokens, resize_output=True)
        tensors["forward_points"] = forward_out["points"].contiguous().float()

        infer_out = model.infer(image, num_tokens=args.num_tokens)
        for k in ("points", "depth", "intrinsics"):
            tensors[f"infer_{k}"] = infer_out[k].contiguous().float()

    fixture_path = out_dir / "surge_fixtures.safetensors"
    save_file(tensors, str(fixture_path))

    metadata = {
        "size": args.size,
        "num_tokens": args.num_tokens,
        "grid": grid,
        "base_h": base_h,
        "base_w": base_w,
        "seed": args.seed,
        "keys": {k: list(v.shape) for k, v in tensors.items()},
    }
    (out_dir / "surge_fixtures.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {fixture_path}")
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)}")


if __name__ == "__main__":
    main()
