"""Benchmark the Torch SurGe reference, to compare against `surge-bench` (Swift).

Uses the export-friendly neighborhood attention (the same math the Swift port
implements; real NATTEN OOMs at stage-4 on Mac). Run from the surge repo with
its venv:

    cd /path/to/python/surge
    PYTHONPATH=src:coreai .venv/bin/python \
        /path/to/mlx-swift-surge/Benchmarks/torch_surge_bench.py \
        --image /path/to/MoGe/example_images/01_HouseIndoor.jpg \
        --tokens 1024 --iterations 10
"""

from __future__ import annotations

import argparse
import time

import torch

from _common import SIZE, fixed_input, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Optional real image (resized to size×size).")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--device", default="cpu", help="cpu or mps (mps may OOM at stage 4).")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=10)
    args = ap.parse_args()

    model = load_model(args.device)

    if args.image:
        import cv2

        bgr = cv2.imread(args.image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_AREA)
        arr = rgb.astype("float32") / 255.0
        image = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    else:
        image = fixed_input(args.size)
    image = image.to(args.device)

    def run_once():
        with torch.inference_mode():
            out = model.infer(image, num_tokens=args.tokens)
            if args.device == "mps":
                torch.mps.synchronize()
        return out

    for _ in range(max(0, args.warmup)):
        run_once()

    times = []
    for _ in range(max(1, args.iterations)):
        start = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - start)

    times.sort()
    mean = sum(times) / len(times)
    median = times[len(times) // 2]
    print("backend=torch")
    print("model=karimknaebel/surge-large")
    print(f"device={args.device}")
    print(f"source_size={args.size}x{args.size}")
    print(f"tokens={args.tokens}")
    print(f"warmup={args.warmup}")
    print(f"iterations={len(times)}")
    print(f"mean_s={mean:.6f}")
    print(f"median_s={median:.6f}")
    print(f"min_s={times[0]:.6f}")
    print(f"max_s={times[-1]:.6f}")


if __name__ == "__main__":
    main()
