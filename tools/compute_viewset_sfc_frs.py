#!/usr/bin/env python3
"""
Offline SFC-FRS++ view selection script.
Reads COLMAP sparse reconstruction, computes optimal training view subset,
and writes selection results to model_path.

Usage:
    python tools/compute_viewset_sfc_frs.py \
        --source_path <DATASET_DIR> \
        --model_path <OUTPUT_DIR> \
        --k 48 \
        --strategy sfc_frs_greedy \
        --alpha 1.0 --beta 0.5 --gamma 0.2 --delta 0.1 \
        --num_points 50000 --shared_sample 200 \
        --seed 42
"""
import argparse
import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# scene.colmap_loader itself only needs numpy/collections/struct, but importing
# via scene package can trigger scene/__init__.py -> gaussian_model -> simple_knn._C.
# For portability (CPU-only preprocessing, CI), try normal import first and fall back.
try:
    from scene.colmap_loader import read_extrinsics_binary
except ImportError:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "colmap_loader",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scene", "colmap_loader.py"))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    read_extrinsics_binary = _mod.read_extrinsics_binary

from utils.view_selection import load_points3D_full, select_views, save_selection


def main():
    parser = argparse.ArgumentParser(description="SFC-FRS++ offline view selection")
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--strategy", type=str, default="sfc_frs_greedy",
                        choices=["sfc_frs_greedy", "frs_greedy", "uniform_pose", "random"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--shared_sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--eval", action="store_true", default=True,
                        help="Apply llffhold test/train split (default: True, matching train.py --eval)")
    parser.add_argument("--no-eval", dest="eval", action="store_false",
                        help="Use ALL views as train candidates (no llffhold split)")
    args = parser.parse_args()

    sparse_dir = os.path.join(args.source_path, "sparse", "0")
    # Read .bin directly - consistent with dataset reader fail-fast path
    cam_extr = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    pts3d = load_points3D_full(sparse_dir)

    # Train/test split must match training --eval setting.
    sorted_keys = sorted(cam_extr.keys(),
                         key=lambda k: os.path.basename(cam_extr[k].name).split(".")[0])
    if args.eval:
        train_keys = [k for idx, k in enumerate(sorted_keys) if idx % args.llffhold != 0]
    else:
        train_keys = list(sorted_keys)
    train_extr = {k: cam_extr[k] for k in train_keys}

    print(f"[ViewSelect] Total: {len(cam_extr)}, Train candidates: {len(train_extr)}, K: {args.k}")
    print(f"[ViewSelect] Points3D: {len(pts3d)}")

    sel_ids, sel_names, meta = select_views(
        train_extr, pts3d, args.k, args.strategy,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
        num_points=args.num_points, shared_sample=args.shared_sample, seed=args.seed)

    jp, tp, mp = save_selection(args.model_path, sel_names, meta)
    print(f"[ViewSelect] Selected {len(sel_names)} views")
    print(f"[ViewSelect] -> {jp}")
    print(f"[ViewSelect] -> {tp}")
    print(f"[ViewSelect] -> {mp}")
    if "coverage" in meta:
        print(f"[ViewSelect] Coverage: {meta['coverage']:.4f}")
    print(f"[ViewSelect] Time: {meta['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
