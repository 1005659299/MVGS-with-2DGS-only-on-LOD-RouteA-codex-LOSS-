#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
except ImportError:
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "colmap_loader",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scene",
            "colmap_loader.py",
        ),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    read_extrinsics_binary = _mod.read_extrinsics_binary
    read_intrinsics_binary = _mod.read_intrinsics_binary

from utils.view_selection import VFConfig, load_points3D_full, save_selection, select_views


def main():
    parser = argparse.ArgumentParser(description="SFC-FRS++ offline view selection")
    bool_action = argparse.BooleanOptionalAction
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument(
        "--strategy",
        type=str,
        default="sfc_frs_greedy",
        choices=["sfc_frs_greedy", "frs_greedy", "uniform_pose", "random"],
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--shared_sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--no-eval", dest="eval", action="store_false")
    parser.add_argument(
        "--vf_backend",
        type=str,
        default="exact_csr_numpy",
        choices=["exact_csr_numpy", "exact_numba", "packed_bitset"],
    )
    parser.add_argument("--vf_enable_point_score", action=bool_action, default=False)
    parser.add_argument("--vf_enable_region", action=bool_action, default=False)
    parser.add_argument("--vf_enable_obs_cap", action=bool_action, default=False)
    parser.add_argument("--vf_enable_viewdep_guard", action=bool_action, default=False)
    parser.add_argument("--vf_enable_shadow", action=bool_action, default=False)
    parser.add_argument("--vf_enable_shadow_guard", action=bool_action, default=False)
    parser.add_argument("--vf_enable_split_safe_voxel", action=bool_action, default=False)
    parser.add_argument("--vf_mask_path", type=str, default="")
    parser.add_argument("--vf_camera_mask_path", type=str, default="")
    parser.add_argument("--vf_edge_margin_px", type=int, default=15)
    parser.add_argument("--images_root", type=str, default="")
    args = parser.parse_args()

    sparse_dir = os.path.join(args.source_path, "sparse", "0")
    cam_extr = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    cam_intr = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    pts3d = load_points3D_full(sparse_dir)

    sorted_keys = sorted(
        cam_extr.keys(),
        key=lambda k: os.path.splitext(os.path.basename(cam_extr[k].name))[0],
    )
    if args.eval:
        train_keys = [k for idx, k in enumerate(sorted_keys) if idx % args.llffhold != 0]
    else:
        train_keys = list(sorted_keys)
    train_extr = {k: cam_extr[k] for k in train_keys}
    image_size_by_name = {
        os.path.splitext(os.path.basename(img.name))[0]: (
            int(cam_intr[img.camera_id].width),
            int(cam_intr[img.camera_id].height),
        )
        for img in train_extr.values()
    }
    vf_cfg = VFConfig(
        backend=args.vf_backend,
        enable_point_score=args.vf_enable_point_score,
        enable_region=args.vf_enable_region,
        enable_obs_cap=args.vf_enable_obs_cap,
        enable_viewdep_guard=args.vf_enable_viewdep_guard,
        enable_shadow=args.vf_enable_shadow,
        enable_shadow_guard=args.vf_enable_shadow_guard,
        enable_split_safe_voxel=args.vf_enable_split_safe_voxel,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        edge_margin_px=args.vf_edge_margin_px,
    )

    print(
        f"[ViewSelect] Total={len(cam_extr)} TrainCandidates={len(train_extr)} "
        f"K={args.k} strategy={args.strategy}"
    )
    print(f"[ViewSelect] Points3D={len(pts3d)} backend={args.vf_backend}")

    result = select_views(
        train_extr,
        pts3d,
        args.k,
        args.strategy,
        vf_cfg=vf_cfg,
        cam_intrinsics=cam_intr,
        image_size_by_name=image_size_by_name,
        images_root=args.images_root or os.path.join(args.source_path, "images"),
        mask_root=args.vf_mask_path,
        camera_mask_root=args.vf_camera_mask_path,
        num_points=args.num_points,
        shared_sample=args.shared_sample,
        seed=args.seed,
    )
    jp, tp, mp = save_selection(
        model_path=args.model_path,
        selected_names=result.selected_names,
        meta=result.meta,
        diagnostics=result.diagnostics,
        support_pack=result.support_pack,
        shadow_pack=result.shadow_pack,
    )
    print(f"[ViewSelect] Selected {len(result.selected_names)} views")
    print(f"[ViewSelect] -> {jp}")
    print(f"[ViewSelect] -> {tp}")
    print(f"[ViewSelect] -> {mp}")
    print(f"[ViewSelect] Time={result.meta['selector_elapsed_seconds']}s PeakMem={result.meta['selector_peak_mem_mb']}MB")


if __name__ == "__main__":
    main()
