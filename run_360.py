import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import GPUtil


SCENE_TABLE = [
    ("bicycle", "360_v2", -1, "largest"),
    ("bonsai", "360_v2", -1, "normal"),
    ("counter", "360_v2", -1, "normal"),
    ("garden", "360_v2", -1, "largest"),
    ("kitchen", "360_v2", -1, "normal"),
    ("room", "360_v2", -1, "normal"),
    ("stump", "360_v2", -1, "largest"),
    ("flowers", "360_extra_scenes", -1, "largest"),
    ("treehill", "360_extra_scenes", -1, "normal"),
]

excluded_gpus = set()


def _bool_opt(name, value):
    return f" --{name}" if value else f" --no-{name}"


def _opt_path(name, value):
    return f" --{name} \"{value}\"" if value else ""


def _set_env(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = "4"


def _run(cmd, dry_run):
    print(cmd)
    if not dry_run:
        ret = os.system(cmd)
        if ret != 0:
            print(f"[WARN] Command exited with code {ret}")


def get_mv_schedule(branch, mv_profile, gamma_override=None):
    table = {
        "Q": {
            "normal": [48, 24, 12, 8],
            "largest": [1, 1, 1, 3],
            "densify_gamma": 0.50,
            "final_iterations": 30000,
        },
        "T1": {
            "normal": [32, 16, 8, 8],
            "largest": [1, 1, 1, 3],
            "densify_gamma": 0.25,
            "final_iterations": 30000,
        },
        "T2": {
            "normal": [32, 16, 8, 8],
            "largest": [1, 1, 1, 3],
            "densify_gamma": gamma_override if gamma_override is not None else 0.00,
            "final_iterations": 24000,
        },
    }
    entry = table[branch]
    return entry[mv_profile], entry["densify_gamma"], entry["final_iterations"]


def run_view_selection(data_root, scene_name, data_subdir, model_path, strategy, k, args, dry_run):
    source_path = os.path.join(data_root, data_subdir, scene_name)
    cmd = (
        f"python tools/compute_viewset_sfc_frs.py"
        f" --source_path \"{source_path}\""
        f" --model_path \"{model_path}\""
        f" --k {k}"
        f" --strategy {strategy}"
        f" --alpha {args.alpha} --beta {args.beta} --gamma {args.gamma} --delta {args.delta}"
        f" --num_points {args.num_points} --shared_sample {args.shared_sample}"
        f" --seed {args.seed}"
        f" --eval"
        f" --vf_backend {args.vf_backend}"
        f"{_bool_opt('vf_enable_point_score', args.vf_enable_point_score)}"
        f"{_bool_opt('vf_enable_region', args.vf_enable_region)}"
        f"{_bool_opt('vf_enable_obs_cap', args.vf_enable_obs_cap)}"
        f"{_bool_opt('vf_enable_viewdep_guard', args.vf_enable_viewdep_guard)}"
        f"{_bool_opt('vf_enable_shadow', args.vf_enable_shadow)}"
        f"{_bool_opt('vf_enable_shadow_guard', args.vf_enable_shadow_guard)}"
        f"{_bool_opt('vf_enable_split_safe_voxel', args.vf_enable_split_safe_voxel)}"
        f"{_opt_path('vf_mask_path', args.vf_mask_path)}"
        f"{_opt_path('vf_camera_mask_path', args.vf_camera_mask_path)}"
        f"{_opt_path('images_root', args.images_root)}"
        f" --vf_edge_margin_px {args.vf_edge_margin_px}"
    )
    _run(cmd, dry_run)
    return os.path.join(model_path, "selected_views.json")


def _train_stage(base_args, mv, resolution_factor, iterations, ckpt, dry_run):
    ckpt_arg = f" --start_checkpoint \"{ckpt}\"" if ckpt else ""
    iter_arg = f" --iterations {iterations}" if iterations is not None else ""
    cmd = f"{base_args} --mv {mv} -r {resolution_factor}{iter_arg}{ckpt_arg}"
    _run(cmd, dry_run)


def train_scene(gpu, data_root, scene_name, data_subdir, factor, mv_profile, output_dir, strategy, k, args, dry_run):
    _set_env(gpu)
    source_path = os.path.join(data_root, data_subdir, scene_name)
    model_path = os.path.join(output_dir, scene_name)
    mvs, densify_gamma, final_iterations = get_mv_schedule(
        args.vf_schedule_branch,
        mv_profile,
        gamma_override=args.densify_sparse_gamma if args.vf_schedule_branch == "T2" else None,
    )
    port = 6009 + int(gpu)
    view_list_arg = ""
    gamma_arg = ""
    support_pack_arg = ""
    if strategy != "none":
        json_path = run_view_selection(data_root, scene_name, data_subdir, model_path, strategy, k, args, dry_run)
        view_list_arg = f" --train_view_list \"{json_path}\""
        gamma_arg = f" --densify_sparse_gamma {densify_gamma}"
        support_pack_path = os.path.join(model_path, "vf_support_pack.npz")
        if os.path.exists(support_pack_path) or dry_run:
            support_pack_arg = f" --vf_support_pack \"{support_pack_path}\""

    vf_train_common = (
        f"{support_pack_arg}"
        f" --vf_schedule_branch {args.vf_schedule_branch}"
        f"{_bool_opt('vf_enable_prior', args.vf_enable_prior)}"
        f"{_bool_opt('vf_enable_shadow_guard', args.vf_enable_shadow_guard)}"
        f" --prior_type {args.prior_type}"
        f"{_opt_path('vf_mask_path', args.vf_mask_path)}"
        f"{_opt_path('vf_camera_mask_path', args.vf_camera_mask_path)}"
        f" --vf_edge_margin_px {args.vf_edge_margin_px}"
    )
    base_args = (
        f"python train.py"
        f" -s \"{source_path}\" -m \"{model_path}\""
        f" --eval --white_background --data_device cpu"
        f"{view_list_arg}{gamma_arg}{vf_train_common}"
        f" --port {port}"
    )

    ckpt = None
    stage_specs = [
        (mvs[0], 8, 3000),
        (mvs[1], 4, 3000),
        (mvs[2], 2, 3000),
        (mvs[3], factor, final_iterations),
    ]
    for mv, resolution_factor, iterations in stage_specs:
        if mv == 1 and resolution_factor in (8, 4, 2):
            continue
        stage_enable_densify_guard = args.vf_enable_densify_guard and (resolution_factor in (-1, 1, 2))
        stage_args = f"{base_args}{_bool_opt('vf_enable_densify_guard', stage_enable_densify_guard)}"
        _train_stage(stage_args, mv, resolution_factor, iterations, ckpt, dry_run)
        if iterations == 3000:
            ckpt = os.path.join(model_path, f"chkpnt3000_mv{mv}.pth")

    _run(f"python render.py -m \"{model_path}\" --data_device cpu --skip_train", dry_run)
    _run(f"python metrics.py -m \"{model_path}\"", dry_run)
    return True


def worker(gpu, data_root, scene_name, data_subdir, factor, mv_profile, output_dir, strategy, k, args, dry_run):
    print(f"\n{'=' * 60}")
    print(f"[START] GPU {gpu} | {scene_name} | strategy={strategy} K={k} branch={args.vf_schedule_branch}")
    print(f"{'=' * 60}")
    train_scene(gpu, data_root, scene_name, data_subdir, factor, mv_profile, output_dir, strategy, k, args, dry_run)
    print(f"[DONE] GPU {gpu} | {scene_name}")


def dispatch_jobs(jobs, executor, dry_run):
    if dry_run:
        for idx, job in enumerate(jobs):
            worker(idx, *job)
        print("\nAll jobs have been processed.")
        return
    future_to_job = {}
    reserved_gpus = set()
    while jobs or future_to_job:
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)
            future_to_job[future] = (gpu, job)
            reserved_gpus.add(gpu)
        done = [f for f in future_to_job if f.done()]
        for future in done:
            gpu = future_to_job.pop(future)[0]
            reserved_gpus.discard(gpu)
            print(f"[RELEASED] GPU {gpu}")
        time.sleep(5)
    print("\nAll jobs have been processed.")


def main():
    parser = argparse.ArgumentParser(description="Route A batch benchmark for Mip-NeRF 360")
    bool_action = argparse.BooleanOptionalAction
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="benchmark_360v2_ours")
    parser.add_argument(
        "--strategy",
        type=str,
        default="none",
        choices=["none", "random", "uniform_pose", "sfc_frs_greedy", "frs_greedy"],
    )
    parser.add_argument("--k", type=int, default=48)
    parser.add_argument("--densify_sparse_gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--shared_sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenes", type=str, nargs="*", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_workers", type=int, default=8)
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
    parser.add_argument("--vf_enable_densify_guard", action=bool_action, default=False)
    parser.add_argument("--vf_enable_prior", action=bool_action, default=False)
    parser.add_argument("--vf_schedule_branch", type=str, default="Q", choices=["Q", "T1", "T2"])
    parser.add_argument("--prior_type", type=str, default="none", choices=["none", "dav2_rel", "dav2_metric", "other"])
    parser.add_argument("--vf_mask_path", type=str, default="")
    parser.add_argument("--vf_camera_mask_path", type=str, default="")
    parser.add_argument("--vf_edge_margin_px", type=int, default=15)
    parser.add_argument("--images_root", type=str, default="")
    args = parser.parse_args()

    table = [scene for scene in SCENE_TABLE if args.scenes is None or scene[0] in args.scenes]
    if args.scenes:
        missing = set(args.scenes) - {scene[0] for scene in table}
        if missing:
            print(f"[WARN] Unknown scenes ignored: {missing}")

    if args.strategy != "none":
        out = f"{args.output_dir}_{args.strategy}_K{args.k}_{args.vf_schedule_branch}"
    else:
        out = args.output_dir

    jobs = []
    for scene_name, data_subdir, factor, mv_profile in table:
        jobs.append(
            (
                args.data_root,
                scene_name,
                data_subdir,
                factor,
                mv_profile,
                out,
                args.strategy,
                args.k,
                args,
                args.dry_run,
            )
        )

    print(f"[Config] strategy={args.strategy} K={args.k} branch={args.vf_schedule_branch}")
    print(f"[Config] data_root={args.data_root} output={out}")
    print(f"[Config] scenes={[scene[0] for scene in table]}")
    print(f"[Config] jobs={len(jobs)} dry_run={args.dry_run}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        dispatch_jobs(list(jobs), executor, args.dry_run)


if __name__ == "__main__":
    main()
