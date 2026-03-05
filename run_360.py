"""
Route A benchmark script for Mip-NeRF 360 + 360_extra scenes.
Supports: Full training / Sparse-view (SFC-FRS++ / Random / UniformPose) with K control.

Usage:
  # Full training (original MVGS behavior)
  python run_360.py

  # Sparse-view with SFC-FRS++ K=48
  python run_360.py --strategy sfc_frs_greedy --k 48

  # Sparse-view with random baseline K=48
  python run_360.py --strategy random --k 48

  # Dry run (print commands only)
  python run_360.py --strategy sfc_frs_greedy --k 48 --dry_run
"""

import os
import sys
import argparse
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Scene definitions: (scene_name, data_subdir, factor, mv_profile)
# mv_profile: "normal" uses MV_normal, "largest" uses MV_largest
SCENE_TABLE = [
    # 360_v2
    ("bicycle", "360_v2", -1, "largest"),
    ("bonsai", "360_v2", -1, "normal"),
    ("counter", "360_v2", -1, "normal"),
    ("garden", "360_v2", -1, "largest"),
    ("kitchen", "360_v2", -1, "normal"),
    ("room", "360_v2", -1, "normal"),
    ("stump", "360_v2", -1, "largest"),
    # 360_extra_scenes
    ("flowers", "360_extra_scenes", -1, "largest"),
    ("treehill", "360_extra_scenes", -1, "normal"),
]

MV_normal = [48, 24, 12, 8]
MV_largest = [1, 1, 1, 3]

# View selection parameters (SFC-FRS++ defaults, matching §4.4.2)
VS_ALPHA = 1.0
VS_BETA = 0.5
VS_GAMMA = 0.2
VS_DELTA = 0.1
VS_NUM_POINTS = 50000
VS_SHARED_SAMPLE = 200
VS_SEED = 42

excluded_gpus = set()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_env(gpu):
    """Set environment variables for a training run (cross-platform)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = "4"


def _run(cmd, dry_run):
    print(cmd)
    if not dry_run:
        ret = os.system(cmd)
        if ret != 0:
            print(f"[WARN] Command exited with code {ret}")


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

def run_view_selection(data_root, scene_name, data_subdir, model_path,
                       strategy, k, dry_run):
    """Run offline view selection script, return path to selected_views.json."""
    source_path = os.path.join(data_root, data_subdir, scene_name)
    cmd = (
        f"python tools/compute_viewset_sfc_frs.py"
        f" --source_path \"{source_path}\""
        f" --model_path \"{model_path}\""
        f" --k {k}"
        f" --strategy {strategy}"
        f" --alpha {VS_ALPHA} --beta {VS_BETA} --gamma {VS_GAMMA} --delta {VS_DELTA}"
        f" --num_points {VS_NUM_POINTS} --shared_sample {VS_SHARED_SAMPLE}"
        f" --seed {VS_SEED}"
        f" --eval"
    )
    _run(cmd, dry_run)
    return os.path.join(model_path, "selected_views.json")


def train_scene(gpu, data_root, scene_name, data_subdir, factor, mv_profile,
                output_dir, strategy, k, densify_gamma, dry_run):
    """Train a single scene with optional view selection."""
    _set_env(gpu)

    source_path = os.path.join(data_root, data_subdir, scene_name)
    model_path = os.path.join(output_dir, scene_name)

    mvs = MV_largest if mv_profile == "largest" else MV_normal
    port = 6009 + int(gpu)

    # ---- View selection (if not "none") ----
    view_list_arg = ""
    gamma_arg = ""
    if strategy != "none":
        json_path = run_view_selection(
            data_root, scene_name, data_subdir, model_path,
            strategy, k, dry_run)
        view_list_arg = f" --train_view_list \"{json_path}\""
        gamma_arg = f" --densify_sparse_gamma {densify_gamma}"

    # ---- Multi-stage LOD training (same as original MVGS) ----
    base_args = (
        f"python train.py"
        f" -s \"{source_path}\" -m \"{model_path}\""
        f" --eval --white_background --data_device cpu"
        f"{view_list_arg}{gamma_arg}"
        f" --port {port}"
    )

    if mvs[0] != 1:
        cmd = f"{base_args} --mv {mvs[0]} --iterations 3000 -r 8"
        _run(cmd, dry_run)

    if mvs[1] != 1:
        ckpt = os.path.join(model_path, f"chkpnt3000_mv{mvs[0]}.pth")
        cmd = f"{base_args} --mv {mvs[1]} --iterations 3000 -r 4 --start_checkpoint \"{ckpt}\""
        _run(cmd, dry_run)

    if mvs[2] != 1:
        ckpt = os.path.join(model_path, f"chkpnt3000_mv{mvs[1]}.pth")
        cmd = f"{base_args} --mv {mvs[2]} --iterations 3000 -r 2 --start_checkpoint \"{ckpt}\""
        _run(cmd, dry_run)

        ckpt = os.path.join(model_path, f"chkpnt3000_mv{mvs[2]}.pth")
        cmd = f"{base_args} --mv {mvs[3]} -r {factor} --start_checkpoint \"{ckpt}\""
        _run(cmd, dry_run)
    else:
        cmd = f"{base_args} --mv {mvs[3]} -r {factor}"
        _run(cmd, dry_run)

    # ---- Render + Metrics ----
    _run(f"python render.py -m \"{model_path}\" --data_device cpu --skip_train", dry_run)
    _run(f"python metrics.py -m \"{model_path}\"", dry_run)
    return True


def worker(gpu, data_root, scene_name, data_subdir, factor, mv_profile,
           output_dir, strategy, k, densify_gamma, dry_run):
    print(f"\n{'='*60}")
    print(f"[START] GPU {gpu} | {scene_name} | strategy={strategy} K={k}")
    print(f"{'='*60}")
    train_scene(gpu, data_root, scene_name, data_subdir, factor, mv_profile,
                output_dir, strategy, k, densify_gamma, dry_run)
    print(f"[DONE] GPU {gpu} | {scene_name}")


def dispatch_jobs(jobs, executor, dry_run):
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

        done_futures = [f for f in future_to_job if f.done()]
        for future in done_futures:
            gpu_job = future_to_job.pop(future)
            gpu = gpu_job[0]
            reserved_gpus.discard(gpu)
            print(f"[RELEASED] GPU {gpu}")

        time.sleep(5)

    print("\nAll jobs have been processed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Route A batch benchmark for Mip-NeRF 360")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root directory containing 360_v2/ and 360_extra_scenes/ subfolders")
    parser.add_argument("--output_dir", type=str, default="benchmark_360v2_ours",
                        help="Output directory for results")
    parser.add_argument("--strategy", type=str, default="none",
                        choices=["none", "random", "uniform_pose", "sfc_frs_greedy", "frs_greedy"],
                        help="View selection strategy ('none' = Full, original MVGS)")
    parser.add_argument("--k", type=int, default=48,
                        help="Number of training views to select (ignored when strategy=none)")
    parser.add_argument("--densify_sparse_gamma", type=float, default=0.5,
                        help="Densify threshold scaling exponent (ignored when strategy=none)")
    parser.add_argument("--scenes", type=str, nargs="*", default=None,
                        help="Subset of scenes to run (default: all 9). E.g. --scenes bicycle bonsai")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Max parallel GPU workers")
    args = parser.parse_args()

    # Filter scenes
    if args.scenes:
        table = [s for s in SCENE_TABLE if s[0] in args.scenes]
        missing = set(args.scenes) - {s[0] for s in table}
        if missing:
            print(f"[WARN] Unknown scenes ignored: {missing}")
    else:
        table = list(SCENE_TABLE)

    # Adjust output_dir to include strategy/K suffix for easy comparison
    if args.strategy != "none":
        out = f"{args.output_dir}_{args.strategy}_K{args.k}"
    else:
        out = args.output_dir

    # Build job list
    jobs = []
    for scene_name, data_subdir, factor, mv_profile in table:
        jobs.append((
            args.data_root, scene_name, data_subdir, factor, mv_profile,
            out, args.strategy, args.k, args.densify_sparse_gamma, args.dry_run
        ))

    print(f"[Config] strategy={args.strategy}, K={args.k}, gamma={args.densify_sparse_gamma}")
    print(f"[Config] data_root={args.data_root}, output={out}")
    print(f"[Config] scenes={[s[0] for s in table]}")
    print(f"[Config] jobs={len(jobs)}, dry_run={args.dry_run}")
    print()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        dispatch_jobs(list(jobs), executor, args.dry_run)


if __name__ == "__main__":
    main()
