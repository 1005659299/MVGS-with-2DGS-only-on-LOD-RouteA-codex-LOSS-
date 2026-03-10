#!/usr/bin/env python3
"""
MVGS Pre-flight Check
=====================
Run this script **on the training server** before starting training.
It verifies GPU, CUDA, conda, data directories, and CUDA extension
compilation readiness.

Usage
-----
    python preflight_check.py [--data_dir /root/Date] [--code_dir /root/Code]

Exit codes:
    0  – all critical checks passed
    1  – one or more critical checks failed
"""

import argparse
import os
import shutil
import subprocess
import sys

# ── colour helpers ──────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
OK = f"{GREEN}[OK]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── individual checks ──────────────────────────────────────────────
def check_python() -> bool:
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 8):
        print(f"  Python version : {version_str}  {OK}")
        return True
    print(f"  Python version : {version_str}  {FAIL}  (need >= 3.8 for PyTorch 2.x)")
    return False


def check_nvidia_smi() -> bool:
    """Check nvidia-smi is available and parse GPU info."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        print(f"  nvidia-smi     : not found  {FAIL}")
        return False

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            # No-GPU mode or driver issue
            print(f"  nvidia-smi     : returned error  {WARN}")
            print(f"    stderr: {result.stderr.strip()}")
            print(f"    (This is expected in no-GPU / cardless mode.)")
            return False
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            name = parts[0] if len(parts) > 0 else "unknown"
            mem = parts[1] if len(parts) > 1 else "?"
            drv = parts[2] if len(parts) > 2 else "?"
            print(f"  GPU            : {name}  (VRAM {mem} MiB, driver {drv})  {OK}")
        return True
    except Exception as exc:
        print(f"  nvidia-smi     : error – {exc}  {WARN}")
        return False


def check_cuda_version() -> bool:
    """Check CUDA toolkit version via nvcc."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        print(f"  nvcc (CUDA)    : not found  {WARN}")
        print(f"    Hint: ensure CUDA toolkit is on $PATH or conda env is active.")
        return False

    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "release" in line.lower():
                print(f"  CUDA toolkit   : {line.strip()}  {OK}")
                # Parse version
                import re
                m = re.search(r"release\s+(\d+)\.(\d+)", line)
                if m:
                    major, minor = int(m.group(1)), int(m.group(2))
                    if major < 12:
                        print(f"    {WARN} RTX 6000 PRO (Ada Lovelace) needs CUDA >= 12.1")
                        return False
                return True
    except Exception as exc:
        print(f"  nvcc           : error – {exc}  {WARN}")
    return False


def check_torch() -> bool:
    """Check PyTorch and its CUDA support."""
    try:
        import torch  # noqa: F811
    except ImportError:
        print(f"  PyTorch        : not installed  {FAIL}")
        return False

    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda or "none"
    torch_version = torch.__version__

    if cuda_available:
        dev_count = torch.cuda.device_count()
        dev_name = torch.cuda.get_device_name(0) if dev_count > 0 else "N/A"
        print(f"  PyTorch        : {torch_version} (CUDA {cuda_version})  {OK}")
        print(f"  torch.cuda     : {dev_count} device(s) – {dev_name}  {OK}")
        return True

    print(f"  PyTorch        : {torch_version} (CUDA {cuda_version})  {WARN}")
    print(f"  torch.cuda     : NOT available")
    print(f"    (Expected in no-GPU mode. Enable GPU to proceed.)")
    return False


def check_conda_env() -> bool:
    """Check if a conda environment is active."""
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    prefix = os.environ.get("CONDA_PREFIX", "")
    if env:
        print(f"  Conda env      : {env} ({prefix})  {OK}")
        return True
    print(f"  Conda env      : none active  {WARN}")
    return False


def check_directory(path: str, label: str) -> bool:
    """Check a directory exists and list top-level contents."""
    if not os.path.isdir(path):
        print(f"  {label:15s}: {path} NOT FOUND  {FAIL}")
        return False
    items = sorted(os.listdir(path))
    print(f"  {label:15s}: {path}  ({len(items)} items)  {OK}")
    for item in items[:20]:
        full = os.path.join(path, item)
        kind = "dir " if os.path.isdir(full) else "file"
        print(f"    [{kind}] {item}")
    if len(items) > 20:
        print(f"    ... and {len(items) - 20} more")
    return True


def check_submodules(code_dir: str) -> bool:
    """Check that CUDA extension submodules are present."""
    subs = [
        "submodules/diff-gaussian-rasterization",
        "submodules/diff-surfel-rasterization",
        "submodules/simple-knn",
    ]
    all_ok = True
    for sub in subs:
        full = os.path.join(code_dir, sub)
        if os.path.isdir(full) and os.listdir(full):
            print(f"  {os.path.basename(sub):30s}: present  {OK}")
        else:
            print(f"  {os.path.basename(sub):30s}: MISSING or empty  {FAIL}")
            all_ok = False
    return all_ok


def check_cuda_ext_build() -> bool:
    """Check whether the CUDA extensions are importable."""
    exts = {
        "diff_surfel_rasterization": "2D-GS rasterizer",
        "diff_gaussian_rasterization": "3D-GS rasterizer (EWA)",
        "simple_knn": "Simple-KNN",
    }
    all_ok = True
    for module, label in exts.items():
        try:
            __import__(module)
            print(f"  {label:30s}: importable  {OK}")
        except ImportError as e:
            print(f"  {label:30s}: NOT importable  {WARN}")
            print(f"    ({e})")
            all_ok = False
    return all_ok


def check_disk_space(path: str = "/") -> bool:
    """Warn if disk space is low."""
    try:
        stat = os.statvfs(path)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
        pct = free_gb / total_gb * 100 if total_gb else 0
        status = OK if free_gb > 20 else WARN
        print(f"  Disk free      : {free_gb:.1f} GB / {total_gb:.1f} GB ({pct:.0f}%)  {status}")
        return free_gb > 20
    except Exception:
        return True


# ── main ───────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="MVGS training pre-flight check")
    parser.add_argument("--data_dir", default="/root/Date",
                        help="Path to the dataset directory (default: /root/Date)")
    parser.add_argument("--code_dir", default="/root/Code",
                        help="Path to the code directory (default: /root/Code)")
    args = parser.parse_args()

    critical_failures = 0

    # 1. System
    section("1. System Environment")
    if not check_python():
        critical_failures += 1
    check_conda_env()
    check_disk_space()

    # 2. GPU & CUDA
    section("2. GPU & CUDA")
    gpu_ok = check_nvidia_smi()
    check_cuda_version()
    torch_ok = check_torch()
    if not gpu_ok and not torch_ok:
        print(f"\n  {YELLOW}No GPU detected – the machine is likely in no-GPU (cardless) mode.")
        print(f"  You must enable a GPU (e.g. RTX 6000 PRO) before training can start.{RESET}")

    # 3. Directories
    section("3. Data & Code Directories")
    if not check_directory(args.data_dir, "Dataset"):
        critical_failures += 1
    if not check_directory(args.code_dir, "Code"):
        critical_failures += 1

    # 4. Submodules & CUDA extensions
    section("4. CUDA Extensions (Submodules)")
    check_submodules(args.code_dir)
    check_cuda_ext_build()

    # 5. Summary
    section("5. Summary & Recommendations")
    if critical_failures == 0:
        if gpu_ok:
            print(f"  {GREEN}All critical checks passed. Training is ready to start!{RESET}")
            print(f"  Recommended command:")
            print(f"    cd {args.code_dir}")
            print(f"    python train.py -s {args.data_dir} --eval -m output/run1 --mv 6")
        else:
            print(f"  {YELLOW}Environment looks good, but no GPU is available yet.")
            print(f"  Enable RTX 6000 PRO, then re-run this check.{RESET}")
    else:
        print(f"  {RED}{critical_failures} critical check(s) failed. See details above.{RESET}")

    print()
    return 1 if critical_failures else 0


if __name__ == "__main__":
    sys.exit(main())
