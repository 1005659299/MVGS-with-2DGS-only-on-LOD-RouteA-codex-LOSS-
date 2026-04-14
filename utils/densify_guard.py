from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from utils.voxel_support import VoxelSupportPack, load_vf_support_pack


CLASS_REJECT = 0
CLASS_SAT2 = 1
CLASS_WEAK2 = 2
CLASS_HARD3 = 3

CLASS_NAMES = {
    CLASS_REJECT: "reject",
    CLASS_SAT2: "sat2",
    CLASS_WEAK2: "weak2",
    CLASS_HARD3: "hard3",
}


@dataclass
class DensifyGuard:
    class_id: torch.Tensor
    clone_allow: torch.BoolTensor
    split_allow: torch.BoolTensor
    tau_clone: torch.Tensor
    tau_split: torch.Tensor
    box_bonus_mask: torch.BoolTensor | None
    big_screen_force_split: torch.BoolTensor | None
    metadata: dict


def _to_numpy_xyz(gaussians_xyz) -> np.ndarray:
    if isinstance(gaussians_xyz, torch.Tensor):
        return gaussians_xyz.detach().cpu().numpy()
    return np.asarray(gaussians_xyz)


def rehash_gaussians_to_voxels(gaussians_xyz, support_pack: VoxelSupportPack) -> np.ndarray:
    xyz = _to_numpy_xyz(gaussians_xyz).astype(np.float64)
    if xyz.size == 0 or support_pack.voxel_origin.size == 0:
        return np.zeros(0, dtype=np.int32)
    voxel_min = support_pack.voxel_origin.astype(np.float64)
    voxel_max = voxel_min + support_pack.voxel_size[:, None].astype(np.float64)
    centers = support_pack.voxel_center.astype(np.float64)
    assignments = np.full(xyz.shape[0], -1, dtype=np.int32)
    for idx, point in enumerate(xyz):
        inside = np.where(np.all((point >= voxel_min) & (point <= voxel_max), axis=1))[0]
        if inside.size == 1:
            assignments[idx] = int(inside[0])
        elif inside.size > 1:
            sizes = support_pack.voxel_size[inside]
            assignments[idx] = int(inside[np.argmin(sizes)])
        else:
            assignments[idx] = int(np.argmin(np.sum((centers - point[None, :]) ** 2, axis=1)))
    return assignments


def compute_current_voxel_support_ok_ratio(
    gaussian2voxel: np.ndarray,
    support_pack: VoxelSupportPack,
) -> np.ndarray:
    n_voxels = int(support_pack.n_pts.shape[0])
    if gaussian2voxel.size == 0 or n_voxels == 0:
        return np.zeros(n_voxels, dtype=np.float32)
    counts = np.bincount(np.clip(gaussian2voxel, 0, n_voxels - 1), minlength=n_voxels)
    density_ratio = counts / np.maximum(support_pack.n_pts, 1)
    if support_pack.final_voxel_support_ok_ratio.size == n_voxels:
        density_ratio = np.maximum(density_ratio, support_pack.final_voxel_support_ok_ratio)
    return np.clip(density_ratio, 0.0, 1.0).astype(np.float32)


def classify_gaussians(
    support_pack: VoxelSupportPack,
    gaussian2voxel: np.ndarray,
    voxel_support_ok_ratio: np.ndarray,
) -> np.ndarray:
    class_id = np.full(gaussian2voxel.shape[0], CLASS_REJECT, dtype=np.int32)
    if gaussian2voxel.size == 0:
        return class_id
    voxel_idx = np.clip(gaussian2voxel, 0, max(len(support_pack.target_hits) - 1, 0))
    reject = support_pack.viewdep_sink[voxel_idx] | (support_pack.obs_cap[voxel_idx] < 2)
    sat2 = (
        (support_pack.target_hits[voxel_idx] == 2)
        & (voxel_support_ok_ratio[voxel_idx] >= 0.50)
    )
    weak2 = (
        (support_pack.target_hits[voxel_idx] == 2)
        & (voxel_support_ok_ratio[voxel_idx] < 0.50)
        & (~support_pack.viewdep_sink[voxel_idx])
    )
    hard3 = support_pack.target_hits[voxel_idx] >= 3
    class_id[sat2] = CLASS_SAT2
    class_id[weak2] = CLASS_WEAK2
    class_id[hard3] = CLASS_HARD3
    class_id[reject] = CLASS_REJECT
    return class_id


def build_densify_guard(
    *,
    gaussians_xyz,
    support_pack: VoxelSupportPack,
    voxel_support_ok_ratio: np.ndarray,
    hotspot_boxes=None,
    densify_t: float,
    size_threshold=None,
    max_radii2D=None,
    lod_resolution_factor: int = 1,
    gaussian2voxel: np.ndarray | None = None,
) -> DensifyGuard:
    xyz_np = _to_numpy_xyz(gaussians_xyz)
    n_gauss = int(xyz_np.shape[0])
    device = gaussians_xyz.device if isinstance(gaussians_xyz, torch.Tensor) else torch.device("cpu")
    if gaussian2voxel is None:
        gaussian2voxel = rehash_gaussians_to_voxels(xyz_np, support_pack)
    if lod_resolution_factor >= 4:
        class_id = torch.full((n_gauss,), CLASS_HARD3, dtype=torch.int32, device=device)
        ones = torch.ones((n_gauss,), dtype=torch.bool, device=device)
        tau = torch.full((n_gauss,), float(densify_t), dtype=torch.float32, device=device)
        return DensifyGuard(
            class_id=class_id,
            clone_allow=ones,
            split_allow=ones,
            tau_clone=tau,
            tau_split=tau,
            box_bonus_mask=None,
            big_screen_force_split=None,
            metadata={"lod_bypass": True, "gaussian2voxel": gaussian2voxel},
        )
    class_id_np = classify_gaussians(support_pack, gaussian2voxel, voxel_support_ok_ratio)
    clone_mul = np.array([np.inf, np.inf, 1.50, 1.00], dtype=np.float32)
    split_mul = np.array([np.inf, 1.75, 1.25, 1.00], dtype=np.float32)
    clone_allow_np = np.isin(class_id_np, [CLASS_WEAK2, CLASS_HARD3])
    split_allow_np = np.isin(class_id_np, [CLASS_WEAK2, CLASS_HARD3, CLASS_SAT2])
    tau_clone_np = np.full(n_gauss, np.inf, dtype=np.float32)
    tau_split_np = np.full(n_gauss, np.inf, dtype=np.float32)
    tau_clone_np[clone_allow_np] = densify_t * clone_mul[class_id_np[clone_allow_np]]
    tau_split_np[split_allow_np] = densify_t * split_mul[class_id_np[split_allow_np]]
    big_screen_force_split = np.zeros(n_gauss, dtype=bool)
    if size_threshold is not None and max_radii2D is not None:
        radii_np = _to_numpy_xyz(max_radii2D).reshape(-1)
        big_screen_force_split = (class_id_np == CLASS_SAT2) & (radii_np > float(size_threshold))
    return DensifyGuard(
        class_id=torch.as_tensor(class_id_np, dtype=torch.int32, device=device),
        clone_allow=torch.as_tensor(clone_allow_np, dtype=torch.bool, device=device),
        split_allow=torch.as_tensor(split_allow_np, dtype=torch.bool, device=device),
        tau_clone=torch.as_tensor(tau_clone_np, dtype=torch.float32, device=device),
        tau_split=torch.as_tensor(tau_split_np, dtype=torch.float32, device=device),
        box_bonus_mask=None,
        big_screen_force_split=torch.as_tensor(big_screen_force_split, dtype=torch.bool, device=device),
        metadata={
            "lod_bypass": False,
            "gaussian2voxel": gaussian2voxel,
            "voxel_support_ok_ratio": voxel_support_ok_ratio,
            "hotspot_boxes": hotspot_boxes or [],
        },
    )


def summarize_densify_stats(
    *,
    guard: DensifyGuard | None,
    clone_mask: torch.Tensor | None = None,
    split_mask: torch.Tensor | None = None,
    scaling: torch.Tensor | None = None,
    mv_view_count: int = 1,
) -> dict:
    stats = {
        "densify_births_by_support_class": {name: 0 for name in CLASS_NAMES.values()},
        "densify_reject_ratio_by_support_class": {name: 0.0 for name in CLASS_NAMES.values()},
        "spiky_gaussian_ratio": 0.0,
        "mv_densify_stat_view_count_mean": float(mv_view_count),
    }
    if guard is None:
        return stats
    class_id = guard.class_id.detach().cpu().numpy()
    clone_np = clone_mask.detach().cpu().numpy() if clone_mask is not None else np.zeros_like(class_id, dtype=bool)
    split_np = split_mask.detach().cpu().numpy() if split_mask is not None else np.zeros_like(class_id, dtype=bool)
    for cid, name in CLASS_NAMES.items():
        members = class_id == cid
        if not np.any(members):
            continue
        births = int(np.count_nonzero(clone_np[members]) + 2 * np.count_nonzero(split_np[members]))
        stats["densify_births_by_support_class"][name] = births
        rejected = np.count_nonzero(~guard.clone_allow.detach().cpu().numpy()[members])
        stats["densify_reject_ratio_by_support_class"][name] = float(rejected / max(np.count_nonzero(members), 1))
    if scaling is not None and scaling.numel():
        s = scaling.detach().cpu().numpy()
        if s.ndim == 2 and s.shape[1] >= 2:
            ratio = np.max(s, axis=1) / np.maximum(np.min(s, axis=1), 1e-6)
            stats["spiky_gaussian_ratio"] = float(np.mean(ratio > 5.0))
    return stats