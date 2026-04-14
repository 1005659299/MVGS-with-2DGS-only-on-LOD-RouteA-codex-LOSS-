from __future__ import annotations

import json
import os
import struct
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np

from utils.view_selection_backend import ExactCSRSelectorBackend, VFWeights
from utils.voxel_support import (
    VoxelSupportPack,
    build_voxel_support,
    finalize_voxel_support_pack,
    save_support_pack_npz,
)
from utils.zero_point_shadow import build_shadow_carriers, save_shadow_diagnostics


@dataclass
class VFConfig:
    backend: str = "exact_csr_numpy"
    enable_point_score: bool = False
    enable_region: bool = False
    enable_obs_cap: bool = False
    enable_viewdep_guard: bool = False
    enable_shadow: bool = False
    enable_shadow_guard: bool = False
    enable_split_safe_voxel: bool = False
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.2
    delta: float = 0.1
    second_hit_weight: float = 0.75
    region_balance_weight: float = 0.25
    region_redundancy_weight: float = 0.10
    tau_pair: float = 0.15
    tau_ok: float = 0.50
    tau_easy: float = 0.35
    tau_hard: float = 0.60
    tau_ok_hard: float = 0.70
    coarse_grid_size: int = 256
    n_split_max: int = 64
    eta_size: float = 1.5
    tau_z_ratio: float = 1.8
    tau_z_abs: float = 0.15
    max_leaf_depth: int = 2
    n_leaf_min: int = 8
    shadow_tau_sp: float = 6.0
    edge_margin_px: int = 15


@dataclass
class ViewSelectionResult:
    selected_image_ids: list
    selected_names: list
    meta: dict
    diagnostics: dict
    support_pack: VoxelSupportPack
    shadow_pack: dict | None = None


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def save_json(path: str, payload: dict | list) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
    return path


def read_points3D_binary_full(path):
    def _read(fid, num_bytes, fmt, endian="<"):
        return struct.unpack(endian + fmt, fid.read(num_bytes))

    pts = {}
    with open(path, "rb") as fid:
        num_points = _read(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read(fid, 43, "QdddBBBd")
            point3D_id = int(props[0])
            xyz = np.asarray(props[1:4], dtype=np.float32)
            error = float(props[7])
            track_length = int(_read(fid, 8, "Q")[0])
            _read(fid, 8 * track_length, "ii" * track_length)
            pts[point3D_id] = {
                "xyz": xyz,
                "reproj_err": error,
                "track_len": track_length,
            }
    return pts


def load_points3D_full(sparse_dir):
    return read_points3D_binary_full(os.path.join(sparse_dir, "points3D.bin"))


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def compute_camera_centers(cam_extrinsics):
    centers = {}
    for img_id, img in cam_extrinsics.items():
        R = qvec2rotmat(img.qvec)
        centers[img_id] = -R.T @ img.tvec
    return centers


def _selected_point_ids(points3D_full, num_points_cap, rng):
    all_pids = sorted(points3D_full.keys())
    if len(all_pids) <= num_points_cap:
        return all_pids
    picked = rng.choice(np.asarray(all_pids, dtype=np.int64), num_points_cap, replace=False)
    return sorted(int(v) for v in picked.tolist())


def build_point_records(cam_extrinsics, points3D_full, num_points_cap, rng):
    chosen_sorted = _selected_point_ids(points3D_full, num_points_cap, rng)
    pid2idx = {pid: idx for idx, pid in enumerate(chosen_sorted)}
    pt_xyz = np.asarray([points3D_full[pid]["xyz"] for pid in chosen_sorted], dtype=np.float32)
    pt_reproj_err = np.asarray([points3D_full[pid]["reproj_err"] for pid in chosen_sorted], dtype=np.float32)
    pt_track_len = np.asarray([points3D_full[pid]["track_len"] for pid in chosen_sorted], dtype=np.int32)
    image_ids = sorted(cam_extrinsics.keys())
    candidate_view_names = [
        os.path.splitext(os.path.basename(cam_extrinsics[iid].name))[0] for iid in image_ids
    ]
    vis_sets = []
    depths_list = []
    vis_pts_ptr = [0]
    vis_pts_idx = []
    vis_z_ptr = [0]
    vis_z_val = []
    point_depth_lists = [[] for _ in range(len(chosen_sorted))]
    view_sparse_xy = []
    view_sparse_depth = []
    view_sparse_track_len = []
    view_sparse_err = []
    for iid in image_ids:
        img = cam_extrinsics[iid]
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        vis = set()
        dep = {}
        local_pts = []
        local_depth = []
        local_xy = []
        local_track = []
        local_err = []
        for obs_idx, pid_raw in enumerate(img.point3D_ids):
            pid = int(pid_raw)
            if pid < 0 or pid not in pid2idx:
                continue
            pt_idx = pid2idx[pid]
            z = float(R[2, :] @ pt_xyz[pt_idx] + t[2])
            if z <= 0:
                continue
            vis.add(pt_idx)
            dep[pt_idx] = z
            local_pts.append(pt_idx)
            local_depth.append(z)
            local_xy.append(np.asarray(img.xys[obs_idx], dtype=np.float32))
            local_track.append(pt_track_len[pt_idx])
            local_err.append(pt_reproj_err[pt_idx])
            point_depth_lists[pt_idx].append(z)
        vis_sets.append(vis)
        depths_list.append(dep)
        vis_pts_idx.extend(local_pts)
        vis_pts_ptr.append(len(vis_pts_idx))
        vis_z_val.extend(local_depth)
        vis_z_ptr.append(len(vis_z_val))
        view_sparse_xy.append(np.asarray(local_xy, dtype=np.float32))
        view_sparse_depth.append(np.asarray(local_depth, dtype=np.float32))
        view_sparse_track_len.append(np.asarray(local_track, dtype=np.int32))
        view_sparse_err.append(np.asarray(local_err, dtype=np.float32))
    return {
        "image_ids": image_ids,
        "candidate_view_names": candidate_view_names,
        "pt_ids": np.asarray(chosen_sorted, dtype=np.int64),
        "pt_xyz": pt_xyz,
        "pt_reproj_err": pt_reproj_err,
        "pt_track_len": pt_track_len,
        "pid2idx": pid2idx,
        "vis_sets": vis_sets,
        "depths_list": depths_list,
        "vis_pts_ptr": np.asarray(vis_pts_ptr, dtype=np.int64),
        "vis_pts_idx": np.asarray(vis_pts_idx, dtype=np.int32),
        "vis_z_ptr": np.asarray(vis_z_ptr, dtype=np.int64),
        "vis_z_val": np.asarray(vis_z_val, dtype=np.float32),
        "point_depth_lists": [np.asarray(v, dtype=np.float32) for v in point_depth_lists],
        "view_sparse_xy": view_sparse_xy,
        "view_sparse_depth": view_sparse_depth,
        "view_sparse_track_len": view_sparse_track_len,
        "view_sparse_err": view_sparse_err,
    }


def precompute_matrices(image_ids, cam_extrinsics, vis_sets, depths_list, pt_xyz, shared_sample, rng):
    N = len(image_ids)
    eps = 1e-6
    info = np.zeros(N, dtype=np.float64)
    for view_idx in range(N):
        for depth in depths_list[view_idx].values():
            info[view_idx] += 1.0 / (depth * depth + eps)
    overlap = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            inter = len(vis_sets[i] & vis_sets[j])
            union = len(vis_sets[i] | vis_sets[j])
            if union > 0:
                overlap[i, j] = overlap[j, i] = inter / union
    centers = compute_camera_centers(cam_extrinsics)
    centers = np.asarray([centers[iid] for iid in image_ids], dtype=np.float64)
    baseline = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            shared = list(vis_sets[i] & vis_sets[j])
            if not shared:
                continue
            if len(shared) > shared_sample:
                shared = rng.choice(np.asarray(shared, dtype=np.int32), shared_sample, replace=False).tolist()
            sin2_vals = []
            for pt_idx in shared:
                point = pt_xyz[pt_idx]
                d0 = point - centers[i]
                d1 = point - centers[j]
                denom = np.linalg.norm(d0) * np.linalg.norm(d1) + eps
                cos_phi = np.clip(np.dot(d0, d1) / denom, -1.0, 1.0)
                sin2_vals.append(1.0 - cos_phi ** 2)
            baseline[i, j] = baseline[j, i] = float(np.mean(sin2_vals))
    return info, overlap, baseline


def _legacy_greedy_select(K, N, vis_sets, info, overlap, baseline, alpha, beta, gamma, delta, num_points_total):
    covered = set()
    selected = []
    info_max = info.max() + 1e-12
    for _ in range(K):
        best_idx = -1
        best_score = -np.inf
        for view_idx in range(N):
            if view_idx in selected:
                continue
            cov_gain = len(vis_sets[view_idx] - covered) / max(num_points_total, 1)
            base_gain = max((baseline[view_idx, s] for s in selected), default=0.0)
            info_val = info[view_idx] / info_max
            ov_pen = max((overlap[view_idx, s] for s in selected), default=0.0)
            score = alpha * cov_gain + beta * base_gain + gamma * info_val - delta * ov_pen
            if score > best_score:
                best_score = score
                best_idx = view_idx
        selected.append(best_idx)
        covered |= vis_sets[best_idx]
    return selected


def select_views_uniform_pose(cam_extrinsics, image_ids, K, rng):
    centers = compute_camera_centers(cam_extrinsics)
    pts = np.asarray([centers[iid] for iid in image_ids], dtype=np.float64)
    N = len(pts)
    assert K < N, f"[ViewSelect] K={K} >= N={N} in uniform_pose selection"
    selected = [int(rng.randint(N))]
    dists = np.full(N, np.inf, dtype=np.float64)
    for _ in range(K - 1):
        last = pts[selected[-1]]
        for idx in range(N):
            d = np.linalg.norm(pts[idx] - last)
            dists[idx] = min(dists[idx], d)
        for chosen in selected:
            dists[chosen] = -1.0
        selected.append(int(np.argmax(dists)))
    return selected


def _point_diagnostics(state, tau_pair: float) -> dict:
    hits = state.hit_count
    support_ok = (hits >= 2) & (state.best_pair_quality >= tau_pair)
    pair_vals = state.best_pair_quality[hits >= 2]
    return {
        "enabled": True,
        "coverage_at_1": float(np.mean(hits >= 1)) if hits.size else 0.0,
        "coverage_at_2": float(np.mean(hits >= 2)) if hits.size else 0.0,
        "coverage_at_3": float(np.mean(hits >= 3)) if hits.size else 0.0,
        "under2_ratio": float(np.mean(hits < 2)) if hits.size else 0.0,
        "median_best_pair_quality_at_2": float(np.median(pair_vals)) if pair_vals.size else 0.0,
        "support_ok_ratio": float(np.mean(support_ok)) if support_ok.size else 0.0,
    }


def _histogram(values: np.ndarray, bins: list[int]) -> dict:
    hist = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        hist[f"[{lo},{hi})"] = int(np.count_nonzero((values >= lo) & (values < hi)))
    hist[f"[{bins[-1]},inf)"] = int(np.count_nonzero(values >= bins[-1]))
    return hist


def _voxel_diagnostics(pack: VoxelSupportPack) -> dict:
    target = pack.target_hits
    current_hits = pack.final_voxel_hit_views if pack.final_voxel_hit_views.size else np.zeros_like(target)
    under_target = (pack.obs_cap >= target) & (current_hits < target)
    return {
        "enabled": True,
        "voxel_support_ok_ratio_histogram": {
            "[0.0,0.25)": int(np.count_nonzero((pack.final_voxel_support_ok_ratio >= 0.0) & (pack.final_voxel_support_ok_ratio < 0.25))),
            "[0.25,0.50)": int(np.count_nonzero((pack.final_voxel_support_ok_ratio >= 0.25) & (pack.final_voxel_support_ok_ratio < 0.50))),
            "[0.50,0.75)": int(np.count_nonzero((pack.final_voxel_support_ok_ratio >= 0.50) & (pack.final_voxel_support_ok_ratio < 0.75))),
            "[0.75,1.00]": int(np.count_nonzero(pack.final_voxel_support_ok_ratio >= 0.75)),
        },
        "region_under_target_ratio": float(np.mean(under_target)) if under_target.size else 0.0,
        "cross_layer_merge_ratio": float(pack.cross_layer_merge_ratio),
        "leaf_split_ratio": float(pack.leaf_split_ratio),
        "viewdep_sink_ratio": float(np.mean(pack.viewdep_sink)) if pack.viewdep_sink.size else 0.0,
        "recoverable_hard_ratio": float(np.mean(pack.recoverable_hard)) if pack.recoverable_hard.size else 0.0,
    }


def _obs_diagnostics(pack: VoxelSupportPack) -> dict:
    intrinsic_low = pack.obs_cap < 2
    return {
        "enabled": True,
        "intrinsic_low_observability_ratio": float(np.mean(intrinsic_low)) if intrinsic_low.size else 0.0,
        "obs_cap_histogram": _histogram(pack.obs_cap, [0, 1, 2, 3]),
    }


def _backend_weights(vf_cfg: VFConfig, overlap_pen: float | None = None) -> VFWeights:
    return VFWeights(
        first_hit=vf_cfg.alpha,
        second_hit=vf_cfg.second_hit_weight if vf_cfg.enable_point_score else 0.0,
        region_balance=vf_cfg.region_balance_weight if (vf_cfg.enable_region and vf_cfg.enable_obs_cap) else 0.0,
        region_redundancy=vf_cfg.region_redundancy_weight if (vf_cfg.enable_region and vf_cfg.enable_obs_cap) else 0.0,
        base_gain=vf_cfg.beta,
        info_gain=vf_cfg.gamma,
        overlap_pen=vf_cfg.delta if overlap_pen is None else overlap_pen,
    )


def _run_vf_selector(*, image_ids, candidate_view_names, backend: ExactCSRSelectorBackend, K: int, vf_cfg: VFConfig):
    _ = vf_cfg
    tracemalloc.start()
    t0 = time.time()
    state = backend.init_state()
    selected = []
    for _ in range(K):
        cand_scores = backend.score_candidates_delta(state)
        best_idx = int(np.argmax(cand_scores))
        state = backend.apply_selection_delta(state, best_idx)
        selected.append(best_idx)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    _ = current_bytes
    tracemalloc.stop()
    elapsed = time.time() - t0
    selected_ids = [image_ids[idx] for idx in selected]
    selected_names = [candidate_view_names[idx] for idx in selected]
    return selected, selected_ids, selected_names, state, elapsed, peak_bytes / (1024.0 * 1024.0)


def select_views(
    cam_extrinsics,
    points3D_full,
    K,
    strategy,
    vf_cfg,
    *,
    cam_intrinsics=None,
    image_size_by_name=None,
    images_root="",
    mask_root="",
    camera_mask_root="",
    num_points=50000,
    shared_sample=200,
    seed=42,
):
    rng = np.random.RandomState(seed)
    point_pack = build_point_records(cam_extrinsics, points3D_full, num_points, rng)
    image_ids = point_pack["image_ids"]
    candidate_view_names = point_pack["candidate_view_names"]
    N = len(image_ids)
    assert K < N, f"[ViewSelect] FATAL: K={K} >= N={N} train candidates."
    info, overlap, baseline = precompute_matrices(
        image_ids,
        cam_extrinsics,
        point_pack["vis_sets"],
        point_pack["depths_list"],
        point_pack["pt_xyz"],
        shared_sample,
        rng,
    )
    voxel_pack = build_voxel_support(point_pack, vf_cfg)
    overlap_pen = 0.0 if strategy == "frs_greedy" else vf_cfg.delta
    if strategy == "random":
        selected = sorted(rng.choice(N, K, replace=False).tolist())
        selected_ids = [image_ids[idx] for idx in selected]
        selected_names = [candidate_view_names[idx] for idx in selected]
        state = ExactCSRSelectorBackend(
            vis_pts_ptr=point_pack["vis_pts_ptr"],
            vis_pts_idx=point_pack["vis_pts_idx"],
            vis_vox_ptr=voxel_pack.view_vox_ptr,
            vis_vox_idx=voxel_pack.view_vox_idx,
            info=info,
            overlap=overlap,
            baseline=baseline,
            voxel_pack=voxel_pack,
            weights=_backend_weights(vf_cfg, overlap_pen=overlap_pen),
            tau_pair=vf_cfg.tau_pair,
            tau_ok=vf_cfg.tau_ok,
            tau_easy=vf_cfg.tau_easy,
            tau_hard=vf_cfg.tau_hard,
            tau_ok_hard=vf_cfg.tau_ok_hard,
            enable_region=vf_cfg.enable_region,
            enable_obs_cap=vf_cfg.enable_obs_cap,
            enable_viewdep_guard=vf_cfg.enable_viewdep_guard,
        ).init_state()
        for idx in selected:
            state.selected_mask[idx] = True
        elapsed = 0.0
        peak_mem_mb = 0.0
    elif strategy == "uniform_pose":
        selected = select_views_uniform_pose(cam_extrinsics, image_ids, K, rng)
        selected_ids = [image_ids[idx] for idx in selected]
        selected_names = [candidate_view_names[idx] for idx in selected]
        state = ExactCSRSelectorBackend(
            vis_pts_ptr=point_pack["vis_pts_ptr"],
            vis_pts_idx=point_pack["vis_pts_idx"],
            vis_vox_ptr=voxel_pack.view_vox_ptr,
            vis_vox_idx=voxel_pack.view_vox_idx,
            info=info,
            overlap=overlap,
            baseline=baseline,
            voxel_pack=voxel_pack,
            weights=_backend_weights(vf_cfg, overlap_pen=overlap_pen),
            tau_pair=vf_cfg.tau_pair,
            tau_ok=vf_cfg.tau_ok,
            tau_easy=vf_cfg.tau_easy,
            tau_hard=vf_cfg.tau_hard,
            tau_ok_hard=vf_cfg.tau_ok_hard,
            enable_region=vf_cfg.enable_region,
            enable_obs_cap=vf_cfg.enable_obs_cap,
            enable_viewdep_guard=vf_cfg.enable_viewdep_guard,
        ).init_state()
        for idx in selected:
            state.selected_mask[idx] = True
        elapsed = 0.0
        peak_mem_mb = 0.0
    elif strategy in ("sfc_frs_greedy", "frs_greedy"):
        backend = ExactCSRSelectorBackend(
            vis_pts_ptr=point_pack["vis_pts_ptr"],
            vis_pts_idx=point_pack["vis_pts_idx"],
            vis_vox_ptr=voxel_pack.view_vox_ptr,
            vis_vox_idx=voxel_pack.view_vox_idx,
            info=info,
            overlap=overlap,
            baseline=baseline,
            voxel_pack=voxel_pack,
            weights=_backend_weights(vf_cfg, overlap_pen=overlap_pen),
            tau_pair=vf_cfg.tau_pair,
            tau_ok=vf_cfg.tau_ok,
            tau_easy=vf_cfg.tau_easy,
            tau_hard=vf_cfg.tau_hard,
            tau_ok_hard=vf_cfg.tau_ok_hard,
            enable_region=vf_cfg.enable_region,
            enable_obs_cap=vf_cfg.enable_obs_cap,
            enable_viewdep_guard=vf_cfg.enable_viewdep_guard,
        )
        selected, selected_ids, selected_names, state, elapsed, peak_mem_mb = _run_vf_selector(
            image_ids=image_ids,
            candidate_view_names=candidate_view_names,
            backend=backend,
            K=K,
            vf_cfg=vf_cfg,
        )
        parity_selected = _legacy_greedy_select(
            K,
            N,
            point_pack["vis_sets"],
            info,
            overlap,
            baseline,
            vf_cfg.alpha,
            vf_cfg.beta,
            vf_cfg.gamma,
            vf_cfg.delta if strategy == "sfc_frs_greedy" else 0.0,
            len(point_pack["pt_xyz"]),
        )
        parity_ok = (
            (not vf_cfg.enable_point_score)
            and (not vf_cfg.enable_region)
            and (not vf_cfg.enable_obs_cap)
            and (not vf_cfg.enable_viewdep_guard)
            and parity_selected == selected
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    if strategy in ("random", "uniform_pose"):
        backend = ExactCSRSelectorBackend(
            vis_pts_ptr=point_pack["vis_pts_ptr"],
            vis_pts_idx=point_pack["vis_pts_idx"],
            vis_vox_ptr=voxel_pack.view_vox_ptr,
            vis_vox_idx=voxel_pack.view_vox_idx,
            info=info,
            overlap=overlap,
            baseline=baseline,
            voxel_pack=voxel_pack,
            weights=_backend_weights(vf_cfg, overlap_pen=overlap_pen),
            tau_pair=vf_cfg.tau_pair,
            tau_ok=vf_cfg.tau_ok,
            tau_easy=vf_cfg.tau_easy,
            tau_hard=vf_cfg.tau_hard,
            tau_ok_hard=vf_cfg.tau_ok_hard,
            enable_region=vf_cfg.enable_region,
            enable_obs_cap=vf_cfg.enable_obs_cap,
            enable_viewdep_guard=vf_cfg.enable_viewdep_guard,
        )
        state = backend.init_state()
        for idx in selected:
            state = backend.apply_selection_delta(state, idx)
        parity_ok = None
    voxel_support_ok_ratio = state.voxel_support_ok_count / np.maximum(voxel_pack.n_pts, 1)
    target_hits = ExactCSRSelectorBackend(
        vis_pts_ptr=point_pack["vis_pts_ptr"],
        vis_pts_idx=point_pack["vis_pts_idx"],
        vis_vox_ptr=voxel_pack.view_vox_ptr,
        vis_vox_idx=voxel_pack.view_vox_idx,
        info=info,
        overlap=overlap,
        baseline=baseline,
        voxel_pack=voxel_pack,
        weights=_backend_weights(vf_cfg, overlap_pen=overlap_pen),
        tau_pair=vf_cfg.tau_pair,
        tau_ok=vf_cfg.tau_ok,
        tau_easy=vf_cfg.tau_easy,
        tau_hard=vf_cfg.tau_hard,
        tau_ok_hard=vf_cfg.tau_ok_hard,
        enable_region=vf_cfg.enable_region,
        enable_obs_cap=vf_cfg.enable_obs_cap,
        enable_viewdep_guard=vf_cfg.enable_viewdep_guard,
    ).current_target_hits(state)
    voxel_pack = finalize_voxel_support_pack(
        voxel_pack,
        selected_view_names=selected_names,
        voxel_hit_views=state.voxel_hit_views,
        voxel_support_ok_ratio=voxel_support_ok_ratio,
        target_hits=target_hits,
    )
    point_diag = _point_diagnostics(state, vf_cfg.tau_pair)
    voxel_diag = _voxel_diagnostics(voxel_pack)
    obs_diag = _obs_diagnostics(voxel_pack)
    shadow_pack = None
    if vf_cfg.enable_shadow and cam_intrinsics is not None and image_size_by_name is not None:
        shadow_pack = build_shadow_carriers(
            point_pack=point_pack,
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            image_size_by_name=image_size_by_name,
            images_root=images_root,
            mask_root=mask_root,
            camera_mask_root=camera_mask_root,
            vf_cfg=vf_cfg,
        )
        voxel_pack.shadow_ids = shadow_pack["shadow_ids"]
        voxel_pack.shadow_view_names = shadow_pack["shadow_view_names"]
        voxel_pack.shadow_bboxes = shadow_pack["shadow_bboxes"]
        voxel_pack.shadow_obs_cap = shadow_pack["shadow_obs_cap"]
        voxel_pack.shadow_hit_views = shadow_pack["shadow_hit_views"]
        voxel_pack.shadow_anchor_conf = shadow_pack["shadow_anchor_conf"]
        voxel_pack.shadow_edge_conf = shadow_pack["shadow_edge_conf"]
        voxel_pack.shadow_ring_anchorable = shadow_pack["shadow_ring_anchorable"]
        voxel_pack.shadow_local_depth_cont = shadow_pack["shadow_local_depth_cont"]
        voxel_pack.shadow_z_nb = shadow_pack["shadow_z_nb"]
        voxel_pack.shadow_trusted_prior = shadow_pack["shadow_trusted_prior"]
    diagnostics = {
        "point": point_diag,
        "voxel": voxel_diag,
        "obs": obs_diag,
        "shadow": shadow_pack["diagnostics"] if shadow_pack is not None else {
            "enabled": False,
            "ghost_anchor_reject_ratio": None,
            "shadow_anchor_conf_hist": None,
            "shadow_edge_conf_hist": None,
            "shadow_hotspot_overlap": None,
        },
    }
    meta = {
        "strategy": strategy,
        "alpha": vf_cfg.alpha,
        "beta": vf_cfg.beta,
        "gamma": vf_cfg.gamma,
        "delta": vf_cfg.delta if strategy == "sfc_frs_greedy" else 0.0,
        "vf_backend": vf_cfg.backend,
        "vf_enable_point_score": vf_cfg.enable_point_score,
        "vf_enable_region": vf_cfg.enable_region,
        "vf_enable_obs_cap": vf_cfg.enable_obs_cap,
        "vf_enable_viewdep_guard": vf_cfg.enable_viewdep_guard,
        "vf_enable_shadow": vf_cfg.enable_shadow,
        "vf_enable_shadow_guard": vf_cfg.enable_shadow_guard,
        "vf_enable_split_safe_voxel": vf_cfg.enable_split_safe_voxel,
        "N": N,
        "K": K,
        "selector_elapsed_seconds": round(float(elapsed), 4),
        "selector_peak_mem_mb": round(float(peak_mem_mb), 4),
        "baseline_parity": parity_ok,
    }
    return ViewSelectionResult(
        selected_image_ids=selected_ids,
        selected_names=selected_names,
        meta=meta,
        diagnostics=diagnostics,
        support_pack=voxel_pack,
        shadow_pack=shadow_pack,
    )


def save_selection(model_path, selected_names, meta, diagnostics, support_pack, shadow_pack=None):
    os.makedirs(model_path, exist_ok=True)
    diag_dir = os.path.join(model_path, "coverage_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    jp = save_json(os.path.join(model_path, "selected_views.json"), list(selected_names))
    tp = os.path.join(model_path, "selected_train_views.txt")
    with open(tp, "w", encoding="utf-8") as f:
        for name in selected_names:
            f.write(name + "\n")
    mp = save_json(os.path.join(model_path, "view_select_meta.json"), meta)
    save_json(os.path.join(diag_dir, "point_support_stats.json"), diagnostics["point"])
    save_json(os.path.join(diag_dir, "voxel_support_stats.json"), diagnostics["voxel"])
    save_json(os.path.join(diag_dir, "observability_stats.json"), diagnostics["obs"])
    save_shadow_diagnostics(model_path, shadow_pack)
    save_support_pack_npz(os.path.join(model_path, "vf_support_pack.npz"), support_pack)
    return jp, tp, mp
