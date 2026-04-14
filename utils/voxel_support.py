from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np


EPS = 1e-8


@dataclass
class VoxelSupportPack:
    candidate_view_names: np.ndarray
    selected_view_names: np.ndarray
    pt_ids: np.ndarray
    pt_xyz: np.ndarray
    pt_reproj_err: np.ndarray
    pt_track_len: np.ndarray
    pt2voxel: np.ndarray
    voxel_keys: np.ndarray
    voxel_origin: np.ndarray
    voxel_size: np.ndarray
    voxel_depth: np.ndarray
    voxel_center: np.ndarray
    n_pts: np.ndarray
    err_med: np.ndarray
    track_med: np.ndarray
    density_conf: np.ndarray
    track_conf: np.ndarray
    difficulty: np.ndarray
    obs_cap: np.ndarray
    target_hits: np.ndarray
    easy_saturated: np.ndarray
    viewdep_sink: np.ndarray
    recoverable_hard: np.ndarray
    cand_view_ptr: np.ndarray
    cand_view_idx: np.ndarray
    view_vox_ptr: np.ndarray
    view_vox_idx: np.ndarray
    final_voxel_hit_views: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    final_voxel_support_ok_ratio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    shadow_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    shadow_view_names: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=object))
    shadow_bboxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.int32))
    shadow_obs_cap: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    shadow_hit_views: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    shadow_anchor_conf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    shadow_edge_conf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    shadow_ring_anchorable: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    shadow_local_depth_cont: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    shadow_z_nb: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    shadow_trusted_prior: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    coarse_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    coarse_cell_size: float = 1.0
    coarse_grid_size: int = 256
    max_leaf_depth: int = 2
    cross_layer_merge_ratio: float = 0.0
    leaf_split_ratio: float = 0.0
    diagnostics: dict = field(default_factory=dict)


def _normalize01(values: np.ndarray, q: float = 95.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    scale = np.percentile(values, q)
    scale = max(float(scale), EPS)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def _connected_components(points: np.ndarray, radius: float) -> int:
    if points.shape[0] <= 1:
        return int(points.shape[0])
    visited = np.zeros(points.shape[0], dtype=bool)
    count = 0
    sq_radius = float(radius) ** 2
    for root in range(points.shape[0]):
        if visited[root]:
            continue
        count += 1
        stack = [root]
        visited[root] = True
        while stack:
            idx = stack.pop()
            diffs = points - points[idx]
            neigh = np.flatnonzero(np.sum(diffs * diffs, axis=1) <= sq_radius)
            for nxt in neigh:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
    return count


def _split_needed(
    xyz_norm: np.ndarray,
    z_med_point: np.ndarray,
    point_indices: np.ndarray,
    cell_size: float,
    *,
    n_split_max: int,
    eta_size: float,
    tau_z_ratio: float,
    tau_z_abs: float,
    n_leaf_min: int,
) -> bool:
    if point_indices.size <= n_leaf_min:
        return False
    local_xyz = xyz_norm[point_indices]
    if point_indices.size > n_split_max:
        return True
    diameter = float(np.linalg.norm(local_xyz.max(axis=0) - local_xyz.min(axis=0)))
    if diameter > eta_size * cell_size:
        return True
    z_vals = z_med_point[point_indices]
    if z_vals.size >= 4:
        q10, q90 = np.percentile(z_vals, [10, 90])
        z_med = max(float(np.median(z_vals)), EPS)
        if q90 / max(q10, EPS) > tau_z_ratio:
            return True
        if (q90 - q10) > tau_z_abs * z_med:
            return True
    tau_layer_sep = 1.5 * cell_size
    if _connected_components(local_xyz, tau_layer_sep) >= 2:
        return True
    return False


def build_leaf_voxels(
    pt_xyz: np.ndarray,
    point_depth_lists: List[np.ndarray],
    *,
    coarse_grid_size: int = 256,
    n_split_max: int = 64,
    eta_size: float = 1.5,
    tau_z_ratio: float = 1.8,
    tau_z_abs: float = 0.15,
    max_leaf_depth: int = 2,
    n_leaf_min: int = 8,
) -> Dict[str, np.ndarray]:
    pt_xyz = np.asarray(pt_xyz, dtype=np.float64)
    xyz_min = pt_xyz.min(axis=0)
    xyz_max = pt_xyz.max(axis=0)
    bbox_diag = float(np.linalg.norm(xyz_max - xyz_min))
    cell_size_0 = bbox_diag / max(int(coarse_grid_size), 1)
    cell_size_0 = max(cell_size_0, EPS)
    xyz_norm = (pt_xyz - xyz_min) / cell_size_0
    z_med_point = np.array(
        [np.median(zs) if len(zs) else 0.0 for zs in point_depth_lists],
        dtype=np.float64,
    )
    coarse_keys = np.floor(xyz_norm).astype(np.int32)
    coarse_map: Dict[Tuple[int, int, int], List[int]] = {}
    for pt_idx, key in enumerate(coarse_keys):
        coarse_map.setdefault(tuple(int(v) for v in key.tolist()), []).append(pt_idx)

    leaf_records: List[dict] = []
    split_roots = 0

    def recurse(
        point_indices: np.ndarray,
        *,
        depth: int,
        ix: int,
        iy: int,
        iz: int,
        origin: np.ndarray,
        cell_size: float,
    ) -> None:
        nonlocal split_roots
        if (
            depth < max_leaf_depth
            and _split_needed(
                xyz_norm,
                z_med_point,
                point_indices,
                cell_size,
                n_split_max=n_split_max,
                eta_size=eta_size,
                tau_z_ratio=tau_z_ratio,
                tau_z_abs=tau_z_abs,
                n_leaf_min=n_leaf_min,
            )
        ):
            split_roots += 1
            child_size = cell_size / 2.0
            local = xyz_norm[point_indices] - origin[None, :]
            child_bits = np.floor(local / max(child_size, EPS)).astype(np.int32)
            child_bits = np.clip(child_bits, 0, 1)
            child_map: Dict[Tuple[int, int, int], List[int]] = {}
            for local_idx, child in zip(point_indices.tolist(), child_bits.tolist()):
                child_map.setdefault(tuple(child), []).append(local_idx)
            for child_key, child_points in child_map.items():
                child_origin = origin + np.array(child_key, dtype=np.float64) * child_size
                recurse(
                    np.asarray(child_points, dtype=np.int32),
                    depth=depth + 1,
                    ix=ix * 2 + child_key[0],
                    iy=iy * 2 + child_key[1],
                    iz=iz * 2 + child_key[2],
                    origin=child_origin,
                    cell_size=child_size,
                )
            return
        leaf_records.append(
            {
                "key": np.array([depth, ix, iy, iz], dtype=np.int32),
                "origin_norm": origin.astype(np.float32),
                "size_norm": float(cell_size),
                "depth": depth,
                "points": np.asarray(point_indices, dtype=np.int32),
            }
        )

    for coarse_key, point_list in coarse_map.items():
        recurse(
            np.asarray(point_list, dtype=np.int32),
            depth=0,
            ix=int(coarse_key[0]),
            iy=int(coarse_key[1]),
            iz=int(coarse_key[2]),
            origin=np.array(coarse_key, dtype=np.float64),
            cell_size=1.0,
        )

    pt2voxel = np.full(pt_xyz.shape[0], -1, dtype=np.int32)
    voxel_keys = []
    voxel_origin = []
    voxel_size = []
    voxel_depth = []
    voxel_center = []
    for voxel_idx, record in enumerate(leaf_records):
        pt2voxel[record["points"]] = voxel_idx
        voxel_keys.append(record["key"])
        origin_world = xyz_min + record["origin_norm"] * cell_size_0
        size_world = record["size_norm"] * cell_size_0
        voxel_origin.append(origin_world.astype(np.float32))
        voxel_size.append(np.float32(size_world))
        voxel_depth.append(np.int32(record["depth"]))
        voxel_center.append((origin_world + 0.5 * size_world).astype(np.float32))

    return {
        "pt2voxel": pt2voxel,
        "voxel_keys": np.asarray(voxel_keys, dtype=np.int32),
        "voxel_origin": np.asarray(voxel_origin, dtype=np.float32),
        "voxel_size": np.asarray(voxel_size, dtype=np.float32),
        "voxel_depth": np.asarray(voxel_depth, dtype=np.int32),
        "voxel_center": np.asarray(voxel_center, dtype=np.float32),
        "coarse_origin": xyz_min.astype(np.float32),
        "coarse_cell_size": float(cell_size_0),
        "cross_layer_merge_ratio": float(split_roots / max(len(coarse_map), 1)),
        "leaf_split_ratio": float(np.mean(np.asarray(voxel_depth, dtype=np.int32) > 0))
        if voxel_depth
        else 0.0,
    }


def compute_density_track_conf(
    pt2voxel: np.ndarray,
    pt_reproj_err: np.ndarray,
    pt_track_len: np.ndarray,
) -> Dict[str, np.ndarray]:
    n_voxels = int(pt2voxel.max() + 1) if pt2voxel.size else 0
    n_pts = np.bincount(pt2voxel, minlength=n_voxels).astype(np.int32)
    err_med = np.zeros(n_voxels, dtype=np.float32)
    track_med = np.zeros(n_voxels, dtype=np.float32)
    for voxel_idx in range(n_voxels):
        mask = pt2voxel == voxel_idx
        err_med[voxel_idx] = float(np.median(pt_reproj_err[mask])) if np.any(mask) else 0.0
        track_med[voxel_idx] = float(np.median(pt_track_len[mask])) if np.any(mask) else 0.0
    density_conf = _normalize01(np.log1p(n_pts))
    track_conf = _normalize01(track_med)
    return {
        "n_pts": n_pts,
        "err_med": err_med,
        "track_med": track_med,
        "density_conf": density_conf,
        "track_conf": track_conf,
    }


def compute_difficulty(
    *,
    density_conf: np.ndarray,
    track_conf: np.ndarray,
    err_med: np.ndarray,
) -> np.ndarray:
    err_bad = _normalize01(err_med)
    difficulty = (
        0.45 * (1.0 - np.asarray(density_conf))
        + 0.35 * (1.0 - np.asarray(track_conf))
        + 0.20 * np.asarray(err_bad)
    )
    return np.clip(difficulty, 0.0, 1.0).astype(np.float32)


def _build_view_voxel_csr(vis_pts_ptr: np.ndarray, vis_pts_idx: np.ndarray, pt2voxel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_views = int(len(vis_pts_ptr) - 1)
    ptr = [0]
    idx: List[int] = []
    for view_idx in range(n_views):
        start = vis_pts_ptr[view_idx]
        end = vis_pts_ptr[view_idx + 1]
        voxels = np.unique(pt2voxel[vis_pts_idx[start:end]])
        idx.extend(voxels.tolist())
        ptr.append(len(idx))
    return np.asarray(ptr, dtype=np.int64), np.asarray(idx, dtype=np.int32)


def _build_voxel_candidate_views(view_vox_ptr: np.ndarray, view_vox_idx: np.ndarray, n_voxels: int) -> Tuple[np.ndarray, np.ndarray]:
    voxel_to_views: List[List[int]] = [[] for _ in range(n_voxels)]
    for view_idx in range(len(view_vox_ptr) - 1):
        start = view_vox_ptr[view_idx]
        end = view_vox_ptr[view_idx + 1]
        for voxel_idx in view_vox_idx[start:end]:
            voxel_to_views[int(voxel_idx)].append(view_idx)
    ptr = [0]
    idx: List[int] = []
    for views in voxel_to_views:
        idx.extend(views)
        ptr.append(len(idx))
    return np.asarray(ptr, dtype=np.int64), np.asarray(idx, dtype=np.int32)


def compute_obs_cap_and_target_hits(
    *,
    cand_view_ptr: np.ndarray,
    difficulty: np.ndarray,
    viewdep_sink: np.ndarray,
    tau_hard: float = 0.60,
) -> Dict[str, np.ndarray]:
    obs_cap = np.diff(cand_view_ptr).astype(np.int32)
    recoverable_hard = (difficulty >= tau_hard) & (~viewdep_sink)
    desired_hits = np.full(obs_cap.shape[0], 2, dtype=np.int32)
    desired_hits[(recoverable_hard) & (obs_cap >= 3)] = 3
    desired_hits[viewdep_sink] = 2
    target_hits = np.minimum(desired_hits, np.maximum(obs_cap, 1))
    return {
        "obs_cap": obs_cap,
        "target_hits": target_hits.astype(np.int32),
        "recoverable_hard": recoverable_hard.astype(bool),
    }


def build_voxel_support(point_pack: dict, vf_cfg) -> VoxelSupportPack:
    voxel_meta = build_leaf_voxels(
        point_pack["pt_xyz"],
        point_pack["point_depth_lists"],
        coarse_grid_size=getattr(vf_cfg, "coarse_grid_size", 256),
        n_split_max=getattr(vf_cfg, "n_split_max", 64),
        eta_size=getattr(vf_cfg, "eta_size", 1.5),
        tau_z_ratio=getattr(vf_cfg, "tau_z_ratio", 1.8),
        tau_z_abs=getattr(vf_cfg, "tau_z_abs", 0.15),
        max_leaf_depth=getattr(vf_cfg, "max_leaf_depth", 2) if getattr(vf_cfg, "enable_split_safe_voxel", False) else 0,
        n_leaf_min=getattr(vf_cfg, "n_leaf_min", 8),
    )
    pt2voxel = voxel_meta["pt2voxel"]
    voxel_stats = compute_density_track_conf(
        pt2voxel,
        point_pack["pt_reproj_err"],
        point_pack["pt_track_len"],
    )
    difficulty = compute_difficulty(
        density_conf=voxel_stats["density_conf"],
        track_conf=voxel_stats["track_conf"],
        err_med=voxel_stats["err_med"],
    )
    err_med = voxel_stats["err_med"]
    track_med = voxel_stats["track_med"]
    density_conf = voxel_stats["density_conf"]
    track_conf = voxel_stats["track_conf"]
    n_pts = voxel_stats["n_pts"]
    err_global = np.asarray(point_pack["pt_reproj_err"], dtype=np.float64)
    if err_global.size:
        q25, q75 = np.percentile(err_global, [25, 75])
        iqr = q75 - q25
        median_err = np.median(err_global)
        err_threshold = max(q75 + 1.5 * iqr, 3.0 * median_err)
    else:
        err_threshold = 0.0
    viewdep_sink = (
        (err_med >= err_threshold)
        & (track_med <= 2.0)
        & ((track_conf <= 0.20) | (density_conf <= 0.15))
    )
    if not getattr(vf_cfg, "enable_viewdep_guard", False):
        viewdep_sink = np.zeros_like(viewdep_sink, dtype=bool)
    view_vox_ptr, view_vox_idx = _build_view_voxel_csr(
        point_pack["vis_pts_ptr"],
        point_pack["vis_pts_idx"],
        pt2voxel,
    )
    cand_view_ptr, cand_view_idx = _build_voxel_candidate_views(
        view_vox_ptr,
        view_vox_idx,
        int(n_pts.shape[0]),
    )
    obs_meta = compute_obs_cap_and_target_hits(
        cand_view_ptr=cand_view_ptr,
        difficulty=difficulty,
        viewdep_sink=viewdep_sink,
        tau_hard=getattr(vf_cfg, "tau_hard", 0.60),
    )
    diagnostics = {
        "enabled": True,
        "cross_layer_merge_ratio": float(voxel_meta["cross_layer_merge_ratio"]),
        "leaf_split_ratio": float(voxel_meta["leaf_split_ratio"]),
        "viewdep_sink_ratio": float(np.mean(viewdep_sink)) if viewdep_sink.size else 0.0,
        "recoverable_hard_ratio": float(np.mean(obs_meta["recoverable_hard"])) if viewdep_sink.size else 0.0,
    }
    return VoxelSupportPack(
        candidate_view_names=np.asarray(point_pack["candidate_view_names"]),
        selected_view_names=np.asarray([], dtype=object),
        pt_ids=np.asarray(point_pack["pt_ids"], dtype=np.int64),
        pt_xyz=np.asarray(point_pack["pt_xyz"], dtype=np.float32),
        pt_reproj_err=np.asarray(point_pack["pt_reproj_err"], dtype=np.float32),
        pt_track_len=np.asarray(point_pack["pt_track_len"], dtype=np.int32),
        pt2voxel=np.asarray(pt2voxel, dtype=np.int32),
        voxel_keys=np.asarray(voxel_meta["voxel_keys"], dtype=np.int32),
        voxel_origin=np.asarray(voxel_meta["voxel_origin"], dtype=np.float32),
        voxel_size=np.asarray(voxel_meta["voxel_size"], dtype=np.float32),
        voxel_depth=np.asarray(voxel_meta["voxel_depth"], dtype=np.int32),
        voxel_center=np.asarray(voxel_meta["voxel_center"], dtype=np.float32),
        n_pts=np.asarray(n_pts, dtype=np.int32),
        err_med=np.asarray(err_med, dtype=np.float32),
        track_med=np.asarray(track_med, dtype=np.float32),
        density_conf=np.asarray(density_conf, dtype=np.float32),
        track_conf=np.asarray(track_conf, dtype=np.float32),
        difficulty=np.asarray(difficulty, dtype=np.float32),
        obs_cap=np.asarray(obs_meta["obs_cap"], dtype=np.int32),
        target_hits=np.asarray(obs_meta["target_hits"], dtype=np.int32),
        easy_saturated=np.zeros_like(obs_meta["target_hits"], dtype=bool),
        viewdep_sink=np.asarray(viewdep_sink, dtype=bool),
        recoverable_hard=np.asarray(obs_meta["recoverable_hard"], dtype=bool),
        cand_view_ptr=np.asarray(cand_view_ptr, dtype=np.int64),
        cand_view_idx=np.asarray(cand_view_idx, dtype=np.int32),
        view_vox_ptr=np.asarray(view_vox_ptr, dtype=np.int64),
        view_vox_idx=np.asarray(view_vox_idx, dtype=np.int32),
        coarse_origin=np.asarray(voxel_meta["coarse_origin"], dtype=np.float32),
        coarse_cell_size=float(voxel_meta["coarse_cell_size"]),
        coarse_grid_size=getattr(vf_cfg, "coarse_grid_size", 256),
        max_leaf_depth=getattr(vf_cfg, "max_leaf_depth", 2) if getattr(vf_cfg, "enable_split_safe_voxel", False) else 0,
        cross_layer_merge_ratio=float(voxel_meta["cross_layer_merge_ratio"]),
        leaf_split_ratio=float(voxel_meta["leaf_split_ratio"]),
        diagnostics=diagnostics,
    )


def finalize_voxel_support_pack(
    pack: VoxelSupportPack,
    *,
    selected_view_names: Iterable[str],
    voxel_hit_views: np.ndarray,
    voxel_support_ok_ratio: np.ndarray,
    target_hits: np.ndarray,
) -> VoxelSupportPack:
    pack.selected_view_names = np.asarray(list(selected_view_names))
    pack.final_voxel_hit_views = np.asarray(voxel_hit_views, dtype=np.int16)
    pack.final_voxel_support_ok_ratio = np.asarray(voxel_support_ok_ratio, dtype=np.float32)
    pack.target_hits = np.asarray(target_hits, dtype=np.int32)
    pack.easy_saturated = (
        (pack.final_voxel_hit_views >= pack.target_hits)
        & (pack.final_voxel_support_ok_ratio >= 0.50)
        & (pack.difficulty < 0.35)
    )
    return pack


def save_support_pack_npz(path: str, support_pack: VoxelSupportPack) -> None:
    arrays = asdict(support_pack)
    serializable = {}
    for key, value in arrays.items():
        if key == "diagnostics":
            continue
        if isinstance(value, np.ndarray):
            serializable[key] = value
        else:
            serializable[key] = np.asarray(value)
    np.savez_compressed(path, **serializable)


def load_vf_support_pack(path: str) -> VoxelSupportPack:
    payload = np.load(path, allow_pickle=True)
    data = {key: payload[key] for key in payload.files}
    return VoxelSupportPack(
        candidate_view_names=data.get("candidate_view_names", np.asarray([], dtype=object)),
        selected_view_names=data.get("selected_view_names", np.asarray([], dtype=object)),
        pt_ids=data.get("pt_ids", np.asarray([], dtype=np.int64)),
        pt_xyz=data.get("pt_xyz", np.zeros((0, 3), dtype=np.float32)),
        pt_reproj_err=data.get("pt_reproj_err", np.asarray([], dtype=np.float32)),
        pt_track_len=data.get("pt_track_len", np.asarray([], dtype=np.int32)),
        pt2voxel=data.get("pt2voxel", np.asarray([], dtype=np.int32)),
        voxel_keys=data.get("voxel_keys", np.zeros((0, 4), dtype=np.int32)),
        voxel_origin=data.get("voxel_origin", np.zeros((0, 3), dtype=np.float32)),
        voxel_size=data.get("voxel_size", np.asarray([], dtype=np.float32)),
        voxel_depth=data.get("voxel_depth", np.asarray([], dtype=np.int32)),
        voxel_center=data.get("voxel_center", np.zeros((0, 3), dtype=np.float32)),
        n_pts=data.get("n_pts", np.asarray([], dtype=np.int32)),
        err_med=data.get("err_med", np.asarray([], dtype=np.float32)),
        track_med=data.get("track_med", np.asarray([], dtype=np.float32)),
        density_conf=data.get("density_conf", np.asarray([], dtype=np.float32)),
        track_conf=data.get("track_conf", np.asarray([], dtype=np.float32)),
        difficulty=data.get("difficulty", np.asarray([], dtype=np.float32)),
        obs_cap=data.get("obs_cap", np.asarray([], dtype=np.int32)),
        target_hits=data.get("target_hits", np.asarray([], dtype=np.int32)),
        easy_saturated=data.get("easy_saturated", np.asarray([], dtype=bool)),
        viewdep_sink=data.get("viewdep_sink", np.asarray([], dtype=bool)),
        recoverable_hard=data.get("recoverable_hard", np.asarray([], dtype=bool)),
        cand_view_ptr=data.get("cand_view_ptr", np.asarray([0], dtype=np.int64)),
        cand_view_idx=data.get("cand_view_idx", np.asarray([], dtype=np.int32)),
        view_vox_ptr=data.get("view_vox_ptr", np.asarray([0], dtype=np.int64)),
        view_vox_idx=data.get("view_vox_idx", np.asarray([], dtype=np.int32)),
        final_voxel_hit_views=data.get("final_voxel_hit_views", np.asarray([], dtype=np.int16)),
        final_voxel_support_ok_ratio=data.get("final_voxel_support_ok_ratio", np.asarray([], dtype=np.float32)),
        shadow_ids=data.get("shadow_ids", np.asarray([], dtype=np.int32)),
        shadow_view_names=data.get("shadow_view_names", np.asarray([], dtype=object)),
        shadow_bboxes=data.get("shadow_bboxes", np.zeros((0, 4), dtype=np.int32)),
        shadow_obs_cap=data.get("shadow_obs_cap", np.asarray([], dtype=np.int32)),
        shadow_hit_views=data.get("shadow_hit_views", np.asarray([], dtype=np.int32)),
        shadow_anchor_conf=data.get("shadow_anchor_conf", np.asarray([], dtype=np.float32)),
        shadow_edge_conf=data.get("shadow_edge_conf", np.asarray([], dtype=np.float32)),
        shadow_ring_anchorable=data.get("shadow_ring_anchorable", np.asarray([], dtype=bool)),
        shadow_local_depth_cont=data.get("shadow_local_depth_cont", np.asarray([], dtype=bool)),
        shadow_z_nb=data.get("shadow_z_nb", np.asarray([], dtype=np.float32)),
        shadow_trusted_prior=data.get("shadow_trusted_prior", np.asarray([], dtype=bool)),
        coarse_origin=data.get("coarse_origin", np.zeros(3, dtype=np.float32)),
        coarse_cell_size=float(np.asarray(data.get("coarse_cell_size", 1.0)).item()),
        coarse_grid_size=int(np.asarray(data.get("coarse_grid_size", 256)).item()),
        max_leaf_depth=int(np.asarray(data.get("max_leaf_depth", 2)).item()),
        cross_layer_merge_ratio=float(np.asarray(data.get("cross_layer_merge_ratio", 0.0)).item()),
        leaf_split_ratio=float(np.asarray(data.get("leaf_split_ratio", 0.0)).item()),
        diagnostics={},
    )