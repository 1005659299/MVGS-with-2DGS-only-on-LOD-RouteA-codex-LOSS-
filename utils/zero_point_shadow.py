from __future__ import annotations

import json
import os
from collections import deque

import numpy as np
from PIL import Image

from utils.prior_alignment import trusted_prior_local


def _find_mask_file(root: str, image_name: str) -> str | None:
    if not root:
        return None
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"):
        candidate = os.path.join(root, image_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_mask(root: str, image_name: str, width: int, height: int) -> np.ndarray:
    path = _find_mask_file(root, image_name)
    if path is None:
        return np.ones((height, width), dtype=bool)
    if path.endswith(".npy"):
        mask = np.load(path)
    else:
        mask = np.array(Image.open(path).convert("L"))
    if mask.shape[:2] != (height, width):
        mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((width, height), resample=Image.NEAREST))
    return mask > 0


def _distance_to_sparse(height: int, width: int, points_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.full((height, width), np.inf, dtype=np.float32)
    yy, xx = np.indices((height, width), dtype=np.float32)
    flat = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    best = np.full(flat.shape[0], np.inf, dtype=np.float32)
    pts = points_xy.astype(np.float32)
    chunk = 512
    for start in range(0, pts.shape[0], chunk):
        block = pts[start:start + chunk]
        diff = flat[:, None, :] - block[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        best = np.minimum(best, np.min(dist2, axis=1))
    return np.sqrt(best).astype(np.float32).reshape(height, width)


def build_nopoint_regions(
    *,
    width: int,
    height: int,
    sparse_xy: np.ndarray,
    static_safe_mask: np.ndarray,
    edge_safe_mask: np.ndarray,
    tau_sp: float = 6.0,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    dist = _distance_to_sparse(height, width, sparse_xy)
    nopoint = (dist > tau_sp) & static_safe_mask & edge_safe_mask
    visited = np.zeros_like(nopoint, dtype=bool)
    components = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not nopoint[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            pixels = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and nopoint[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(pixels) < 16:
                continue
            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            mask = np.zeros_like(nopoint, dtype=bool)
            mask[tuple(np.array(pixels).T)] = True
            components.append(
                {
                    "mask": mask,
                    "bbox": [int(min(ys)), int(min(xs)), int(max(ys) + 1), int(max(xs) + 1)],
                    "area": int(len(pixels)),
                }
            )
    return nopoint, components, dist


def _compute_local_depth_continuity(
    *,
    region_mask: np.ndarray,
    sparse_xy: np.ndarray,
    sparse_depth: np.ndarray,
    static_safe_mask: np.ndarray,
    anchor_depths: np.ndarray,
    r_nb: int = 8,
) -> tuple[bool, float | None]:
    if sparse_xy.size == 0 or anchor_depths.size == 0:
        return False, None
    ys, xs = np.where(region_mask)
    if ys.size == 0:
        return False, None
    y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
    nb_box = np.zeros_like(region_mask, dtype=bool)
    nb_box[
        max(y0 - r_nb, 0):min(y1 + r_nb, region_mask.shape[0]),
        max(x0 - r_nb, 0):min(x1 + r_nb, region_mask.shape[1]),
    ] = True
    nb_mask = nb_box & (~region_mask)
    rounded = np.round(sparse_xy).astype(np.int32)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < region_mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < region_mask.shape[0])
    )
    if not np.any(in_bounds):
        return False, None
    rounded_valid = rounded[in_bounds]
    neighborhood_ids = np.flatnonzero(in_bounds)[
        nb_mask[rounded_valid[:, 1], rounded_valid[:, 0]]
        & static_safe_mask[rounded_valid[:, 1], rounded_valid[:, 0]]
    ]
    if neighborhood_ids.size == 0:
        return False, None
    z_nb = float(np.median(sparse_depth[neighborhood_ids]))
    z_anchor = float(np.median(anchor_depths))
    z_anchor_iqr = (
        float(np.subtract(*np.percentile(anchor_depths, [75, 25])))
        if anchor_depths.size >= 4
        else 0.0
    )
    tau_z_cont = max(0.25 * max(z_anchor, 1e-6), z_anchor_iqr, 0.15)
    return abs(z_anchor - z_nb) <= tau_z_cont, z_nb


def filter_ring_anchors(
    *,
    region_mask: np.ndarray,
    sparse_xy: np.ndarray,
    sparse_depth: np.ndarray,
    pt_track_len: np.ndarray,
    pt_reproj_err: np.ndarray,
    static_safe_mask: np.ndarray,
    tau_track_anchor: int,
    tau_err_anchor: float,
    lambda_far: float,
) -> dict:
    if sparse_xy.size == 0:
        return {"anchor_indices": np.zeros(0, dtype=np.int32), "z_far_valid": 0.0}
    ys, xs = np.where(region_mask)
    y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
    r_out = 4
    ring_box = np.zeros_like(region_mask, dtype=bool)
    ring_box[max(y0 - r_out, 0):min(y1 + r_out, region_mask.shape[0]), max(x0 - r_out, 0):min(x1 + r_out, region_mask.shape[1])] = True
    ring_mask = ring_box & (~region_mask)
    anchor_mask = np.zeros(sparse_xy.shape[0], dtype=bool)
    rounded = np.round(sparse_xy).astype(np.int32)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < region_mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < region_mask.shape[0])
    )
    rounded = rounded[in_bounds]
    anchor_mask[in_bounds] = ring_mask[rounded[:, 1], rounded[:, 0]]
    base_depth_mask = (pt_track_len >= tau_track_anchor) & (pt_reproj_err <= tau_err_anchor)
    z_base = sparse_depth[base_depth_mask]
    if z_base.size:
        q25, q75 = np.percentile(z_base, [25, 75])
        z_far_valid = np.percentile(z_base, 95) + lambda_far * (q75 - q25)
    else:
        z_far_valid = float(np.max(sparse_depth) if sparse_depth.size else 0.0)
    valid_anchor = anchor_mask & base_depth_mask & (sparse_depth <= z_far_valid)
    if np.any(valid_anchor):
        xy = np.round(sparse_xy[valid_anchor]).astype(np.int32)
        static_ok = static_safe_mask[xy[:, 1], xy[:, 0]]
        valid_ids = np.flatnonzero(valid_anchor)[static_ok]
    else:
        valid_ids = np.zeros(0, dtype=np.int32)
    return {"anchor_indices": valid_ids.astype(np.int32), "z_far_valid": float(z_far_valid)}


def anchor_valid(anchor_info: dict) -> bool:
    return bool(anchor_info.get("ring_anchorable", False))


def compute_shadow_confidence(anchor_depths: np.ndarray, n_ring: int, shadow_edge_conf: float) -> float:
    if anchor_depths.size == 0:
        return 0.0
    depth_iqr = np.subtract(*np.percentile(anchor_depths, [75, 25]))
    depth_med = max(float(np.median(anchor_depths)), 1e-6)
    count_conf = min(anchor_depths.size / max(n_ring, 1), 1.0)
    spread_conf = np.clip(1.0 - depth_iqr / max(0.25 * depth_med, 1e-6), 0.0, 1.0)
    return float(np.clip(count_conf * spread_conf * shadow_edge_conf, 0.0, 1.0))


def trusted_prior_shadow(anchor_info: dict, trusted_view: bool, tau_edge_shadow: float = 0.60) -> bool:
    return trusted_prior_local(
        ring_anchorable=bool(anchor_info.get("ring_anchorable", False)),
        shadow_edge_conf=float(anchor_info.get("shadow_edge_conf", 0.0)),
        trusted_view=trusted_view,
        tau_edge_shadow=tau_edge_shadow,
    )


def build_shadow_carriers(
    *,
    point_pack: dict,
    cam_extrinsics: dict,
    cam_intrinsics,
    image_size_by_name: dict,
    images_root: str,
    mask_root: str,
    camera_mask_root: str,
    vf_cfg,
) -> dict:
    carriers = []
    anchor_conf_hist = []
    edge_conf_hist = []
    ghost_anchor_reject = 0
    ghost_anchor_total = 0
    tau_track_anchor = 3
    err_scene = np.asarray(point_pack["pt_reproj_err"], dtype=np.float32)
    tau_err_anchor = min(3.0 * float(np.median(err_scene)) if err_scene.size else 1.0, float(np.percentile(err_scene, 90)) if err_scene.size else 1.0)
    lambda_far = 1.5
    for view_idx, image_id in enumerate(point_pack["image_ids"]):
        _ = image_id
        image_name = point_pack["candidate_view_names"][view_idx]
        width, height = image_size_by_name[image_name]
        static_mask = _load_mask(mask_root, image_name, width, height)
        camera_mask = _load_mask(camera_mask_root, image_name, width, height) if camera_mask_root else np.ones((height, width), dtype=bool)
        static_safe = static_mask & camera_mask
        edge_margin = max(int(getattr(vf_cfg, "edge_margin_px", 15)), int(round(0.01 * min(height, width))))
        yy, xx = np.indices((height, width))
        d_edge = np.minimum.reduce([xx, width - 1 - xx, yy, height - 1 - yy]).astype(np.float32)
        edge_safe = d_edge >= edge_margin
        edge_soft = np.clip(d_edge / max(edge_margin, 1), 0.0, 1.0)
        pts = point_pack["view_sparse_xy"][view_idx]
        depths = point_pack["view_sparse_depth"][view_idx]
        nopoint_mask, regions, _ = build_nopoint_regions(
            width=width,
            height=height,
            sparse_xy=pts,
            static_safe_mask=static_safe,
            edge_safe_mask=edge_safe,
            tau_sp=getattr(vf_cfg, "shadow_tau_sp", 6.0),
        )
        _ = nopoint_mask
        for region in regions:
            anchor_info = filter_ring_anchors(
                region_mask=region["mask"],
                sparse_xy=pts,
                sparse_depth=depths,
                pt_track_len=point_pack["view_sparse_track_len"][view_idx],
                pt_reproj_err=point_pack["view_sparse_err"][view_idx],
                static_safe_mask=static_safe,
                tau_track_anchor=tau_track_anchor,
                tau_err_anchor=tau_err_anchor,
                lambda_far=lambda_far,
            )
            anchor_ids = anchor_info["anchor_indices"]
            ghost_anchor_total += 1
            if anchor_ids.size == 0:
                ghost_anchor_reject += 1
                continue
            anchor_depths = depths[anchor_ids]
            if anchor_depths.size >= 4:
                anchor_med = np.median(anchor_depths)
                anchor_iqr = np.subtract(*np.percentile(anchor_depths, [75, 25]))
            else:
                anchor_med = float(np.median(anchor_depths))
                anchor_iqr = 0.0
            local_depth_cont, z_nb = _compute_local_depth_continuity(
                region_mask=region["mask"],
                sparse_xy=pts,
                sparse_depth=depths,
                static_safe_mask=static_safe,
                anchor_depths=anchor_depths,
            )
            ring_anchorable = bool(anchor_ids.size >= 8 and anchor_iqr <= 0.25 * max(anchor_med, 1e-6) and local_depth_cont)
            shadow_edge_conf = float(np.mean(edge_soft[region["mask"]]))
            shadow_anchor_conf = compute_shadow_confidence(anchor_depths, 8, shadow_edge_conf)
            anchor_summary = {
                "ring_anchorable": ring_anchorable,
                "shadow_edge_conf": shadow_edge_conf,
            }
            region_record = {
                "view_name": image_name,
                "bbox": region["bbox"],
                "area": region["area"],
                "anchor_count": int(anchor_ids.size),
                "shadow_obs_cap": 1,
                "shadow_hit_views": 0,
                "shadow_anchor_conf": shadow_anchor_conf,
                "shadow_edge_conf": shadow_edge_conf,
                "ring_anchorable": ring_anchorable,
                "local_depth_cont": bool(local_depth_cont),
                "z_nb": None if z_nb is None else float(z_nb),
                "trusted_prior_shadow": bool(
                    trusted_prior_shadow(anchor_summary, trusted_view=True, tau_edge_shadow=0.60)
                    and shadow_anchor_conf >= 0.50
                ),
            }
            if not ring_anchorable:
                ghost_anchor_reject += 1
                continue
            carriers.append(region_record)
            anchor_conf_hist.append(shadow_anchor_conf)
            edge_conf_hist.append(shadow_edge_conf)
    return {
        "carriers": carriers,
        "shadow_ids": np.arange(len(carriers), dtype=np.int32),
        "shadow_view_names": np.asarray([c["view_name"] for c in carriers], dtype=object),
        "shadow_bboxes": np.asarray([c["bbox"] for c in carriers], dtype=np.int32) if carriers else np.zeros((0, 4), dtype=np.int32),
        "shadow_obs_cap": np.asarray([c["shadow_obs_cap"] for c in carriers], dtype=np.int32),
        "shadow_hit_views": np.asarray([c["shadow_hit_views"] for c in carriers], dtype=np.int32),
        "shadow_anchor_conf": np.asarray([c["shadow_anchor_conf"] for c in carriers], dtype=np.float32),
        "shadow_edge_conf": np.asarray([c["shadow_edge_conf"] for c in carriers], dtype=np.float32),
        "shadow_ring_anchorable": np.asarray([c["ring_anchorable"] for c in carriers], dtype=bool),
        "shadow_local_depth_cont": np.asarray([c["local_depth_cont"] for c in carriers], dtype=bool),
        "shadow_z_nb": np.asarray([
            np.nan if c["z_nb"] is None else c["z_nb"] for c in carriers
        ], dtype=np.float32),
        "shadow_trusted_prior": np.asarray([c["trusted_prior_shadow"] for c in carriers], dtype=bool),
        "diagnostics": {
            "enabled": True,
            "ghost_anchor_reject_ratio": float(ghost_anchor_reject / max(ghost_anchor_total, 1)),
            "shadow_anchor_conf_hist": anchor_conf_hist,
            "shadow_edge_conf_hist": edge_conf_hist,
            "shadow_hotspot_overlap": None,
        },
    }


def save_shadow_diagnostics(model_path: str, shadow_pack: dict | None) -> None:
    diag_dir = os.path.join(model_path, "coverage_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    stats_path = os.path.join(diag_dir, "zero_point_shadow_stats.json")
    carriers_path = os.path.join(diag_dir, "shadow_carriers.json")
    if shadow_pack is None:
        stats = {
            "enabled": False,
            "ghost_anchor_reject_ratio": None,
            "shadow_anchor_conf_hist": None,
            "shadow_edge_conf_hist": None,
            "shadow_hotspot_overlap": None,
        }
        carriers = {"enabled": False, "carriers": []}
    else:
        stats = shadow_pack["diagnostics"]
        carriers = {"enabled": True, "carriers": shadow_pack["carriers"]}
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(carriers_path, "w", encoding="utf-8") as f:
        json.dump(carriers, f, indent=2)