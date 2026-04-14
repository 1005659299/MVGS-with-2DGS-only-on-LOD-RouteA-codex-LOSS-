from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch


def _ensure_2d(mask: torch.Tensor | None, height: int, width: int, device) -> torch.Tensor:
    if mask is None:
        return torch.ones((height, width), dtype=torch.float32, device=device)
    mask = mask.to(device=device, dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return mask


def _project_points(camera, xyz_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = camera.full_proj_transform.device
    pts = torch.as_tensor(xyz_world, dtype=torch.float32, device=device)
    ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device)
    pts_h = torch.cat([pts, ones], dim=1)
    clip = pts_h @ camera.full_proj_transform
    w = clip[:, 3].clamp_min(1e-6)
    ndc = clip[:, :3] / w[:, None]
    inside = (
        (w > 0)
        & (ndc[:, 0] >= -1)
        & (ndc[:, 0] <= 1)
        & (ndc[:, 1] >= -1)
        & (ndc[:, 1] <= 1)
    )
    x = ((ndc[:, 0] + 1.0) * 0.5 * (camera.image_width - 1)).round().long()
    y = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * (camera.image_height - 1)).round().long()
    return torch.stack([x, y], dim=1).detach().cpu().numpy(), inside.detach().cpu().numpy()


def _draw_disks(height: int, width: int, points_xy: np.ndarray, radius: int, device) -> torch.Tensor:
    mask = torch.zeros((height, width), dtype=torch.float32, device=device)
    if points_xy.size == 0:
        return mask
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    for x, y in points_xy.tolist():
        x0 = max(int(x) - radius, 0)
        x1 = min(int(x) + radius + 1, width)
        y0 = max(int(y) - radius, 0)
        y1 = min(int(y) + radius + 1, height)
        sub_x = xx[y0:y1, x0:x1]
        sub_y = yy[y0:y1, x0:x1]
        disk = ((sub_x - int(x)) ** 2 + (sub_y - int(y)) ** 2) <= radius ** 2
        mask[y0:y1, x0:x1] = torch.maximum(mask[y0:y1, x0:x1], disk.float())
    return mask


def project_voxel_risk_to_image(viewpoint_camera, support_pack, radius: int = 8) -> dict:
    device = viewpoint_camera.original_image.device
    h = int(viewpoint_camera.image_height)
    w = int(viewpoint_camera.image_width)
    if support_pack is None or support_pack.voxel_center.size == 0:
        zero = torch.zeros((h, w), dtype=torch.float32, device=device)
        return {"under_target": zero, "intrinsic_low_obs": zero, "shadow": zero, "shadow_meta": []}
    xy, inside = _project_points(viewpoint_camera, support_pack.voxel_center)
    current_hits = support_pack.final_voxel_hit_views if support_pack.final_voxel_hit_views.size else np.zeros_like(support_pack.target_hits)
    under_target = (
        (support_pack.obs_cap >= support_pack.target_hits)
        & (current_hits < support_pack.target_hits)
        & (~support_pack.viewdep_sink)
    )
    intrinsic_low_obs = (support_pack.obs_cap < 2) & (~support_pack.viewdep_sink)
    risk_masks = {}
    risk_masks["under_target"] = _draw_disks(h, w, xy[inside & under_target], radius, device)
    risk_masks["intrinsic_low_obs"] = _draw_disks(h, w, xy[inside & intrinsic_low_obs], radius, device)
    shadow = torch.zeros((h, w), dtype=torch.float32, device=device)
    shadow_meta = []
    if getattr(support_pack, "shadow_bboxes", np.zeros((0, 4))).size:
        shadow_bboxes = support_pack.shadow_bboxes
        shadow_count = int(len(shadow_bboxes))
        shadow_view_names = getattr(support_pack, "shadow_view_names", np.asarray([""] * shadow_count, dtype=object))
        shadow_obs_cap = getattr(support_pack, "shadow_obs_cap", np.ones(shadow_count, dtype=np.int32))
        shadow_hit_views = getattr(support_pack, "shadow_hit_views", np.zeros(shadow_count, dtype=np.int32))
        shadow_anchor_conf = getattr(support_pack, "shadow_anchor_conf", np.zeros(shadow_count, dtype=np.float32))
        shadow_edge_conf = getattr(support_pack, "shadow_edge_conf", np.zeros(shadow_count, dtype=np.float32))
        shadow_ring = getattr(support_pack, "shadow_ring_anchorable", np.zeros(shadow_count, dtype=bool))
        shadow_local_depth_cont = getattr(support_pack, "shadow_local_depth_cont", np.zeros(shadow_count, dtype=bool))
        shadow_z_nb = getattr(support_pack, "shadow_z_nb", np.full(shadow_count, np.nan, dtype=np.float32))
        shadow_trusted_prior = getattr(support_pack, "shadow_trusted_prior", np.zeros(shadow_count, dtype=bool))
        for shadow_idx in range(shadow_count):
            view_name = str(shadow_view_names[shadow_idx]) if shadow_idx < len(shadow_view_names) else ""
            if str(view_name) != viewpoint_camera.image_name:
                continue
            bbox = shadow_bboxes[shadow_idx]
            shadow_obs = int(shadow_obs_cap[shadow_idx]) if shadow_idx < len(shadow_obs_cap) else 1
            shadow_hit = int(shadow_hit_views[shadow_idx]) if shadow_idx < len(shadow_hit_views) else 0
            anchor_conf = float(shadow_anchor_conf[shadow_idx]) if shadow_idx < len(shadow_anchor_conf) else 0.0
            edge_conf = float(shadow_edge_conf[shadow_idx]) if shadow_idx < len(shadow_edge_conf) else 0.0
            ring_anchorable = bool(shadow_ring[shadow_idx]) if shadow_idx < len(shadow_ring) else False
            local_depth_cont = bool(shadow_local_depth_cont[shadow_idx]) if shadow_idx < len(shadow_local_depth_cont) else False
            z_nb = float(shadow_z_nb[shadow_idx]) if shadow_idx < len(shadow_z_nb) else float("nan")
            trusted_prior = bool(shadow_trusted_prior[shadow_idx]) if shadow_idx < len(shadow_trusted_prior) else False
            shadow_need = shadow_hit < min(2, max(shadow_obs, 1))
            shadow_meta.append(
                {
                    "shadow_idx": shadow_idx,
                    "bbox": [int(v) for v in bbox],
                    "shadow_obs_cap": shadow_obs,
                    "shadow_hit_views": shadow_hit,
                    "shadow_anchor_conf": anchor_conf,
                    "shadow_edge_conf": edge_conf,
                    "ring_anchorable": ring_anchorable,
                    "local_depth_cont": local_depth_cont,
                    "z_nb": None if np.isnan(z_nb) else z_nb,
                    "trusted_prior_shadow_offline": trusted_prior,
                    "shadow_need": shadow_need,
                }
            )
            if not shadow_need or not ring_anchorable:
                continue
            if anchor_conf < 0.50 or edge_conf < 0.60:
                continue
            y0, x0, y1, x1 = [int(v) for v in bbox]
            y0 = max(y0, 0)
            x0 = max(x0, 0)
            y1 = min(y1, h)
            x1 = min(x1, w)
            if y1 > y0 and x1 > x0:
                shadow[y0:y1, x0:x1] = 1.0
    risk_masks["shadow"] = shadow
    risk_masks["shadow_meta"] = shadow_meta
    return risk_masks


def build_hotspot_mask(image_like: torch.Tensor, box=None) -> torch.Tensor:
    if image_like.ndim == 3:
        _, h, w = image_like.shape
        device = image_like.device
    else:
        h, w = image_like.shape[-2:]
        device = image_like.device
    mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    if box is None:
        return mask
    y0, x0, y1, x1 = [int(v) for v in box]
    y0 = max(y0, 0)
    x0 = max(x0, 0)
    y1 = min(y1, h)
    x1 = min(x1, w)
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = 1.0
    return mask


def build_comp_mask(
    *,
    hotspot_mask: torch.Tensor,
    risk_mask: torch.Tensor,
    render_alpha: torch.Tensor,
    prior_valid_mask: torch.Tensor | None = None,
    static_mask: torch.Tensor | None = None,
    edge_safe_mask: torch.Tensor | None = None,
    tau_alpha: float = 0.05,
) -> torch.Tensor:
    if render_alpha.ndim == 3:
        render_alpha_2d = render_alpha.squeeze(0)
    else:
        render_alpha_2d = render_alpha
    h, w = render_alpha_2d.shape[-2:]
    device = render_alpha_2d.device
    prior_valid = _ensure_2d(prior_valid_mask, h, w, device)
    static_safe = _ensure_2d(static_mask, h, w, device)
    edge_safe = _ensure_2d(edge_safe_mask, h, w, device)
    mask = hotspot_mask.to(device) * risk_mask.to(device)
    mask = mask * static_safe * edge_safe
    mask = mask * (render_alpha_2d > tau_alpha).float()
    mask = mask * prior_valid
    return mask


@dataclass
class HotspotTracker:
    model_path: str
    history: list = field(default_factory=list)

    def update(self, *, iteration: int, hotspot_mask: torch.Tensor, risk_masks: dict) -> dict:
        hotspot = hotspot_mask > 0
        hotspot_pixels = float(torch.count_nonzero(hotspot).item())
        overlap_under = float(torch.count_nonzero(hotspot & (risk_masks["under_target"] > 0)).item())
        overlap_low = float(torch.count_nonzero(hotspot & (risk_masks["intrinsic_low_obs"] > 0)).item())
        overlap_shadow = float(torch.count_nonzero(hotspot & (risk_masks["shadow"] > 0)).item())
        denom = max(hotspot_pixels, 1.0)
        record = {
            "iteration": int(iteration),
            "hotspot_under_target_overlap": overlap_under / denom,
            "hotspot_intrinsic_low_obs_overlap": overlap_low / denom,
            "shadow_hotspot_overlap": overlap_shadow / denom,
        }
        self.history.append(record)
        return {
            "summary": compute_hotspot_overlap_stats(self.history),
            "latest": record,
            "history": self.history[-32:],
        }


def compute_hotspot_overlap_stats(history: list) -> dict:
    if not history:
        return {
            "hotspot_under_target_overlap": 0.0,
            "hotspot_intrinsic_low_obs_overlap": 0.0,
            "shadow_hotspot_overlap": None,
        }
    latest = history[-1]
    return {
        "hotspot_under_target_overlap": latest["hotspot_under_target_overlap"],
        "hotspot_intrinsic_low_obs_overlap": latest["hotspot_intrinsic_low_obs_overlap"],
        "shadow_hotspot_overlap": latest["shadow_hotspot_overlap"],
    }


def write_online_diagnostics(model_path: str, tracker_payload: dict, densify_stats: dict | None = None) -> None:
    diag_dir = os.path.join(model_path, "coverage_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    hotspot_path = os.path.join(diag_dir, "hotspot_overlap_history.json")
    with open(hotspot_path, "w", encoding="utf-8") as f:
        json.dump(tracker_payload, f, indent=2)
    shadow_overlap = tracker_payload.get("summary", {}).get("shadow_hotspot_overlap")
    shadow_stats_path = os.path.join(diag_dir, "zero_point_shadow_stats.json")
    if shadow_overlap is not None and os.path.exists(shadow_stats_path):
        with open(shadow_stats_path, "r", encoding="utf-8") as f:
            shadow_stats = json.load(f)
        shadow_stats["shadow_hotspot_overlap"] = shadow_overlap
        with open(shadow_stats_path, "w", encoding="utf-8") as f:
            json.dump(shadow_stats, f, indent=2)
    if densify_stats is not None:
        densify_path = os.path.join(diag_dir, "training_densify_stats.json")
        with open(densify_path, "w", encoding="utf-8") as f:
            json.dump(densify_stats, f, indent=2)