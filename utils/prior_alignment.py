from __future__ import annotations

from dataclasses import dataclass

import torch


EPS = 1e-6


@dataclass
class PriorProvider:
    prior_type: str = "none"

    def __call__(self, viewpoint_camera) -> dict:
        _ = viewpoint_camera
        if self.prior_type == "none":
            return {
                "depth_prior": None,
                "conf_prior": None,
                "prior_type": "none",
                "enabled": False,
            }
        return {
            "depth_prior": None,
            "conf_prior": None,
            "valid_mask": None,
            "prior_type": self.prior_type,
            "enabled": False,
            "reason": "provider_not_wired",
        }


def build_prior_provider(prior_type: str) -> PriorProvider:
    return PriorProvider(prior_type=prior_type)


def _flatten_masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values is None:
        return torch.zeros(0, device=mask.device)
    values = values.squeeze()
    return values[mask]


def solve_scale_shift_wls(
    prior_depth: torch.Tensor,
    sfm_depth: torch.Tensor,
    anchor_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    trim_top_ratio: float = 0.10,
) -> dict:
    valid = anchor_mask.bool()
    if valid.ndim == 3:
        valid = valid.squeeze(0)
    if not torch.any(valid):
        return {
            "scale": 1.0,
            "shift": 0.0,
            "aligned_depth": prior_depth,
            "align_rel_err": float("inf"),
            "n_anchor": 0,
            "trusted_prior_view": False,
        }
    prior_vals = _flatten_masked(prior_depth, valid).float()
    sfm_vals = _flatten_masked(sfm_depth, valid).float()
    if weights is None:
        w = torch.ones_like(prior_vals)
    else:
        w = _flatten_masked(weights, valid).float().clamp_min(EPS)
    W = torch.sum(w).clamp_min(EPS)
    mu_p = torch.sum(w * prior_vals) / W
    mu_s = torch.sum(w * sfm_vals) / W
    numerator = torch.sum(w * (prior_vals - mu_p) * (sfm_vals - mu_s))
    denominator = torch.sum(w * (prior_vals - mu_p) ** 2).clamp_min(EPS)
    scale = torch.clamp(numerator / denominator, min=EPS)
    shift = mu_s - scale * mu_p
    aligned_samples = scale * prior_vals + shift
    residuals = torch.abs(aligned_samples - sfm_vals)
    if trim_top_ratio > 0.0 and residuals.numel() >= 16:
        keep = int(max(residuals.numel() * (1.0 - trim_top_ratio), 1))
        keep_idx = torch.argsort(residuals)[:keep]
        return solve_scale_shift_wls(
            prior_vals[keep_idx],
            sfm_vals[keep_idx],
            torch.ones_like(prior_vals[keep_idx], dtype=torch.bool),
            weights=w[keep_idx],
            trim_top_ratio=0.0,
        )
    aligned = scale * prior_depth + shift
    rel_err = torch.median(torch.abs(aligned_samples - sfm_vals) / sfm_vals.clamp_min(EPS))
    conf_median = float(torch.median(w).item()) if w.numel() else 1.0
    return {
        "scale": float(scale.item()),
        "shift": float(shift.item()),
        "aligned_depth": aligned,
        "align_rel_err": float(rel_err.item()),
        "n_anchor": int(prior_vals.numel()),
        "median_conf": conf_median,
        "trusted_prior_view": trusted_prior_view(
            n_anchor=int(prior_vals.numel()),
            align_rel_err=float(rel_err.item()),
            median_conf=conf_median,
        ),
    }


def trim_and_resolve(*args, **kwargs) -> dict:
    return solve_scale_shift_wls(*args, **kwargs)


def trusted_prior_view(
    *,
    n_anchor: int,
    align_rel_err: float,
    median_conf: float = 1.0,
    n_anchor_min: int = 50,
    tau_align: float = 0.20,
    tau_conf: float = 0.50,
) -> bool:
    return (
        n_anchor >= n_anchor_min
        and align_rel_err <= tau_align
        and median_conf >= tau_conf
    )


def trusted_prior_local(
    *,
    ring_anchorable: bool,
    shadow_edge_conf: float,
    trusted_view: bool,
    tau_edge_shadow: float = 0.60,
) -> bool:
    return bool(ring_anchorable and trusted_view and shadow_edge_conf >= tau_edge_shadow)