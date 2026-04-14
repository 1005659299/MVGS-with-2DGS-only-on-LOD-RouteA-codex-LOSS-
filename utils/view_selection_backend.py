from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VFWeights:
    first_hit: float = 1.0
    second_hit: float = 0.75
    region_balance: float = 0.25
    region_redundancy: float = 0.10
    base_gain: float = 0.50
    info_gain: float = 0.20
    overlap_pen: float = 0.10


@dataclass
class VFState:
    selected_mask: np.ndarray
    hit_count: np.ndarray
    owner: np.ndarray
    best_pair_quality: np.ndarray
    support_ok: np.ndarray
    voxel_hit_views: np.ndarray
    voxel_support_ok_count: np.ndarray


class ExactCSRSelectorBackend:
    def __init__(
        self,
        *,
        vis_pts_ptr: np.ndarray,
        vis_pts_idx: np.ndarray,
        vis_vox_ptr: np.ndarray,
        vis_vox_idx: np.ndarray,
        info: np.ndarray,
        overlap: np.ndarray,
        baseline: np.ndarray,
        voxel_pack,
        weights: VFWeights,
        tau_pair: float = 0.15,
        tau_ok: float = 0.50,
        tau_easy: float = 0.35,
        tau_hard: float = 0.60,
        tau_ok_hard: float = 0.70,
        enable_region: bool = False,
        enable_obs_cap: bool = False,
        enable_viewdep_guard: bool = False,
    ):
        self.vis_pts_ptr = np.asarray(vis_pts_ptr, dtype=np.int64)
        self.vis_pts_idx = np.asarray(vis_pts_idx, dtype=np.int64)
        self.vis_vox_ptr = np.asarray(vis_vox_ptr, dtype=np.int64)
        self.vis_vox_idx = np.asarray(vis_vox_idx, dtype=np.int64)
        self.info = np.asarray(info, dtype=np.float64)
        self.overlap = np.asarray(overlap, dtype=np.float64)
        self.baseline = np.asarray(baseline, dtype=np.float64)
        self.voxel_pack = voxel_pack
        self.weights = weights
        self.tau_pair = float(tau_pair)
        self.tau_ok = float(tau_ok)
        self.tau_easy = float(tau_easy)
        self.tau_hard = float(tau_hard)
        self.tau_ok_hard = float(tau_ok_hard)
        self.enable_region = bool(enable_region)
        self.enable_obs_cap = bool(enable_obs_cap)
        self.enable_viewdep_guard = bool(enable_viewdep_guard)
        self.info_max = float(np.max(self.info)) + 1e-12
        self.n_points = int(self.vis_pts_idx.max() + 1) if self.vis_pts_idx.size else 0
        self.n_voxels = int(self.voxel_pack.n_pts.shape[0])
        self.w_balance = 0.5 + 0.5 * np.asarray(self.voxel_pack.difficulty, dtype=np.float64)
        self.w_red = 1.0 - np.asarray(self.voxel_pack.difficulty, dtype=np.float64)

    def init_state(self) -> VFState:
        return VFState(
            selected_mask=np.zeros(self.info.shape[0], dtype=bool),
            hit_count=np.zeros(self.n_points, dtype=np.int16),
            owner=np.full(self.n_points, -1, dtype=np.int32),
            best_pair_quality=np.zeros(self.n_points, dtype=np.float32),
            support_ok=np.zeros(self.n_points, dtype=bool),
            voxel_hit_views=np.zeros(self.n_voxels, dtype=np.int16),
            voxel_support_ok_count=np.zeros(self.n_voxels, dtype=np.int32),
        )

    def _view_points(self, view_idx: int) -> np.ndarray:
        start = self.vis_pts_ptr[view_idx]
        end = self.vis_pts_ptr[view_idx + 1]
        return self.vis_pts_idx[start:end]

    def _view_voxels(self, view_idx: int) -> np.ndarray:
        start = self.vis_vox_ptr[view_idx]
        end = self.vis_vox_ptr[view_idx + 1]
        return self.vis_vox_idx[start:end]

    def _g_pair_quality(self, q: np.ndarray) -> np.ndarray:
        denom = max(1.0 - self.tau_pair, 1e-6)
        return np.clip((q - self.tau_pair) / denom, 0.0, 1.0)

    def current_voxel_support_ok_ratio(self, state: VFState) -> np.ndarray:
        return state.voxel_support_ok_count / np.maximum(self.voxel_pack.n_pts, 1)

    def current_target_hits(self, state: VFState) -> np.ndarray:
        obs_cap = np.asarray(self.voxel_pack.obs_cap, dtype=np.int32)
        desired_hits = np.full(self.n_voxels, 2, dtype=np.int32)
        if self.enable_obs_cap:
            support_ok_ratio = self.current_voxel_support_ok_ratio(state)
            hard_third = (
                np.asarray(self.voxel_pack.recoverable_hard, dtype=bool)
                & (obs_cap >= 3)
                & (support_ok_ratio < self.tau_ok_hard)
            )
            desired_hits[hard_third] = 3
        if self.enable_viewdep_guard:
            desired_hits[np.asarray(self.voxel_pack.viewdep_sink, dtype=bool)] = 2
        return np.minimum(desired_hits, np.maximum(obs_cap, 1))

    def current_easy_saturated(self, state: VFState) -> np.ndarray:
        target_hits = self.current_target_hits(state)
        support_ok_ratio = self.current_voxel_support_ok_ratio(state)
        return (
            (state.voxel_hit_views >= target_hits)
            & (support_ok_ratio >= self.tau_ok)
            & (np.asarray(self.voxel_pack.difficulty) < self.tau_easy)
        )

    def first_hit_gain(self, state: VFState, view_idx: int) -> float:
        pts = self._view_points(view_idx)
        if pts.size == 0:
            return 0.0
        return float(np.count_nonzero(state.hit_count[pts] == 0) / max(self.n_points, 1))

    def second_hit_eff(self, state: VFState, view_idx: int) -> float:
        pts = self._view_points(view_idx)
        if pts.size == 0:
            return 0.0
        one_hit_mask = state.hit_count[pts] == 1
        if not np.any(one_hit_mask):
            return 0.0
        owners = state.owner[pts[one_hit_mask]]
        raw_q = self.baseline[view_idx, owners]
        return float(np.mean(self._g_pair_quality(raw_q)))

    def region_balance_gain_delta(self, state: VFState, view_idx: int) -> float:
        if not (self.enable_region and self.enable_obs_cap):
            return 0.0
        voxels = self._view_voxels(view_idx)
        if voxels.size == 0:
            return 0.0
        target_hits = self.current_target_hits(state)
        eligible = (
            (np.asarray(self.voxel_pack.obs_cap)[voxels] >= 2)
            & (state.voxel_hit_views[voxels] < target_hits[voxels])
        )
        if not np.any(eligible):
            return 0.0
        return float(
            np.sum(
                self.w_balance[voxels][eligible]
                / np.maximum(target_hits[voxels][eligible], 1)
            )
        )

    def region_redundancy_pen(self, state: VFState, view_idx: int) -> float:
        if not (self.enable_region and self.enable_obs_cap):
            return 0.0
        voxels = self._view_voxels(view_idx)
        if voxels.size == 0:
            return 0.0
        easy = self.current_easy_saturated(state)
        if not np.any(easy[voxels]):
            return 0.0
        return float(np.sum(self.w_red[voxels][easy[voxels]]))

    def score_candidates_delta(self, state: VFState) -> np.ndarray:
        scores = np.full(self.info.shape[0], -np.inf, dtype=np.float64)
        selected = np.flatnonzero(state.selected_mask)
        for view_idx in range(self.info.shape[0]):
            if state.selected_mask[view_idx]:
                continue
            first_hit = self.first_hit_gain(state, view_idx)
            second_hit = self.second_hit_eff(state, view_idx)
            region_balance = self.region_balance_gain_delta(state, view_idx)
            region_red = self.region_redundancy_pen(state, view_idx)
            if selected.size:
                base_gain = float(np.max(self.baseline[view_idx, selected]))
                ov_pen = float(np.max(self.overlap[view_idx, selected]))
            else:
                base_gain = 0.0
                ov_pen = 0.0
            info_val = float(self.info[view_idx] / self.info_max)
            scores[view_idx] = (
                self.weights.first_hit * first_hit
                + self.weights.second_hit * second_hit
                + self.weights.region_balance * region_balance
                - self.weights.region_redundancy * region_red
                + self.weights.base_gain * base_gain
                + self.weights.info_gain * info_val
                - self.weights.overlap_pen * ov_pen
            )
        return scores

    def apply_selection_delta(self, state: VFState, view_idx: int) -> VFState:
        pts = self._view_points(view_idx)
        if pts.size:
            prev_support = state.support_ok[pts].copy()
            zeros = state.hit_count[pts] == 0
            ones_or_more = ~zeros
            state.owner[pts[zeros]] = view_idx
            state.hit_count[pts[zeros]] = 1
            if np.any(ones_or_more):
                owners = state.owner[pts[ones_or_more]]
                raw_q = self.baseline[view_idx, owners]
                state.best_pair_quality[pts[ones_or_more]] = np.maximum(
                    state.best_pair_quality[pts[ones_or_more]],
                    raw_q.astype(np.float32),
                )
                state.hit_count[pts[ones_or_more]] += 1
            state.support_ok[pts] = (
                (state.hit_count[pts] >= 2)
                & (state.best_pair_quality[pts] >= self.tau_pair)
            )
            became_support = state.support_ok[pts] & (~prev_support)
            if np.any(became_support):
                support_voxels = np.asarray(self.voxel_pack.pt2voxel)[pts[became_support]]
                np.add.at(state.voxel_support_ok_count, support_voxels, 1)

        voxels = self._view_voxels(view_idx)
        if voxels.size:
            state.voxel_hit_views[voxels] += 1
        state.selected_mask[view_idx] = True
        return state