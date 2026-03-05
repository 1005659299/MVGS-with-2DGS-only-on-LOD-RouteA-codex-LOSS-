"""
SFC-FRS++ View Selection Module
Route A: COLMAP-based training view selection for sparse-view 2D-GS training.

Core objective function:
  F(S) = alpha * C_cov(S) + beta * C_base(S) + gamma * C_info(S) - delta * C_overlap(S)
"""

import numpy as np
import json
import os
import time
import struct


# ---------------------------------------------------------------------------
# COLMAP readers that preserve point3D_id (existing readers discard it)
# ---------------------------------------------------------------------------

def read_points3D_binary_full(path):
    """Read points3D.bin -> dict {point3D_id: xyz(3,)}"""
    def _read(fid, num_bytes, fmt, endian="<"):
        return struct.unpack(endian + fmt, fid.read(num_bytes))

    pts = {}
    with open(path, "rb") as fid:
        num_points = _read(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read(fid, 43, "QdddBBBd")
            point3D_id = props[0]
            xyz = np.array(props[1:4])
            track_length = _read(fid, 8, "Q")[0]
            _read(fid, 8 * track_length, "ii" * track_length)
            pts[point3D_id] = xyz
    return pts


def load_points3D_full(sparse_dir):
    """Load points3D.bin with IDs."""
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    return read_points3D_binary_full(bin_path)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def qvec2rotmat(qvec):
    """Quaternion -> 3x3 rotation matrix (self-contained, no colmap_loader dependency)."""
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def compute_camera_centers(cam_extrinsics):
    """Compute world-frame camera centers C = -R^T @ t for all images."""
    centers = {}
    for img_id, img in cam_extrinsics.items():
        R = qvec2rotmat(img.qvec)
        centers[img_id] = -R.T @ img.tvec
    return centers


# ---------------------------------------------------------------------------
# Precompute matrices (Layer A - one-time cost)
# ---------------------------------------------------------------------------

def build_visibility(cam_extrinsics, points3D_full, num_points_cap, rng):
    """Build per-image visibility sets and depth info.

    Returns:
        image_ids, id2idx, vis_sets, depths_list, pt_xyz, pid2idx
    """
    all_pids = sorted(points3D_full.keys())
    if len(all_pids) > num_points_cap:
        chosen_set = set(rng.choice(all_pids, num_points_cap, replace=False).tolist())
    else:
        chosen_set = set(all_pids)

    chosen_sorted = sorted(chosen_set)
    pid2idx = {pid: i for i, pid in enumerate(chosen_sorted)}
    pt_xyz = np.array([points3D_full[pid] for pid in chosen_sorted])

    image_ids = sorted(cam_extrinsics.keys())
    id2idx = {iid: i for i, iid in enumerate(image_ids)}

    vis_sets = []
    depths_list = []
    for iid in image_ids:
        img = cam_extrinsics[iid]
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        vis = set()
        dep = {}
        for pid_raw in img.point3D_ids:
            pid = int(pid_raw)
            if pid < 0 or pid not in pid2idx:
                continue
            pt_idx = pid2idx[pid]
            z = float(R[2, :] @ pt_xyz[pt_idx] + t[2])
            if z > 0:
                vis.add(pt_idx)
                dep[pt_idx] = z
        vis_sets.append(vis)
        depths_list.append(dep)

    return image_ids, id2idx, vis_sets, depths_list, pt_xyz, pid2idx


def precompute_matrices(image_ids, cam_extrinsics, vis_sets, depths_list,
                        pt_xyz, shared_sample, rng):
    """Precompute Info, Overlap, Baseline matrices.

    Returns: Info(N,), Overlap(N,N), Baseline(N,N)
    """
    N = len(image_ids)
    eps = 1e-6

    # Info[k] = sum 1/(z^2 + eps)
    Info = np.zeros(N)
    for k in range(N):
        for pt_idx, z in depths_list[k].items():
            Info[k] += 1.0 / (z * z + eps)

    # Overlap[k,l] = Jaccard(V_k, V_l)
    Overlap = np.zeros((N, N))
    for k in range(N):
        for l in range(k + 1, N):
            inter = len(vis_sets[k] & vis_sets[l])
            union = len(vis_sets[k] | vis_sets[l])
            if union > 0:
                val = inter / union
                Overlap[k, l] = Overlap[l, k] = val

    # Baseline[k,l] = E[sin^2(phi)] over shared points
    _all_centers = compute_camera_centers(cam_extrinsics)   # call once, reuse result
    centers = np.array([_all_centers[iid] for iid in image_ids])
    Baseline = np.zeros((N, N))
    for k in range(N):
        for l in range(k + 1, N):
            shared = list(vis_sets[k] & vis_sets[l])
            if not shared:
                continue
            if len(shared) > shared_sample:
                shared = rng.choice(shared, shared_sample, replace=False).tolist()
            sin2_vals = []
            for pt_idx in shared:
                p = pt_xyz[pt_idx]
                dk = p - centers[k]
                dl = p - centers[l]
                cos_phi = np.dot(dk, dl) / (np.linalg.norm(dk) * np.linalg.norm(dl) + eps)
                cos_phi = np.clip(cos_phi, -1.0, 1.0)
                sin2_vals.append(1.0 - cos_phi ** 2)
            val = np.mean(sin2_vals)
            Baseline[k, l] = Baseline[l, k] = val

    return Info, Overlap, Baseline


# ---------------------------------------------------------------------------
# Greedy selection (Layer B - K iterations of table lookups)
# ---------------------------------------------------------------------------

def greedy_select(K, N, vis_sets, Info, Overlap, Baseline,
                  alpha, beta, gamma, delta, num_points_total):
    """Greedy SFC-FRS++ selection.

    Returns: (selected_indices, meta_dict)
    """
    covered = set()
    selected = []
    info_max = Info.max() + 1e-12

    for step in range(K):
        best_score = -np.inf
        best_idx = -1
        for k in range(N):
            if k in selected:
                continue
            cov_gain = len(vis_sets[k] - covered) / max(num_points_total, 1)
            base_gain = max((Baseline[k, s] for s in selected), default=0.0)
            info_val = Info[k] / info_max
            ov_pen = max((Overlap[k, s] for s in selected), default=0.0)
            score = alpha * cov_gain + beta * base_gain + gamma * info_val - delta * ov_pen
            if score > best_score:
                best_score = score
                best_idx = k
        selected.append(best_idx)
        covered |= vis_sets[best_idx]

    return selected, {
        "coverage": len(covered) / max(num_points_total, 1),
        "num_selected": len(selected),
        "num_points_used": num_points_total,
    }


def select_views_uniform_pose(cam_extrinsics, image_ids, K, rng):
    """Farthest-point sampling in camera center space."""
    centers = compute_camera_centers(cam_extrinsics)
    pts = np.array([centers[iid] for iid in image_ids])
    N = len(pts)
    assert K < N, f"[ViewSelect] K={K} >= N={N} in uniform_pose selection"
    selected = [rng.randint(N)]
    dists = np.full(N, np.inf)
    for _ in range(K - 1):
        last = pts[selected[-1]]
        for i in range(N):
            d = np.linalg.norm(pts[i] - last)
            dists[i] = min(dists[i], d)
        for s in selected:
            dists[s] = -1
        selected.append(int(np.argmax(dists)))
    return selected


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def select_views(cam_extrinsics, points3D_full, K, strategy,
                 alpha=1.0, beta=0.5, gamma=0.2, delta=0.1,
                 num_points=50000, shared_sample=200, seed=42):
    """Main view selection entry point.

    Args:
        cam_extrinsics: dict of COLMAP Image namedtuples
        points3D_full: dict {point3D_id: xyz}
        K: target number of training views
        strategy: 'sfc_frs_greedy' | 'frs_greedy' | 'uniform_pose' | 'random'

    Returns:
        selected_image_ids, selected_names (stems), meta dict
    """
    rng = np.random.RandomState(seed)
    t0 = time.time()

    image_ids, id2idx, vis_sets, depths_list, pt_xyz, pid2idx = \
        build_visibility(cam_extrinsics, points3D_full, num_points, rng)

    N = len(image_ids)
    # fail-fast: K>=N indicates invalid setting
    assert K < N, (
        f"[ViewSelect] FATAL: K={K} >= N={N} train candidates. "
        f"Reduce --k or check --llffhold split settings."
    )
    if strategy == "random":
        sel = sorted(rng.choice(N, K, replace=False).tolist())
        meta = {"strategy": "random", "K": K, "N": N}
    elif strategy == "uniform_pose":
        sel = select_views_uniform_pose(cam_extrinsics, image_ids, K, rng)
        meta = {"strategy": "uniform_pose", "K": K, "N": N}
    elif strategy in ("sfc_frs_greedy", "frs_greedy"):
        Info, Overlap, Baseline = precompute_matrices(
            image_ids, cam_extrinsics, vis_sets, depths_list,
            pt_xyz, shared_sample, rng)
        d = delta if strategy == "sfc_frs_greedy" else 0.0
        sel, meta = greedy_select(K, N, vis_sets, Info, Overlap, Baseline,
                                  alpha, beta, gamma, d, len(pt_xyz))
        meta.update({"strategy": strategy, "alpha": alpha, "beta": beta,
                     "gamma": gamma, "delta": d, "N": N, "K": K})
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    meta["elapsed_seconds"] = round(time.time() - t0, 2)
    sel_ids = [image_ids[i] for i in sel]
    sel_names = [os.path.basename(cam_extrinsics[iid].name).split(".")[0] for iid in sel_ids]
    return sel_ids, sel_names, meta


def save_selection(model_path, selected_names, meta):
    """Save selection results to model_path directory."""
    os.makedirs(model_path, exist_ok=True)
    jp = os.path.join(model_path, "selected_views.json")
    with open(jp, "w") as f:
        json.dump(selected_names, f, indent=2)
    tp = os.path.join(model_path, "selected_train_views.txt")
    with open(tp, "w") as f:
        for n in selected_names:
            f.write(n + "\n")
    mp = os.path.join(model_path, "view_select_meta.json")
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2)
    return jp, tp, mp
