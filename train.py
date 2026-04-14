#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import sys
import time
import uuid
from random import randint

import torch
from tqdm import tqdm

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.camera_utils import set_rays_od
from utils.densify_guard import (
    build_densify_guard,
    compute_current_voxel_support_ok_ratio,
    rehash_gaussians_to_voxels,
    summarize_densify_stats,
)
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.prior_alignment import build_prior_provider, solve_scale_shift_wls
from utils.risk_projection import (
    HotspotTracker,
    build_comp_mask,
    build_hotspot_mask,
    project_voxel_risk_to_image,
    write_online_diagnostics,
)
from utils.voxel_support import load_vf_support_pack
from utils.zero_point_shadow import trusted_prior_shadow
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _validate_runtime_options(opt):
    if opt.vf_schedule_branch not in {"Q", "T1", "T2"}:
        raise ValueError(f"Unsupported vf_schedule_branch: {opt.vf_schedule_branch}")
    if opt.prior_type not in {"none", "dav2_rel", "dav2_metric", "other"}:
        raise ValueError(f"Unsupported prior_type: {opt.prior_type}")


def _lod_resolution_factor(resolution):
    if resolution in (-1, 1):
        return 1
    return int(resolution)


def lambda_depth_local(iteration, opt):
    if iteration < opt.prior_t_on:
        return 0.0
    if iteration >= opt.prior_t_full:
        return opt.lambda_depth_local_max
    alpha = (iteration - opt.prior_t_on) / max(opt.prior_t_full - opt.prior_t_on, 1)
    return float(alpha * opt.lambda_depth_local_max)


def local_depth_loss(surf_depth, prior_depth_aligned, comp_mask):
    eps = 1e-6
    surf = torch.log(surf_depth.clamp_min(eps))
    prior = torch.log(prior_depth_aligned.clamp_min(eps))
    residual = torch.sqrt((surf - prior) ** 2 + 1e-6)
    if comp_mask.ndim == 2:
        comp_mask = comp_mask.unsqueeze(0)
    denom = comp_mask.sum().clamp_min(eps)
    return (residual * comp_mask).sum() / denom


def _project_sparse_anchors(viewpoint_camera, vf_support_pack):
    if vf_support_pack is None or vf_support_pack.pt_xyz.size == 0:
        device = viewpoint_camera.original_image.device
        h = int(viewpoint_camera.image_height)
        w = int(viewpoint_camera.image_width)
        return {
            "depth_map": torch.zeros((1, h, w), device=device),
            "anchor_mask": torch.zeros((h, w), dtype=torch.bool, device=device),
        }
    device = viewpoint_camera.original_image.device
    pts = torch.as_tensor(vf_support_pack.pt_xyz, dtype=torch.float32, device=device)
    ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device)
    pts_h = torch.cat([pts, ones], dim=1)
    clip = pts_h @ viewpoint_camera.full_proj_transform
    w_comp = clip[:, 3].clamp_min(1e-6)
    ndc = clip[:, :3] / w_comp[:, None]
    inside = (
        (w_comp > 0)
        & (ndc[:, 0] >= -1)
        & (ndc[:, 0] <= 1)
        & (ndc[:, 1] >= -1)
        & (ndc[:, 1] <= 1)
    )
    x = ((ndc[:, 0] + 1.0) * 0.5 * (viewpoint_camera.image_width - 1)).round().long()
    y = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * (viewpoint_camera.image_height - 1)).round().long()
    depth = clip[:, 2] / w_comp
    h = int(viewpoint_camera.image_height)
    w = int(viewpoint_camera.image_width)
    depth_map = torch.zeros((h, w), dtype=torch.float32, device=device)
    anchor_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    valid_idx = torch.where(inside)[0]
    for idx in valid_idx.tolist():
        xi = int(x[idx].item())
        yi = int(y[idx].item())
        if 0 <= xi < w and 0 <= yi < h:
            if not anchor_mask[yi, xi] or depth[idx] < depth_map[yi, xi]:
                depth_map[yi, xi] = depth[idx]
                anchor_mask[yi, xi] = True
    return {"depth_map": depth_map.unsqueeze(0), "anchor_mask": anchor_mask}


def _find_hotspot_box(rendered_image, gt_image, w_box=40, h_box=20):
    l1map = torch.abs(gt_image - rendered_image)
    _, h, w = l1map.shape
    loss_max = torch.tensor(0.0, device=l1map.device)
    max_id = [0, 0]
    for i in range(0, max(h - h_box + 1, 1), h_box):
        for j in range(0, max(w - w_box + 1, 1), w_box):
            region = l1map[:, i:min(i + h_box, h), j:min(j + w_box, w)]
            loss_region = torch.mean(region)
            if loss_region > loss_max:
                loss_max = loss_region
                max_id = [i, j]
    i, j = max_id
    return [i, j, min(i + h_box, h), min(j + w_box, w)], float(loss_max.item())


def _maybe_prior_loss(iteration, opt, viewpoint_cam, render_pkg, hotspot_box, vf_support_pack, prior_provider):
    if not opt.vf_enable_prior or vf_support_pack is None or iteration < opt.prior_t_on:
        return None
    prior_pkg = prior_provider(viewpoint_cam)
    if not prior_pkg.get("enabled", False):
        return None
    prior_depth = prior_pkg.get("depth_prior")
    if prior_depth is None:
        return None
    anchor_pkg = _project_sparse_anchors(viewpoint_cam, vf_support_pack)
    align_pkg = solve_scale_shift_wls(
        prior_depth=prior_depth,
        sfm_depth=anchor_pkg["depth_map"],
        anchor_mask=anchor_pkg["anchor_mask"],
        weights=prior_pkg.get("conf_prior"),
    )
    risk_masks = project_voxel_risk_to_image(viewpoint_cam, vf_support_pack)
    hotspot_mask = build_hotspot_mask(render_pkg["render"], hotspot_box)
    prior_valid = prior_pkg.get("valid_mask")
    if prior_valid is None:
        prior_valid = torch.ones_like(hotspot_mask)
    base_risk_mask = torch.maximum(risk_masks["under_target"], risk_masks["intrinsic_low_obs"])
    base_comp_mask = build_comp_mask(
        hotspot_mask=hotspot_mask,
        risk_mask=base_risk_mask,
        render_alpha=render_pkg["rend_alpha"],
        prior_valid_mask=prior_valid,
        static_mask=viewpoint_cam.vf_static_mask,
        edge_safe_mask=viewpoint_cam.vf_edge_safe_mask,
        tau_alpha=opt.tau_comp_alpha,
    )
    base_comp_area = int(torch.count_nonzero(base_comp_mask).item())
    trusted_local = bool(align_pkg["trusted_prior_view"] and base_comp_area > 32)
    shadow_comp_mask = torch.zeros_like(base_comp_mask)
    shadow_comp_area = 0
    trusted_shadow = False
    shadow_records = []
    if opt.vf_enable_shadow_guard:
        for shadow_info in risk_masks.get("shadow_meta", []):
            shadow_need = bool(shadow_info.get("shadow_need", True))
            if not shadow_need:
                shadow_records.append(
                    {
                        "shadow_idx": shadow_info.get("shadow_idx", -1),
                        "bbox": shadow_info["bbox"],
                        "shadow_need": False,
                        "trusted_prior_shadow": False,
                        "comp_mask_area": 0,
                    }
                )
                continue
            shadow_local_ok = bool(shadow_info.get("local_depth_cont", False)) and trusted_prior_shadow(
                shadow_info,
                align_pkg["trusted_prior_view"],
            )
            bbox_mask = build_hotspot_mask(render_pkg["render"], shadow_info["bbox"])
            shadow_local_mask = build_comp_mask(
                hotspot_mask=hotspot_mask,
                risk_mask=bbox_mask,
                render_alpha=render_pkg["rend_alpha"],
                prior_valid_mask=prior_valid,
                static_mask=viewpoint_cam.vf_static_mask,
                edge_safe_mask=viewpoint_cam.vf_edge_safe_mask,
                tau_alpha=opt.tau_comp_alpha,
            )
            local_area = int(torch.count_nonzero(shadow_local_mask).item())
            shadow_records.append(
                {
                    "shadow_idx": shadow_info.get("shadow_idx", -1),
                    "bbox": shadow_info["bbox"],
                    "shadow_need": True,
                    "trusted_prior_shadow": bool(shadow_local_ok and local_area > 32),
                    "comp_mask_area": local_area,
                }
            )
            if shadow_local_ok and local_area > 32:
                shadow_idx = int(shadow_info.get("shadow_idx", -1))
                if 0 <= shadow_idx < len(getattr(vf_support_pack, "shadow_hit_views", [])):
                    vf_support_pack.shadow_hit_views[shadow_idx] = max(
                        int(vf_support_pack.shadow_hit_views[shadow_idx]),
                        1,
                    )
                shadow_comp_mask = torch.maximum(shadow_comp_mask, shadow_local_mask)
                trusted_shadow = True
    shadow_comp_area = int(torch.count_nonzero(shadow_comp_mask).item())
    comp_mask = torch.maximum(base_comp_mask, shadow_comp_mask)
    comp_mask_area = int(torch.count_nonzero(comp_mask).item())
    if (trusted_local or trusted_shadow) and comp_mask_area > 32:
        weight = lambda_depth_local(iteration, opt)
        if weight <= 0.0:
            return None
        loss = weight * local_depth_loss(
            render_pkg["surf_depth"],
            align_pkg["aligned_depth"],
            comp_mask,
        )
        return {
            "loss": loss,
            "align_pkg": align_pkg,
            "hotspot_mask": hotspot_mask,
            "risk_masks": risk_masks,
            "comp_mask_area": comp_mask_area,
            "base_comp_mask_area": base_comp_area,
            "shadow_comp_mask_area": shadow_comp_area,
            "trusted_prior_local": trusted_local,
            "trusted_prior_shadow": trusted_shadow,
            "shadow_records": shadow_records,
        }
    return {
        "loss": None,
        "align_pkg": align_pkg,
        "hotspot_mask": hotspot_mask,
        "risk_masks": risk_masks,
        "comp_mask_area": comp_mask_area,
        "base_comp_mask_area": base_comp_area,
        "shadow_comp_mask_area": shadow_comp_area,
        "trusted_prior_local": trusted_local,
        "trusted_prior_shadow": trusted_shadow,
        "shadow_records": shadow_records,
    }


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    _validate_runtime_options(opt)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        print("loading ckpt", checkpoint)

        print("resolution: ", dataset.resolution, "MV:", pipe.mv)
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        first_iter = 0

    vf_support_pack = None
    if getattr(dataset, "vf_support_pack", "") and os.path.exists(dataset.vf_support_pack):
        vf_support_pack = load_vf_support_pack(dataset.vf_support_pack)
    hotspot_tracker = HotspotTracker(model_path=dataset.model_path)
    prior_provider = build_prior_provider(opt.prior_type)
    prior_stats = {"enabled": bool(opt.vf_enable_prior), "records": []}
    densify_diag_payload = {
        "enabled": bool(opt.vf_enable_densify_guard),
        "history": [],
        "training_wall_clock_sec": 0.0,
        "view_render_budget_total": 0,
        "final_gaussian_count": 0,
        "densify_births_by_support_class": {},
        "densify_reject_ratio_by_support_class": {},
        "mv_densify_stat_view_count_mean": 0.0,
        "spiky_gaussian_ratio": 0.0,
    }
    train_start_time = time.time()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    set_rays_od(scene.getTrainCameras())

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        total_loss = 0.0
        densify_pkgs = []
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        for _ in range(pipe.mv):
            # Pick a random Camera
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, rng_step=iteration)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            photometric_loss = loss

            # [2D-GS] Geometric regularization losses with warm-up
            lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

            rend_dist = render_pkg["rend_dist"]
            rend_normal = render_pkg["rend_normal"]
            surf_normal = render_pkg["surf_normal"]
            rend_alpha = render_pkg["rend_alpha"].detach().clamp(0.0, 1.0)
            alpha_denom = rend_alpha.sum() + 1e-6

            # Normal Consistency Loss (masked)
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0, keepdim=True))
            normal_loss = lambda_normal * (normal_error * rend_alpha).sum() / alpha_denom

            # Depth Distortion Loss (masked)
            dist_loss = lambda_dist * (rend_dist * rend_alpha).sum() / alpha_denom

            hotspot_box, hotspot_loss = _find_hotspot_box(image, gt_image)
            total_loss += photometric_loss + dist_loss + normal_loss
            prior_payload = _maybe_prior_loss(
                iteration,
                opt,
                viewpoint_cam,
                render_pkg,
                hotspot_box,
                vf_support_pack,
                prior_provider,
            )
            if prior_payload is not None and prior_payload["loss"] is not None:
                total_loss = total_loss + prior_payload["loss"]
                prior_stats["records"].append(
                    {
                        "iteration": int(iteration),
                        "view_name": viewpoint_cam.image_name,
                        "align_rel_err": prior_payload["align_pkg"]["align_rel_err"],
                        "n_anchor": prior_payload["align_pkg"]["n_anchor"],
                        "trusted_prior_view": prior_payload["align_pkg"]["trusted_prior_view"],
                        "trusted_prior_local": prior_payload["trusted_prior_local"],
                        "trusted_prior_shadow": prior_payload["trusted_prior_shadow"],
                        "comp_mask_area": prior_payload["comp_mask_area"],
                        "base_comp_mask_area": prior_payload["base_comp_mask_area"],
                        "shadow_comp_mask_area": prior_payload["shadow_comp_mask_area"],
                        "shadow_records": prior_payload["shadow_records"],
                    }
                )
            densify_pkgs.append(
                {
                    "viewspace_point_tensor": viewspace_point_tensor,
                    "visibility_filter": visibility_filter,
                    "radii": radii,
                    "normal_error": normal_error.detach(),
                    "rend_alpha": rend_alpha.detach(),
                    "alpha_denom": alpha_denom.detach(),
                    "hotspot_box": hotspot_box,
                    "hotspot_loss": hotspot_loss,
                    "viewpoint_cam": viewpoint_cam,
                    "render_pkg": render_pkg,
                    "Ll1": Ll1,
                    "loss": loss,
                    "photo_loss": photometric_loss,
                }
            )
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            last_pkg = densify_pkgs[-1]
            ema_loss_for_log = 0.4 * last_pkg["photo_loss"].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, last_pkg["Ll1"], last_pkg["photo_loss"], l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                densify_diag_payload["view_render_budget_total"] += len(densify_pkgs)
                for pkg in densify_pkgs:
                    vf = pkg["visibility_filter"]
                    gaussians.max_radii2D[vf] = torch.max(gaussians.max_radii2D[vf], pkg["radii"][vf])
                    gaussians.add_densification_stats(pkg["viewspace_point_tensor"], vf)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    camera_t = [
                        pkg["viewpoint_cam"].camera_center / torch.norm(pkg["viewpoint_cam"].camera_center)
                        for pkg in densify_pkgs
                    ]
                    if len(camera_t) >= 2:
                        diffs = []
                        for i, cam1 in enumerate(camera_t):
                            for j, cam2 in enumerate(camera_t):
                                if i != j:
                                    diffs.append(torch.sqrt(torch.sum((cam1 - cam2) ** 2)))
                        diffs = torch.stack(diffs)
                        if torch.any(diffs > 1):
                            densify_t = opt.densify_grad_threshold * 0.5
                        else:
                            densify_t = opt.densify_grad_threshold
                    else:
                        densify_t = opt.densify_grad_threshold
                    assert scene.n_train_full > 0, "n_train_full == 0: no training views loaded"
                    rho = float(scene.n_train_selected) / float(scene.n_train_full)
                    densify_t = densify_t * (rho ** opt.densify_sparse_gamma)
                    if opt.normal_guided_densify:
                        mean_ne_num = sum((pkg["normal_error"] * pkg["rend_alpha"]).sum() for pkg in densify_pkgs)
                        mean_ne_den = sum(pkg["alpha_denom"] for pkg in densify_pkgs)
                        mean_ne = mean_ne_num / (mean_ne_den + 1e-6)
                        factor = torch.clamp(1.0 - mean_ne / 2.0, 0.5, 1.0).item()
                        densify_t *= factor

                    sorted_pkgs = sorted(densify_pkgs, key=lambda pkg: pkg["hotspot_loss"], reverse=True)
                    sorted_boxes = [pkg["hotspot_box"] for pkg in sorted_pkgs]
                    sorted_cams = [pkg["viewpoint_cam"] for pkg in sorted_pkgs]
                    guard = None
                    if opt.vf_enable_densify_guard and vf_support_pack is not None:
                        gaussian2voxel = rehash_gaussians_to_voxels(gaussians.get_xyz.detach(), vf_support_pack)
                        current_voxel_support_ok_ratio = compute_current_voxel_support_ok_ratio(gaussian2voxel, vf_support_pack)
                        guard = build_densify_guard(
                            gaussians_xyz=gaussians.get_xyz.detach(),
                            support_pack=vf_support_pack,
                            voxel_support_ok_ratio=current_voxel_support_ok_ratio,
                            hotspot_boxes=sorted_boxes,
                            densify_t=densify_t,
                            size_threshold=size_threshold,
                            max_radii2D=gaussians.max_radii2D.detach(),
                            lod_resolution_factor=_lod_resolution_factor(dataset.resolution),
                            gaussian2voxel=gaussian2voxel,
                        )

                    densify_result = gaussians.densify_and_prune(
                        densify_t,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        sorted_cams,
                        sorted_boxes,
                        densify_guard=guard,
                    )
                    densify_stats = summarize_densify_stats(
                        guard=guard,
                        clone_mask=densify_result["clone_mask"] if densify_result else None,
                        split_mask=densify_result["split_mask"] if densify_result else None,
                        scaling=gaussians.get_scaling,
                        mv_view_count=len(densify_pkgs),
                    )
                    densify_stats["iteration"] = int(iteration)
                    densify_diag_payload["history"].append(densify_stats)
                    densify_diag_payload["densify_births_by_support_class"] = densify_stats["densify_births_by_support_class"]
                    densify_diag_payload["densify_reject_ratio_by_support_class"] = densify_stats["densify_reject_ratio_by_support_class"]
                    densify_diag_payload["spiky_gaussian_ratio"] = densify_stats["spiky_gaussian_ratio"]
                    densify_diag_payload["mv_densify_stat_view_count_mean"] = densify_stats["mv_densify_stat_view_count_mean"]
                    top_pkg = sorted_pkgs[0]
                    hotspot_mask = build_hotspot_mask(top_pkg["render_pkg"]["render"], top_pkg["hotspot_box"])
                    risk_masks = project_voxel_risk_to_image(top_pkg["viewpoint_cam"], vf_support_pack) if vf_support_pack is not None else {
                        "under_target": torch.zeros_like(hotspot_mask),
                        "intrinsic_low_obs": torch.zeros_like(hotspot_mask),
                        "shadow": torch.zeros_like(hotspot_mask),
                    }
                    tracker_payload = hotspot_tracker.update(
                        iteration=iteration,
                        hotspot_mask=hotspot_mask,
                        risk_masks=risk_masks,
                    )
                    write_online_diagnostics(dataset.model_path, tracker_payload, densify_diag_payload)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations) and dataset.resolution != -1:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + "_mv" +str(pipe.mv) + ".pth")

    densify_diag_payload["training_wall_clock_sec"] = round(time.time() - train_start_time, 4)
    densify_diag_payload["final_gaussian_count"] = int(scene.gaussians.get_xyz.shape[0])
    write_online_diagnostics(
        dataset.model_path,
        {
            "summary": hotspot_tracker.history[-1] if hotspot_tracker.history else {},
            "history": hotspot_tracker.history,
            "latest": hotspot_tracker.history[-1] if hotspot_tracker.history else {},
        },
        densify_diag_payload,
    )
    priors_dir = os.path.join(dataset.model_path, "priors")
    os.makedirs(priors_dir, exist_ok=True)
    _write_json(os.path.join(priors_dir, "prior_alignment_stats.json"), prior_stats)
    _write_json(
        os.path.join(priors_dir, "trusted_prior_views.json"),
        [record for record in prior_stats["records"] if record.get("trusted_prior_view")],
    )
    _write_json(
        os.path.join(priors_dir, "trusted_prior_locals.json"),
        [
            {
                "iteration": record["iteration"],
                "view_name": record["view_name"],
                "comp_mask_area": record["comp_mask_area"],
                "trusted_prior_local": record.get("trusted_prior_local", False),
                "trusted_prior_shadow": record.get("trusted_prior_shadow", False),
                "shadow_comp_mask_area": record.get("shadow_comp_mask_area", 0),
            }
            for record in prior_stats["records"]
            if record.get("trusted_prior_local") or record.get("trusted_prior_shadow")
        ],
    )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        # Keep run artifacts out of the source tree by default:
        # train.py lives in <repo>/MVGS-master, so default to <repo>/output.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.model_path = os.path.join(repo_root, "output", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
