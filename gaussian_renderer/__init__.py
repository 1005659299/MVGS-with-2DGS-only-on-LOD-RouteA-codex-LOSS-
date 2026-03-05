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
# [2D-GS + LOD] Hybrid rendering with distance-driven LOD

import torch
import math
# [LOD] Import both 2D-GS (Surfel) and 3D-GS rasterizers
from diff_surfel_rasterization import GaussianRasterizationSettings as SurfelSettings
from diff_surfel_rasterization import GaussianRasterizer as SurfelRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as GaussianSettings
from diff_gaussian_rasterization import GaussianRasterizer as GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal


LOD_TRANSITION_FIXED_SEED = 0
LOD_TRANSITION_SEED_PRIME = 1000003
LOD_TRAIN_SEED_STRIDE = 9176


def _stable_string_hash(text: str) -> int:
    """Deterministic string hash that is stable across Python processes."""
    h = 2166136261  # FNV-1a 32-bit offset basis
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _camera_seed_base(viewpoint_camera) -> int:
    # uid is guaranteed by cameraList_from_camInfos.
    cam_key = int(viewpoint_camera.uid)
    return (LOD_TRANSITION_FIXED_SEED + cam_key * LOD_TRANSITION_SEED_PRIME) & 0x7FFFFFFF


def _deterministic_rand_like(ref_tensor: torch.Tensor, seed: int) -> torch.Tensor:
    gen_device = "cuda" if ref_tensor.is_cuda else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(int(seed) & 0x7FFFFFFF)
    return torch.rand(
        ref_tensor.shape,
        dtype=ref_tensor.dtype,
        device=ref_tensor.device,
        generator=generator,
    )


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, rng_step=None):
    """
    [LOD] Hybrid render using distance-driven LOD model.
    - Cascade 0 (near): Full 2D-GS Ray-Plane intersection
    - Cascade 1 (transition): Stochastic 2D-GS (1/n sampling)
    - Cascade 2 (far): 3D-GS EWA splatting
    """
    # 1. Calculate view-space depth for all Gaussians
    w2c = viewpoint_camera.world_view_transform
    R_w2c = w2c[:3, :3].transpose(0, 1)
    t_w2c = w2c[3, :3]
    p_view = (pc.get_xyz @ R_w2c) + t_w2c
    depths = p_view[:, 2]

    # 2. LOD parameters from pipeline config
    Z_NEAR_LIMIT = pipe.lod_near_limit
    Z_TRANSITION_LIMIT = pipe.lod_transition_limit
    MIN_PROB = pipe.lod_min_prob

    # 3. Generate cascade masks (Perceptually-Driven Inverse-Depth Sampling)
    mask_near = depths <= Z_NEAR_LIMIT
    mask_trans = (depths > Z_NEAR_LIMIT) & (depths <= Z_TRANSITION_LIMIT)

    # Inverse-depth probability: P = clamp(Z_near/z, P_min, 1.0)
    perceptual_prob = torch.clamp(
        Z_NEAR_LIMIT / (depths + 1e-6),
        min=MIN_PROB,
        max=1.0
    )
    cam_seed = _camera_seed_base(viewpoint_camera)
    if rng_step is not None:
        current_seed = (cam_seed + int(rng_step) * LOD_TRAIN_SEED_STRIDE) & 0x7FFFFFFF
    else:
        current_seed = cam_seed
    rand_vals = _deterministic_rand_like(depths, current_seed)
    mask_trans_sampled = mask_trans & (rand_vals < perceptual_prob)

    mask_surfel_engine = mask_near | mask_trans_sampled
    mask_far_engine = depths > Z_TRANSITION_LIMIT

    # 4. Prepare common data
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    means3D = pc.get_xyz
    means2D = torch.zeros_like(means3D, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except:
        pass
    opacities = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # Pre-compute colors from SH
    colors_precomp = override_color
    if colors_precomp is None:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # 5. Pass 1: Far (Background) - 3D-GS Engine
    idx_far = torch.nonzero(mask_far_engine).squeeze(-1)
    if idx_far.numel() > 0:
        raster_settings_3d = GaussianSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizer_3d = GaussianRasterizer(raster_settings=raster_settings_3d)
        # Convert 2D scales (N,2) to 3D scales (N,3) for 3D-GS
        scales_far_3d = torch.cat([scales[idx_far], scales[idx_far][:, :1]], dim=-1)
        image_far, radii_far = rasterizer_3d(
            means3D=means3D[idx_far],
            means2D=means2D[idx_far],
            shs=None,
            colors_precomp=colors_precomp[idx_far],
            opacities=opacities[idx_far],
            scales=scales_far_3d,
            rotations=rotations[idx_far],
            cov3D_precomp=None
        )
        rendered_far = image_far
    else:
        h = int(viewpoint_camera.image_height)
        w = int(viewpoint_camera.image_width)
        rendered_far = bg_color[:, None, None].expand(3, h, w)
        radii_far = torch.zeros((0,), device="cuda", dtype=torch.int32)

    # 6. Pass 2: Near + Transition (Foreground) - 2D-GS Engine
    idx_surfel = torch.nonzero(mask_surfel_engine).squeeze(-1)
    if idx_surfel.numel() > 0:
        raster_settings_2d = SurfelSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.zeros(3, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizer_2d = SurfelRasterizer(raster_settings=raster_settings_2d)
        image_near, radii_near, allmap_near = rasterizer_2d(
            means3D=means3D[idx_surfel],
            means2D=means2D[idx_surfel],
            shs=None,
            colors_precomp=colors_precomp[idx_surfel],
            opacities=opacities[idx_surfel],
            scales=scales[idx_surfel],
            rotations=rotations[idx_surfel],
            cov3D_precomp=None
        )
        alpha_near = allmap_near[1:2]
    else:
        image_near = torch.zeros_like(rendered_far)
        alpha_near = torch.zeros((1, rendered_far.shape[1], rendered_far.shape[2]), device="cuda")
        radii_near = torch.zeros((0,), device="cuda", dtype=torch.int32)
        allmap_near = None

    # 7. Compositing: Final = Near + Far * (1 - Alpha_Near)
    final_image = image_near + rendered_far * (1.0 - alpha_near)

    # 8. Process geometric information (from 2D-GS only)
    h, w = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
    if idx_surfel.numel() > 0 and allmap_near is not None:
        render_alpha = allmap_near[1:2]
        render_normal = allmap_near[2:5]
        render_normal = (render_normal.permute(1, 2, 0) @
                        (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)
        render_dist = allmap_near[6:7]
        render_depth_expected = allmap_near[0:1]
        render_depth_median = allmap_near[5:6]
        render_depth_expected = render_depth_expected / render_alpha.clamp(min=1e-8)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
        surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + pipe.depth_ratio * render_depth_median
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2, 0, 1) * render_alpha.detach()
    else:
        render_alpha = torch.zeros((1, h, w), device="cuda")
        render_normal = torch.zeros((3, h, w), device="cuda")
        render_dist = torch.zeros((1, h, w), device="cuda")
        surf_normal = torch.zeros((3, h, w), device="cuda")
        surf_depth = torch.zeros((1, h, w), device="cuda")

    # Combine radii from both engines
    all_radii = torch.zeros(means3D.shape[0], device="cuda", dtype=torch.int32)
    if idx_surfel.numel() > 0 and radii_near.numel() > 0:
        all_radii[mask_surfel_engine] = radii_near
    if idx_far.numel() > 0 and radii_far.numel() > 0:
        all_radii[mask_far_engine] = radii_far

    return {
        "render": final_image,
        "viewspace_points": means2D,
        "visibility_filter": all_radii > 0,
        "radii": all_radii,
        "rend_alpha": render_alpha,
        "rend_normal": render_normal,
        "rend_dist": render_dist,
        "surf_depth": surf_depth,
        "surf_normal": surf_normal,
    }
