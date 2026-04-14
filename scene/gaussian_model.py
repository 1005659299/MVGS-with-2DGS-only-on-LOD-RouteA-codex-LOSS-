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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        # [2D-GS] Build 4x4 transformation matrix for 2D Gaussian (Surfel)
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            # Extend 2D scaling to 3D by padding with 1 (third dimension)
            scaling_3d = torch.cat([scaling * scaling_modifier, torch.ones_like(scaling[:, :1])], dim=-1)
            RS = build_scaling_rotation(scaling_3d, rotation).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier=1):
        # [2D-GS] Pass center (xyz) to build 4x4 transformation matrix
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # [2D-GS] Use 2D scaling: repeat(1, 2) instead of repeat(1, 3)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        # [2D-GS] Random rotation initialization
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _pad_guard_tensor(self, tensor, n_init_points, fill_value):
        if tensor is None:
            return None
        if tensor.shape[0] >= n_init_points:
            return tensor[:n_init_points]
        pad_shape = (n_init_points - tensor.shape[0],) + tensor.shape[1:]
        pad = torch.full(pad_shape, fill_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=0)

    def _points_in_box3ds(self, box3ds):
        mask = torch.zeros_like(self.get_xyz[:, 0], dtype=torch.bool)
        for box3d in box3ds:
            x_min_3d, y_min_3d, z_min_3d, x_max_3d, y_max_3d, z_max_3d = box3d
            mask_xyz = (
                (self.get_xyz[:, 0] > x_min_3d)
                & (self.get_xyz[:, 0] < x_max_3d)
                & (self.get_xyz[:, 1] > y_min_3d)
                & (self.get_xyz[:, 1] < y_max_3d)
                & (self.get_xyz[:, 2] > z_min_3d)
                & (self.get_xyz[:, 2] < z_max_3d)
            )
            mask = torch.logical_or(mask, mask_xyz)
        return mask

    def densify_and_split(
        self,
        grads,
        grad_threshold,
        scene_extent,
        N=2,
        *,
        split_allow=None,
        tau_split=None,
        big_screen_force_split=None,
    ):
        """[2D-GS] Modified densify_and_split to handle 2D scaling"""
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        split_allow = self._pad_guard_tensor(split_allow, n_init_points, True)
        tau_split = self._pad_guard_tensor(tau_split, n_init_points, float(grad_threshold))
        big_screen_force_split = self._pad_guard_tensor(big_screen_force_split, n_init_points, False)
        if split_allow is None:
            split_allow = torch.ones((n_init_points,), dtype=torch.bool, device="cuda")
        if tau_split is None:
            tau_split = torch.full((n_init_points,), float(grad_threshold), dtype=torch.float32, device="cuda")
        if big_screen_force_split is None:
            big_screen_force_split = torch.zeros((n_init_points,), dtype=torch.bool, device="cuda")
        split_grad_ok = padded_grad >= tau_split
        selected_pts_mask = split_allow & split_grad_ok
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        selected_pts_mask = selected_pts_mask | (split_allow & big_screen_force_split)
        if not torch.any(selected_pts_mask):
            return selected_pts_mask

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        return selected_pts_mask

    def densify_and_clone(
        self,
        grads,
        grad_threshold,
        scene_extent,
        box3ds,
        *,
        clone_allow=None,
        tau_clone=None,
        box_bonus_mask=None,
    ):
        n_init_points = self.get_xyz.shape[0]
        clone_allow = self._pad_guard_tensor(clone_allow, n_init_points, True)
        tau_clone = self._pad_guard_tensor(tau_clone, n_init_points, float(grad_threshold))
        if clone_allow is None:
            clone_allow = torch.ones((n_init_points,), dtype=torch.bool, device="cuda")
        if tau_clone is None:
            tau_clone = torch.full((n_init_points,), float(grad_threshold), dtype=torch.float32, device="cuda")
        if box_bonus_mask is None:
            box_bonus_mask = self._points_in_box3ds(box3ds) if box3ds else torch.zeros((n_init_points,), dtype=torch.bool, device="cuda")
        else:
            box_bonus_mask = self._pad_guard_tensor(box_bonus_mask, n_init_points, False)
        grad_norm = torch.norm(grads, dim=-1)
        clone_grad_ok = grad_norm >= tau_clone[:grad_norm.shape[0]]
        clone_grad_ok = self._pad_guard_tensor(clone_grad_ok, n_init_points, False)
        selected_pts_mask = clone_allow & clone_grad_ok
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )
        selected_pts_mask = selected_pts_mask | (clone_allow & box_bonus_mask)
        if not torch.any(selected_pts_mask):
            return selected_pts_mask
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        return selected_pts_mask


    def gather_rays(self, rays, region):
        top = rays[region[1], region[0]:region[2]]
        bottom = rays[region[3]-1, region[0]:region[2]]
        left = rays[region[1]:region[3], region[0]]
        right = rays[region[1]:region[3], region[2]-1]
        return torch.cat([top, bottom, left, right], dim=0)

    def intersect_lines(self, ray1_origin, ray1_dir, ray2_origin, ray2_dir):
        # Normalize direction vectors
        ray1_dir = ray1_dir / torch.norm(ray1_dir)
        ray2_dir = ray2_dir / torch.norm(ray2_dir)

        # Cross product of direction vectors (using explicit dim=-1 for 3D vectors)
        cross_dir = torch.linalg.cross(ray1_dir, ray2_dir, dim=-1)
        cross_dir_norm = torch.norm(cross_dir)

        # Check if the rays are parallel
        if cross_dir_norm < 1e-6:
            return None  # Rays are parallel and do not intersect

        # Line between the origins
        origin_diff = ray2_origin - ray1_origin

        # Calculate the distance along the cross product direction
        t1 = torch.dot(torch.linalg.cross(origin_diff, ray2_dir, dim=-1), cross_dir) / (cross_dir_norm ** 2)
        t2 = torch.dot(torch.linalg.cross(origin_diff, ray1_dir, dim=-1), cross_dir) / (cross_dir_norm ** 2)

        # Closest points on each ray
        closest_point1 = ray1_origin + t1 * ray1_dir
        closest_point2 = ray2_origin + t2 * ray2_dir

        # Midpoint between the two closest points as the intersection point
        intersection_point = (closest_point1 + closest_point2) / 2.0

        return intersection_point


    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        cams,
        boxes,
        *,
        densify_guard=None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        box3ds = []
        if cams is not None:
            assert len(cams) == len(boxes), "cams/boxes length mismatch"
            for i, cam_0 in enumerate(cams):
                for off, cam_1 in enumerate(cams[i+1:]):
                    j = i + 1 + off
                    ray0_o = cam_0.rayo
                    ray0_d = cam_0.rayd
                    box0 = boxes[i]

                    ray0_o_topleft = ray0_o[0,:, box0[0], box0[1]]
                    ray0_d_topleft = ray0_d[0,:, box0[0], box0[1]]

                    ray0_o_bottomright = ray0_o[0,:, box0[2], box0[3]]
                    ray0_d_bottomright = ray0_d[0,:, box0[2], box0[3]]

                    ray0_o_bottomleft = ray0_o[0,:, box0[2], box0[1]]
                    ray0_d_bottomleft = ray0_d[0,:, box0[2], box0[1]]

                    ray0_o_topright = ray0_o[0, :, box0[0], box0[3]]
                    ray0_d_topright = ray0_d[0, :, box0[0], box0[3]]

                    ray1_o = cam_1.rayo
                    ray1_d = cam_1.rayd
                    box1 = boxes[j]

                    ray1_o_topleft = ray1_o[0, :, box1[0], box1[1]]
                    ray1_d_topleft = ray1_d[0, :, box1[0], box1[1]]

                    ray1_o_bottomright = ray1_o[0, :, box1[2], box1[3]]
                    ray1_d_bottomright = ray1_d[0, :, box1[2], box1[3]]

                    ray1_o_bottomleft = ray1_o[0, :, box1[2], box1[1]]
                    ray1_d_bottomleft = ray1_d[0, :, box1[2], box1[1]]

                    ray1_o_topright = ray1_o[0, :, box1[0], box1[3]]
                    ray1_d_topright = ray1_d[0, :, box1[0], box1[3]]


                    topleft_intersect = self.intersect_lines(ray0_o_topleft, ray0_d_topleft, ray1_o_topleft, ray1_d_topleft)
                    bottomright_intersect = self.intersect_lines(ray0_o_bottomright, ray0_d_bottomright, ray1_o_bottomright, ray1_d_bottomright)
                    bottomleft_interset=self.intersect_lines(ray0_o_bottomleft, ray0_d_bottomleft, ray1_o_bottomleft, ray1_d_bottomleft)
                    topright_intersect = self.intersect_lines(ray0_o_topright, ray0_d_topright, ray1_o_topright, ray1_d_topright)


                    region3d = [topleft_intersect,bottomright_intersect, bottomleft_interset, topright_intersect]
                    if len(region3d)==0 or None in region3d:
                        continue
                    region3d = torch.vstack(region3d)
                    x_min_3d = torch.min(region3d[:, 0])
                    y_min_3d = torch.min(region3d[:, 1])
                    z_min_3d = torch.min(region3d[:, 2])

                    x_max_3d = torch.max(region3d[:, 0])
                    y_max_3d = torch.max(region3d[:, 1])
                    z_max_3d = torch.max(region3d[:, 2])



                    box3d = [x_min_3d, y_min_3d, z_min_3d, x_max_3d, y_max_3d, z_max_3d]
                    box3ds.append(box3d)
        if densify_guard is None:
            clone_allow = split_allow = tau_clone = tau_split = None
            box_bonus_mask = big_screen_force_split = None
        else:
            clone_allow = densify_guard.clone_allow
            split_allow = densify_guard.split_allow
            tau_clone = densify_guard.tau_clone
            tau_split = densify_guard.tau_split
            box_bonus_mask = densify_guard.box_bonus_mask
            big_screen_force_split = densify_guard.big_screen_force_split

        clone_mask = self.densify_and_clone(
            grads,
            max_grad,
            extent,
            box3ds,
            clone_allow=clone_allow,
            tau_clone=tau_clone,
            box_bonus_mask=box_bonus_mask,
        )
        split_mask = self.densify_and_split(
            grads,
            max_grad,
            extent,
            split_allow=split_allow,
            tau_split=tau_split,
            big_screen_force_split=big_screen_force_split,
        )

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        return {
            "clone_mask": clone_mask,
            "split_mask": split_mask,
            "prune_mask": prune_mask,
        }

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1