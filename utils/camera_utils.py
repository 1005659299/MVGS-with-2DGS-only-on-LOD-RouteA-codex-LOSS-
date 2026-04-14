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

import os

import numpy as np
import torch
from PIL import Image
from kornia import create_meshgrid

from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0

WARNED = False


def load_mask_by_image_name(root, image_name, resolution):
    if not root:
        return None
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"):
        path = os.path.join(root, image_name + ext)
        if not os.path.exists(path):
            continue
        if ext == ".npy":
            arr = np.load(path)
            mask = Image.fromarray((arr > 0).astype(np.uint8) * 255)
        else:
            mask = Image.open(path).convert("L")
        resized = PILtoTorch(mask, resolution)
        return (resized[:1, ...] > 0.5).float()
    return None


def build_edge_safe_mask(width, height, edge_margin_px, static_mask=None, camera_mask=None):
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    d_edge = torch.minimum(
        torch.minimum(xx, width - 1 - xx),
        torch.minimum(yy, height - 1 - yy),
    ).float()
    edge_safe = (d_edge >= float(edge_margin_px)).float().unsqueeze(0)
    if static_mask is not None:
        edge_safe = edge_safe * static_mask.float()
    if camera_mask is not None:
        edge_safe = edge_safe * camera_mask.float()
    return edge_safe

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None
    vf_static_mask = None
    vf_camera_mask = None
    vf_edge_safe_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if getattr(args, "vf_mask_path", ""):
        vf_static_mask = load_mask_by_image_name(
            args.vf_mask_path,
            cam_info.image_name,
            resolution,
        )
    if getattr(args, "vf_camera_mask_path", ""):
        vf_camera_mask = load_mask_by_image_name(
            args.vf_camera_mask_path,
            cam_info.image_name,
            resolution,
        )
    vf_edge_safe_mask = build_edge_safe_mask(
        width=resolution[0],
        height=resolution[1],
        edge_margin_px=getattr(args, "vf_edge_margin_px", 15),
        static_mask=vf_static_mask,
        camera_mask=vf_camera_mask,
    )

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  vf_static_mask=vf_static_mask,
                  vf_camera_mask=vf_camera_mask,
                  vf_edge_safe_mask=vf_edge_safe_mask,
                  data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def set_rays_od(cams):
    for id, cam in enumerate(cams):
        rayd=1
        if rayd is not None:
            projectinverse = cam.projection_matrix.T.inverse()
            camera2wold = cam.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(cam.image_height, cam.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            xindx = pixgrid[:,:,0] # x
            yindx = pixgrid[:,:,1] # y
            ndcy, ndcx = pix2ndc(yindx, cam.image_height), pix2ndc(xindx, cam.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4
            projected = ndccamera @ projectinverse.T
            diretioninlocal = projected / projected[:,:,3:] #v
            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T
            # rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)
            rays_d = direction
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            cam.rayo = cam.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0).cpu()
            cam.rayd = rays_d.permute(2, 0, 1).unsqueeze(0).cpu()
        else :
            cam.rayo = None
            cam.rayd = None

def set_rays(scene,resolution_scales=[1.0]):
    set_rays_od(scene.getTrainCameras())
    for resolution_scale in resolution_scales:
        for cam in scene.train_cameras[resolution_scale]:
            if cam.rayo is not None:
                cam.rays = torch.cat([cam.rayo, cam.rayd], dim=1)
