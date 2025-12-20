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
import sys
import torch
import numpy as np

sys.path.append("./")
from xfield.dataset.cameras import Camera, Simple_Camera

# from scene.cameras import MiniCam, Simple_Camera

def create_views(points, axis=torch.tensor([0., 0., 1.]), up=None, radius=3., num_views=360, start=0.):
    center = points.mean(dim=0)  # Assuming points is a tensor
    device = points.device  # Get the device (CPU/GPU) from the points tensor
    cams = []
    axis = axis.to(device).float()
    camera_positions = []

    # Find a vector orthogonal to the rotation axis
    if torch.allclose(axis, torch.tensor([0., 0., 1.], device=device)):
        orthogonal_vector = torch.tensor([1., 0., 0.], device=device)
    else:
        orthogonal_vector = torch.cross(axis, torch.tensor([0., 0., -1.], device=device))
    orthogonal_vector = orthogonal_vector / torch.norm(orthogonal_vector)

    # Scale the orthogonal vector to the desired radius
    initial_camera_pos = orthogonal_vector * radius + center

    for angle in np.linspace(2 * np.pi,  start,num=num_views, endpoint=False):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        cos_a = torch.tensor(cos_a, device=device).float()
        sin_a = torch.tensor(sin_a, device=device).float()
        one_minus_cos_a = 1 - cos_a
        ux, uy, uz = axis

        rotation_matrix = torch.tensor([
            [cos_a + ux**2 * one_minus_cos_a, ux * uy * one_minus_cos_a - uz * sin_a, ux * uz * one_minus_cos_a + uy * sin_a],
            [uy * ux * one_minus_cos_a + uz * sin_a, cos_a + uy**2 * one_minus_cos_a, uy * uz * one_minus_cos_a - ux * sin_a],
            [uz * ux * one_minus_cos_a - uy * sin_a, uz * uy * one_minus_cos_a + ux * sin_a, cos_a + uz**2 * one_minus_cos_a]
        ], device=device).float()

        camera_pos = torch.matmul(rotation_matrix, initial_camera_pos - center) + center
        camera_positions.append(camera_pos)
        # T = -camera_pos  # Translation vector

        # z-axis should point from the camera towards the center
        z_axis = (center - camera_pos) / torch.norm(center - camera_pos)
        # Arbitrary up vector, typically the global y or z axis
        # if up is None:
        # up = torch.tensor([2., 1., 3.], device=device).float()
        # up = up / torch.norm(up)
        # Compute the x-axis as the cross product of up and z_axis
        x_axis = torch.cross(up, z_axis)
        x_axis /= torch.norm(x_axis)
        # y-axis is the cross product of z_axis and x_axis
        y_axis = torch.cross(z_axis, x_axis)

        # Form the rotation matrix
        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        T = -torch.matmul(R.transpose(0, 1), camera_pos)

        # Camera initialization, assuming Simple_Camera is your camera class
        cam = Simple_Camera(colmap_id=0, R=R.cpu().numpy(), T=T.cpu().numpy(), FoVx=1.0, FoVy=0.69, h=1060, w=1600, image_name='', uid=0, scale=1.0, data_device=device, qvec=0)
        cams.append(cam)
    camera_positions = torch.stack(camera_positions, dim=0)

    return cams, camera_positions


def compute_obb(points):
    # points shape: [N, 3]
    
    # Step 1: Compute the centroid
    centroid = points.mean(dim=0)
    
    # Step 2: Centralize points
    centralized_points = points - centroid
    
    # Step 3: Compute the covariance matrix
    covariance_matrix = torch.matmul(centralized_points.t(), centralized_points) / centralized_points.shape[0]
    
    # Step 4: Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
    
    # Step 5: Sorting the eigenvectors by eigenvalues
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    principal_axes = eigenvectors[:, sorted_indices]
    
    # Step 6: Project points onto principal axes
    projected_points = torch.matmul(centralized_points, principal_axes)
    min_bounds = projected_points.min(dim=0).values
    max_bounds = projected_points.max(dim=0).values
    
    # The center of the OBB in the original coordinates
    obb_center = centroid + torch.matmul(0.5 * (min_bounds + max_bounds), principal_axes.t())
    
    # The length of the OBB along each principal axis
    obb_lengths = max_bounds - min_bounds
    
    return obb_center, principal_axes, obb_lengths


def look_at(camera_position, target, up):
    z_axis = (camera_position - target) / torch.norm(camera_position - target)  # Forward
    x_axis = torch.cross(up, z_axis) / torch.norm(torch.cross(up, z_axis))  # Right
    y_axis = torch.cross(z_axis, x_axis) / torch.norm(torch.cross(z_axis, x_axis))  # Up

    view_matrix = torch.eye(4).float().to(camera_position.device)
    view_matrix[:3, :3] = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    view_matrix[:3, 3] = camera_position

    return view_matrix

def loadCam(args, id, cam_info):
    gt_image = torch.from_numpy(cam_info.image)[None]

    return Camera(
        colmap_id=cam_info.uid,
        scanner_cfg=cam_info.scanner_cfg,
        R=cam_info.R,
        T=cam_info.T,
        angle=cam_info.angle,
        mode=cam_info.mode,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
    )


def cameraList_from_camInfos(cam_infos, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.eye(4)
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T

    W2C = Rt
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "mode": camera.mode,
        "position_w2c": pos.tolist(),
        "rotation_w2c": serializable_array_2d,
        "FovY": camera.FovY,
        "FovX": camera.FovX,
    }
    return camera_entry
