import torch
from xfield.dataset import Scene
import os
from tqdm import tqdm
from os import makedirs
from xfield.gaussian import GaussianModel, render, query, initialize_gaussian
import torchvision
from xfield.utils.general_utils import safe_state
from xfield.utils.camera_utils import compute_obb, create_views
from argparse import ArgumentParser
from xfield.arguments import ModelParams, PipelineParams, get_combined_args
import imageio
import numpy as np
import open3d as o3d

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.normalized_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_360(model_path, name, iteration, views, gaussians, pipeline):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    renderings = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        renderings.append(rendering)

    if render_path is not None:
        video_filename = os.path.join(render_path, "360.mp4")
        writer = imageio.get_writer(video_filename, fps=60)   
        for rendering in renderings:
            rgb8 = to8b(rendering.permute(1, 2, 0).detach().cpu().numpy())
            writer.append_data(rgb8)
        if writer is not None:
            writer.close()   

# Render different method on same camera poses based on point cloud 
def render_sets_coordinate(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, pointcloud_path):
    with torch.no_grad():

        scene = Scene(
            dataset,
            shuffle=False,
        )

        # Set up some parameters
        volume_to_world = min(scene.scanner_cfg["sVoxel"])
        scale_bound = None
        if dataset.scale_min and dataset.scale_max:
            scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world

        # Set up Gaussians
        gaussians = GaussianModel(scale_bound)
        loaded_iter = initialize_gaussian(gaussians, dataset, iteration)

        ply_data = o3d.io.read_point_cloud(pointcloud_path)

        xyz = np.asarray(ply_data.points)
        xyz = torch.from_numpy(xyz).float() 

        obb_center, principal_axes, obb_lengths = compute_obb(xyz)
        axis_idx = 0
        train_start = 0.
        test_start = 1.
        radius = 5.
        up = 1. * principal_axes[:, axis_idx]
        views, camera_positions = create_views(xyz, axis=principal_axes[:, axis_idx], up=up, radius=radius, num_views=360, start=train_start)
        
        render_360(dataset.model_path, "360", loaded_iter, views, gaussians, pipeline)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():

        scene = Scene(
            dataset,
            shuffle=False,
        )

        # Set up some parameters
        volume_to_world = min(scene.scanner_cfg["sVoxel"])
        scale_bound = None
        if dataset.scale_min and dataset.scale_max:
            scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world

        # Set up Gaussians
        gaussians = GaussianModel(scale_bound)
        loaded_iter = initialize_gaussian(gaussians, dataset, iteration)

        obb_center, principal_axes, obb_lengths = compute_obb(gaussians._xyz)
        axis_idx = 0
        train_start = 0.
        test_start = 1.
        radius = 5.
        up = -1. * principal_axes[:, axis_idx]
        views, camera_positions = create_views(gaussians._xyz, axis=principal_axes[:, axis_idx], up=up, radius=radius, num_views=360, start=train_start)
        render_360(dataset.model_path, "360", loaded_iter, views, gaussians, pipeline)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-p", "--pointcloud_path", help="Path to the point cloud file")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    safe_state(args.quiet)

    render_sets_coordinate(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.pointcloud_path)