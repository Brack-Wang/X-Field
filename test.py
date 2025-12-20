import os
import os.path as osp
import sys
import torch
from tqdm import tqdm
import torchvision
from time import time
import numpy as np
import yaml
import pickle
from argparse import ArgumentParser

sys.path.append("./")
from xfield.arguments import ModelParams, PipelineParams, get_combined_args
from xfield.dataset import Scene
from xfield.gaussian import GaussianModel, render, initialize_gaussian
from xfield.utils.general_utils import safe_state
from xfield.utils.image_utils import metric_proj


def testing(
    dataset: ModelParams,
    pipeline: PipelineParams,
    iteration: int,
    skip_render_train: bool,
    skip_render_test: bool,
):
    """Main evaluation function.

    Args:
        dataset: Model parameters including paths
        pipeline: Pipeline parameters for rendering
        iteration: Model iteration to load (-1 for latest)
        skip_render_train: Skip rendering training views
        skip_render_test: Skip rendering test views
    """
    scene = Scene(dataset, shuffle=False)

    # Initialize Gaussians from checkpoint
    gaussians = GaussianModel(None)
    loaded_iter = initialize_gaussian(gaussians, dataset, iteration)
    scene.gaussians = gaussians

    save_path = osp.join(dataset.model_path, "test", f"iter_{loaded_iter}")

    print(f"Train cameras: {len(scene.getTrainCameras())}, Test cameras: {len(scene.getTestCameras())}")

    # Evaluate projection rendering on training views
    if not skip_render_train:
        evaluate_render(save_path, "render_train", scene.getTrainCameras(), gaussians, pipeline)

    # Evaluate projection rendering on test views
    if not skip_render_test:
        evaluate_render(save_path, "render_test", scene.getTestCameras(), gaussians, pipeline)


def evaluate_render(save_path, name, views, gaussians, pipeline):
    """Evaluate projection rendering with global normalization.

    Args:
        save_path: Directory to save results
        name: Name of this evaluation (e.g., 'render_train', 'render_test')
        views: List of camera views to render
        gaussians: Gaussian model
        pipeline: Pipeline parameters
    """
    proj_save_path = osp.join(save_path, name)

    # Skip if already rendered
    if osp.exists(osp.join(save_path, f"eval2d_{name}.yml")):
        print(f"{name} in {save_path} already rendered. Skipping.")
        return

    os.makedirs(proj_save_path, exist_ok=True)

    # Brightness adjustment factor for predictions
    brightness_scale = 0.8

    # First pass: render all views and collect data
    gt_list = []
    render_list = []
    data_to_save = []
    inference_times = []
    total_start_time = time()

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        start_time = time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        rendering = render(view, gaussians, pipeline)["render"]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_time = time() - start_time
        inference_times.append(inference_time)

        # Apply brightness scaling to reduce over-brightness
        rendering = rendering * brightness_scale

        gt = view.original_image
        gt_list.append(gt)
        render_list.append(rendering)
        data_to_save.append({
            "angle": view.angle,
            "rendering": rendering,
            "gt": gt,
            "inference_time": inference_time,
        })

    total_time = time() - total_start_time

    # Compute global min/max for consistent normalization
    global_min_render = min(r.min().item() for r in render_list)
    global_max_render = max(r.max().item() for r in render_list)
    global_min_gt = min(g.min().item() for g in gt_list)
    global_max_gt = max(g.max().item() for g in gt_list)

    # Use combined range for both GT and predictions
    global_min = min(global_min_render, global_min_gt)
    global_max = max(global_max_render, global_max_gt)

    # Second pass: normalize and save images
    for idx, (rendering, gt) in enumerate(zip(render_list, gt_list)):
        # Normalize with global values
        norm_render = (rendering - global_min) / (global_max - global_min + 1e-8)
        norm_gt = (gt - global_min) / (global_max - global_min + 1e-8)

        # Save images
        torchvision.utils.save_image(norm_render, osp.join(proj_save_path, f"{idx:05d}_pred.png"))
        torchvision.utils.save_image(norm_gt, osp.join(proj_save_path, f"{idx:05d}_gt.png"))

    # Save pickle data
    pickle_file_path = osp.join(save_path, f"views_renderings_{name}.pkl")
    with open(pickle_file_path, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Data saved to {pickle_file_path}")

    # Compute metrics on original (non-normalized) data
    images = torch.cat(render_list, 0).permute(1, 2, 0)
    gt_images = torch.cat(gt_list, 0).permute(1, 2, 0)
    psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
    ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")

    # Statistics
    avg_time = np.mean(inference_times)
    eval_dict = {
        "psnr_2d": psnr_2d,
        "ssim_2d": ssim_2d,
        "psnr_2d_projs": psnr_2d_projs,
        "ssim_2d_projs": ssim_2d_projs,
        "avg_inference_time": avg_time,
        "min_inference_time": float(np.min(inference_times)),
        "max_inference_time": float(np.max(inference_times)),
        "std_inference_time": float(np.std(inference_times)),
        "total_time": total_time,
        "fps": len(views) / total_time,
        "global_min": global_min,
        "global_max": global_max,
    }

    with open(osp.join(save_path, f"eval2d_{name}.yml"), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

    print(f"{name} complete. PSNR: {psnr_2d:.2f}, SSIM: {ssim_2d:.4f}, FPS: {len(views)/total_time:.2f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_render_train", action="store_true", default=False)
    parser.add_argument("--skip_render_test", action="store_true", default=False)

    args = get_combined_args(parser)
    safe_state(args.quiet)

    with torch.no_grad():
        testing(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
            args.skip_render_train,
            args.skip_render_test,
        )
