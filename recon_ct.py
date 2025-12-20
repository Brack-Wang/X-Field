import os
import os.path as osp
import sys
import copy
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("./")
from xfield.utils.ct_utils import get_geometry, run_ct_recon_algs
from xfield.arguments import ModelParams
from xfield.dataset import Scene
from xfield.utils.general_utils import t2a

np.random.seed(0)


def save_slice(volume, slice_index=None, output_path="output_slice.png"):
    """Save a 2D slice of a 3D volume to a file."""
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()

    if slice_index is None:
        slice_index = volume.shape[2] // 2

    slice_data = volume[:, :, slice_index]
    plt.imshow(slice_data, cmap="gray")
    plt.colorbar()
    plt.title(f"Slice {slice_index}")
    plt.savefig(output_path)
    plt.close()
    print(f"Slice saved to {output_path}")


def load_renderings(pickle_file_path):
    """Load rendered projections from pickle file."""
    with open(pickle_file_path, "rb") as pkl_file:
        data = pickle.load(pkl_file)

    angles = np.array([item["angle"] for item in data])
    renderings = np.stack(
        [item["rendering"].squeeze(0).cpu().numpy() for item in data], axis=0
    )
    return angles, renderings


def main(args, model_args: ModelParams):
    # Load scene
    model_args.source_path = args.data
    scene = Scene(model_args, False)

    scanner_cfg = scene.scanner_cfg
    geo = get_geometry(scanner_cfg)

    save_path = osp.join(args.path, f"ct_recon_{args.n_real}")
    os.makedirs(save_path, exist_ok=True)

    vol_gt = scene.vol_gt.detach().cpu().numpy()
    save_slice(vol_gt, 50, osp.join(save_path, "gt_vol.png"))

    # Load original projections
    train_cameras = scene.getTrainCameras()
    projs_train = np.concatenate(
        [t2a(cam.original_image) for cam in train_cameras], axis=0
    )
    angles_train = np.stack([t2a(cam.angle) for cam in train_cameras], axis=0)

    # Load rendered projections
    views_train, renderings_train = load_renderings(
        osp.join(args.path, "views_renderings_render_train.pkl")
    )
    views_test, renderings_test = load_renderings(
        osp.join(args.path, "views_renderings_render_test.pkl")
    )

    # Replace some rendered projections with real ones
    n_real = args.n_real
    total_images = renderings_train.shape[0]
    step = total_images // n_real
    selected_indices = np.arange(0, total_images, step)[:n_real]

    print(f"projs_train shape: {projs_train.shape}")
    print(f"renderings_train shape: {renderings_train.shape}")
    print(f"Replacing {n_real} rendered projections with real ones")

    renderings_train[selected_indices] = projs_train[selected_indices]

    # Combine all projections
    renderings_all = np.concatenate([renderings_train, renderings_test], axis=0)
    views_all = np.concatenate([views_train, views_test], axis=0)

    print(f"Total projections for CT reconstruction: {renderings_all.shape}")

    # Run CT reconstruction
    run_ct_recon_algs(
        renderings_all, views_all, copy.deepcopy(geo), vol_gt, save_path, args.method
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT reconstruction from rendered projections")
    lp = ModelParams(parser)
    parser.add_argument("--data", type=str, required=True, help="Path to data")
    parser.add_argument("--path", type=str, required=True, help="Path to test output (with pkl files)")
    parser.add_argument("--method", type=str, default="fdk", help="CT reconstruction method")
    parser.add_argument("--n_real", type=int, default=20, help="Number of real projections to use")

    args = parser.parse_args()
    main(args, lp.extract(args))
