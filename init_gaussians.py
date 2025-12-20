"""Initialize Gaussian point clouds from CT reconstruction."""
import os
import os.path as osp
import sys
import argparse
import copy
import numpy as np

sys.path.append("./")
from xfield.utils.ct_utils import get_geometry, recon_volume
from xfield.arguments import ParamGroup, ModelParams, PipelineParams
from xfield.dataset import Scene
from xfield.utils.general_utils import t2a

np.random.seed(0)


class InitParams(ParamGroup):
    def __init__(self, parser):
        self.recon_method = "fdk"
        self.n_points = 50000
        self.density_thresh = 0.05
        self.density_rescale = 0.15
        self.random_density_max = 1.0
        super().__init__(parser, "Initialization Parameters")


def init_pcd(projs, angles, geo, scanner_cfg, args: InitParams, save_path):
    """Initialize Gaussians from CT reconstruction or random sampling."""
    recon_method = args.recon_method
    n_points = args.n_points

    assert recon_method in ["random", "fdk", "cgls", "sart", "hybrid_cgls_sart_tv"], \
        f"--recon_method '{recon_method}' not supported."

    if recon_method == "random":
        print("Initialize random point clouds.")
        sampled_positions = np.array(scanner_cfg["offOrigin"])[None, ...] + np.array(
            scanner_cfg["sVoxel"]
        )[None, ...] * (np.random.rand(n_points, 3) - 0.5)
        sampled_densities = np.random.rand(n_points) * args.random_density_max
    else:
        print(f"Initialize point clouds with volume reconstructed from {recon_method}.")
        vol = recon_volume(projs, angles, copy.deepcopy(geo), recon_method)

        density_mask = vol > args.density_thresh
        valid_indices = np.argwhere(density_mask)
        offOrigin = np.array(scanner_cfg["offOrigin"])
        dVoxel = np.array(scanner_cfg["dVoxel"])
        sVoxel = np.array(scanner_cfg["sVoxel"])

        assert valid_indices.shape[0] >= n_points, \
            "Valid voxels less than target number of sampling. Check threshold."

        sampled_indices = valid_indices[
            np.random.choice(len(valid_indices), n_points, replace=False)
        ]
        sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
        sampled_densities = vol[
            sampled_indices[:, 0],
            sampled_indices[:, 1],
            sampled_indices[:, 2],
        ]
        sampled_densities = sampled_densities * args.density_rescale

    out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    np.save(save_path, out)
    print(f"Initialization saved to {save_path}")


def main(args, init_args: InitParams, model_args: ModelParams, pipe_args: PipelineParams):
    data_path = args.data
    model_args.source_path = data_path
    scene = Scene(model_args, False)
    train_cameras = scene.getTrainCameras()
    projs_train = np.concatenate(
        [t2a(cam.original_image) for cam in train_cameras], axis=0
    )
    angles_train = np.stack([t2a(cam.angle) for cam in train_cameras], axis=0)
    scanner_cfg = scene.scanner_cfg
    geo = get_geometry(scanner_cfg)

    save_path = args.output
    if not save_path:
        if osp.exists(osp.join(data_path, "meta_data.json")):
            save_path = osp.join(data_path, "init_" + osp.basename(data_path) + ".npy")
        elif data_path.split(".")[-1] in ["pickle", "pkl"]:
            save_path = osp.join(
                osp.dirname(data_path),
                "init_" + osp.basename(data_path).split(".")[0] + ".npy",
            )
        else:
            assert False, f"Could not recognize scene type: {data_path}."

    assert not osp.exists(save_path), \
        f"Initialization file {save_path} exists! Delete it first."
    os.makedirs(osp.dirname(save_path), exist_ok=True)

    init_pcd(
        projs=projs_train,
        angles=angles_train,
        geo=geo,
        scanner_cfg=scanner_cfg,
        args=init_args,
        save_path=save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Gaussian point clouds")
    init_parser = InitParams(parser)
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--data", type=str, required=True, help="Path to data")
    parser.add_argument("--output", default=None, type=str, help="Path to output")

    args = parser.parse_args()
    main(args, init_parser.extract(args), lp.extract(args), pp.extract(args))
