import os
import sys
import numpy as np
import tigre
import os.path as osp
import yaml
import time
import tigre.algorithms as algs
from tqdm import trange
from tigre.utilities.im3Dnorm import im3DNORM
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_gradient_magnitude


sys.path.append("./")
from xfield.utils.image_utils import metric_vol


def apply_tv_regularization(volume, tv_lambda=0.02, num_iterations=10):
    """
    Apply Total Variation regularization to the volume.
    """
    for _ in range(num_iterations):
        # Compute the gradient magnitude (can use any approximation method)
        grad = gaussian_gradient_magnitude(volume, sigma=1)
        
        # Update the volume by reducing gradient magnitude scaled by tv_lambda
        volume = volume - tv_lambda * grad
    return volume



def hybrid_cgls_sart_tv(projs, angles, geo, cgls_iters=30, sart_iters=20, tv_iters=10, tv_lambda=0.02):
    """
    Hybrid Progressive Initialization: CGLS -> SART -> TV

    1. CGLS: Conjugate Gradient Least Squares for coarse global estimate
    2. SART: Simultaneous Algebraic Reconstruction Technique for local details
    3. TV: Total Variation regularization to suppress noise and preserve edges
    """
    print("Step 1: CGLS for coarse global estimate...")
    vol = algs.cgls(projs, geo, angles, niter=cgls_iters)

    print("Step 2: SART for local detail enhancement...")
    vol = algs.sart(projs, geo, angles, niter=sart_iters, init=vol)

    print("Step 3: TV regularization for noise suppression...")
    vol = apply_tv_regularization(vol, tv_lambda=tv_lambda, num_iterations=tv_iters)

    print("Hybrid initialization completed")
    return vol



def recon_volume(projs, angles, geo, recon_method):
    """Reconstruct CT with traditional methods."""
    if recon_method == "fdk":
        vol = algs.fdk(projs[:, ::-1, :], geo, angles)
    elif recon_method == "cgls":
        vol, _ = algs.cgls(projs[:, ::-1, :], geo, angles, 60, computel2=True)
    elif recon_method == "sart":
        vol = algs.sart(projs[:, ::-1, :], geo, angles, 100)
    elif recon_method == "ossart":
        vol = algs.ossart(projs[:, ::-1, :], geo, angles, 100)
    elif recon_method == "mlem":
        vol = algs.mlem(projs[:, ::-1, :], geo, angles, 100)
    elif recon_method == "osem":
        vol = algs.osem(projs[:, ::-1, :], geo, angles, 100)
    elif recon_method == "grad_desc":
        vol = algs.gradient(projs[:, ::-1, :], geo, angles, 100)
    elif recon_method == "tv":
        vol = algs.tv(projs[:, ::-1, :], geo, angles, 100, regularization=0.02)
    elif recon_method == "hybrid_cgls_sart_tv":
        vol = hybrid_cgls_sart_tv(projs[:, ::-1, :], angles, geo, cgls_iters=30, sart_iters=20, tv_iters=10, tv_lambda=0.02)
    else:
        raise ValueError("Unsupported reconstruction method")
    
    vol = np.transpose(vol, (2, 1, 0))
    return vol

def get_geometry(cfg):
    if cfg["mode"] == "parallel":
        geo = tigre.geometry(mode="parallel", nVoxel=np.array(cfg["nVoxel"][::-1]))
    elif cfg["mode"] == "cone":
        geo = tigre.geometry(mode="cone")
    else:
        raise NotImplementedError("Unsupported scanner mode!")

    geo.DSD = cfg["DSD"]  # Distance Source Detector
    geo.DSO = cfg["DSO"]  # Distance Source Origin
    # Detector parameters
    geo.nDetector = np.array(cfg["nDetector"])  # number of pixels
    geo.sDetector = np.array(cfg["sDetector"])  # size of each pixel
    geo.dDetector = geo.sDetector / geo.nDetector  # total size of the detector
    # Image parameters
    geo.nVoxel = np.array(cfg["nVoxel"][::-1])  # number of voxels
    geo.sVoxel = np.array(cfg["sVoxel"][::-1])  # size of each voxel
    geo.dVoxel = geo.sVoxel / geo.nVoxel  # total size of the image
    # Offsets
    geo.offOrigin = np.array(cfg["offOrigin"][::-1])  # Offset of image from origin
    geo.offDetector = np.array(
        [cfg["offDetector"][1], cfg["offDetector"][0], 0]
    )  # Offset of Detector
    # Auxiliary
    geo.accuracy = cfg["accuracy"]  # Accuracy of FWD proj
    # Mode
    geo.filter = cfg["filter"]
    return geo


def run_ct_recon_algs(projs, angles, geo, ct_gt, save_path, method):
    print("Run {}...".format(method))
    save_path = osp.join(save_path, method)
    slice_save_path = osp.join(save_path, "slice_{}".format(method))
    os.makedirs(slice_save_path, exist_ok=True)
    start_time = time.time()
    

    if method == "fdk":
        ct_pred = algs.fdk(projs[:, ::-1, :], geo, angles)
    elif method == "sart":
        lmbda = 1
        lambdared = 0.999
        initmode = None
        verbose = True
        qualmeas = ["RMSE", "SSIM", "PSNR"]
        blcks = 10
        order = "ordered"
        ct_pred, _ = algs.sart(
            projs[:, ::-1, :],
            geo,
            angles,
            50,
            lmbda=lmbda,
            lmbda_red=lambdared,
            verbose=verbose,
            Quameasopts=qualmeas,
            computel2=True,
        )
    elif method == "ossart":
        lmbda = 1
        lambdared = 0.999
        initmode = None
        verbose = True
        qualmeas = ["RMSE"]
        blcks = 10
        order = "ordered"
        ct_pred, qualityOSSART = algs.ossart(
            projs[:, ::-1, :],
            geo,
            angles,
            20,
            lmbda=lmbda,
            lmbda_red=lambdared,
            verbose=verbose,
            Quameasopts=qualmeas,
            computel2=False,
            blocksize=blcks,
            OrderStrategy=order,
        )
    elif method == "asd_pocs":
        epsilon = (
            im3DNORM(
                tigre.Ax(algs.fdk(projs[:, ::-1, :], geo, angles), geo, angles)
                - projs[:, ::-1, :],
                2,
            )
            * 0.15
        )
        alpha = 0.002
        ng = 20
        lmbda = 1
        lambdared = 0.9999
        alpha_red = 0.95
        ratio = 0.94
        verb = True
        order = "ordered"
        ct_pred = algs.asd_pocs(
            projs[:, ::-1, :],
            geo,
            angles,
            10,  # these are very important
            tviter=ng,
            maxl2err=epsilon,
            alpha=alpha,  # less important.
            lmbda=lmbda,
            lmbda_red=lambdared,
            rmax=ratio,
            verbose=verb,
        )
    elif method == "os_asd_pocs":
        epsilon = (
            im3DNORM(
                tigre.Ax(algs.fdk(projs[:, ::-1, :], geo, angles), geo, angles)
                - projs[:, ::-1, :],
                2,
            )
            * 0.15
        )
        alpha = 0.002
        ng = 20
        lmbda = 1
        lambdared = 0.9999
        alpha_red = 0.95
        ratio = 0.94
        verb = True
        order = "ordered"
        blcks = 10
        ct_pred = algs.os_asd_pocs(
            projs[:, ::-1, :],
            geo,
            angles,
            10,  # these are very important
            tviter=ng,
            maxl2err=epsilon,
            alpha=alpha,  # less important.
            lmbda=lmbda,
            lmbda_red=lambdared,
            rmax=ratio,
            verbose=verb,
            OrderStrategy=order,
            blocksize=blcks,
        )
    elif method == "cgls":
        ct_pred, _ = algs.cgls(projs[:, ::-1, :], geo, angles, 60, computel2=True)
    else:
        raise NotImplementedError("Unsupported reconstruction method!")
    ct_pred = ct_pred.transpose((2, 1, 0))

    duration = time.time() - start_time

    psnr_3d, _ = metric_vol(ct_gt, ct_pred, "psnr")
    ssim_3d, ssim_3d_axis = metric_vol(ct_gt, ct_pred, "ssim")

    np.save(osp.join(save_path, "ct_gt.npy"), ct_gt)
    np.save(osp.join(save_path, "ct_pred.npy"), ct_pred)

    n_slice = ct_gt.shape[2]
    for i_slice in trange(n_slice, desc="[{}] Save slice".format(method), leave=False):
        plt.imsave(
            osp.join(slice_save_path, "{0:05d}_gt.png".format(i_slice)),
            ct_gt[:, :, i_slice],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plt.imsave(
            osp.join(slice_save_path, "{0:05d}_pred.png".format(i_slice)),
            ct_pred[:, :, i_slice],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    report_dict = {
        "method": method,
        "psnr_3d": psnr_3d,
        "ssim_3d": float(ssim_3d),
        "ssim_3d_x": ssim_3d_axis[0],
        "ssim_3d_y": ssim_3d_axis[1],
        "ssim_3d_z": ssim_3d_axis[2],
        "duration (sec)": duration,
        "duration (min)": duration / 60,
    }
    with open(osp.join(save_path, "eval_3d.yml"), "w") as f:
        yaml.dump(report_dict, f, default_flow_style=False, sort_keys=False)

    print("[{}] psnr_3d: {}, ssim_3d: {}".format(method, psnr_3d, ssim_3d))
    return report_dict, ct_pred, ct_gt
