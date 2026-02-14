import json
import os
from scipy.stats import gaussian_kde
import numpy as np
from tqdm import tqdm

from Experiments.CSV_read import csv_input_sampler_SyntheticGeneration
from Algorithms.WDHA_Kim.implementation2D.functions import frechet_mean

def build_unit_density_grids_from_samples(
    input_samples_collection,
    num_measures,
    n1=2048,
    n2=2048,
    lo=-200,                 # square domain lower bound (x,y)
    hi=200,                 # square domain upper bound (x,y)
    truncated_radius=200,   # circle radius in the SAME coordinate system as [lo,hi]
    bw_method=None,
    eps=1e-15
):
    """
    For each measure's samples (Nx2), do:
      1) KDE on [lo,hi]x[lo,hi] grid (in original coords)
      2) Truncate to circle centered at origin with radius=truncated_radius:
            density=0 outside, renormalize so integral over circle = 1
      3) Map to unit square [0,1]x[0,1] with the SAME grid resolution (n1 x n2),
         and convert density via Jacobian so it remains a proper density on [0,1]^2.
    Returns:
      dists_unit: list of (n2,n1) density grids on [0,1]^2 (outside mapped circle region = 0)
      (lo, hi, truncated_radius)
    """
    L = float(hi - lo)
    if L <= 0:
        raise ValueError(f"Require hi > lo, got lo={lo}, hi={hi}")
    if truncated_radius <= 0:
        raise ValueError(f"Require truncated_radius > 0, got {truncated_radius}")

    # collect samples
    samples_list = []
    for i in range(num_measures):
        s = np.asarray(input_samples_collection[i], dtype=float)
        if s.ndim != 2 or s.shape[1] != 2:
            raise ValueError(f"Measure {i}: expected (N,2) samples, got {s.shape}")
        samples_list.append(s)

    # Use centers to represent grid cells
    x = lo + (np.arange(n1) + 0.5) * (L / n1)
    y = lo + (np.arange(n2) + 0.5) * (L / n2)
    X, Y = np.meshgrid(x, y, indexing="xy")     # shapes (n2,n1)

    # Circle mask in original coords (centered at origin)
    mask_circle = (X * X + Y * Y) <= (truncated_radius * truncated_radius)

    pts = np.vstack([X.ravel(), Y.ravel()])     # (2, n1*n2)

    # Area per cell in original coordinates
    dA_xy = (L / n1) * (L / n2)

    dists_unit_density = []
    for s in tqdm(samples_list, desc="KDE -> circle-truncated -> unit density"):
        kde = gaussian_kde(s.T, bw_method=bw_method)

        dens_xy = kde(pts).reshape(n2, n1)  # density w.r.t (x,y)  # density values inside circle

        # dens_xy is (n2, n1)
        dens_xy_trunc = dens_xy * mask_circle
        mass_in_circle = float(dens_xy_trunc.sum() * dA_xy)

        if mass_in_circle <= eps:
            dens_xy_trunc[:] = 0.0
        else:
            dens_xy_trunc /= mass_in_circle  # now total mass on circle = 1

        # map to unit square coords u=(x-lo)/L, v=(y-lo)/L with same grid count
        # Density transform by multiplying by Jacobian factor (i.e. the area scaling)
        dens_uv = dens_xy_trunc * (L * L)
        dists_unit_density.append(dens_uv)

    return dists_unit_density


if __name__ == "__main__":
    
    from pathlib import Path

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]
    dim = params["dim"]
    truncated_radius = params["truncated_radius"]
    instance_identifier = params["instance_identifier"]
    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    num_samples = params["num_samples"]
    num_measures = params["num_measures"]
    MC_size = params["MC_size"]

    hi = truncated_radius
    lo = -truncated_radius

    n1, n2 = 2048, 2048  # grid size for WGHA_Kim

    outputs_dir = f"{instance_dir}/outputs/WDHA_Kim_outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    input_csv_dir = f"{instance_dir}/input_samples/csv_files"
    input_sampler = csv_input_sampler_SyntheticGeneration(input_csv_dir, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler.set_streamers()

    input_samples_collection = input_sampler.sample(num_samples)

    dists_unit_density = build_unit_density_grids_from_samples(
                                                        input_samples_collection,
                                                        num_measures,
                                                        n1=n1,
                                                        n2=n2,
                                                        lo=lo,                 
                                                        hi=hi,                
                                                        truncated_radius=truncated_radius,   
                                                        bw_method=None,
                                                        eps=1e-15
                                                    )
    
    # save densities in dists_unit_density
    for i, dens_uv in enumerate(dists_unit_density):
        np.save(f"{outputs_dir}/density_unit_{i}.npy", dens_uv)
    
    mu_WGHA_unit = frechet_mean(dists_unit_density, 500, 'barycenter', save_option = False, return_option = True)
    np.save(f"{outputs_dir}/barycenter_density_unit.npy", mu_WGHA_unit)

    # mu_WGHA_scaled = mu_WGHA_unit / (L * L)  # density w.r.t (x,y) coords

    