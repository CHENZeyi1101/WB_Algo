import json
from Experiments.CSV_read import *
from Algorithms.WDHA_Kim.implementation2D.functions import *
from Experiments.Synthetic_Generation.metrics_to_compare import *
import os

import numpy as np

from scipy.stats import gaussian_kde

def infer_global_support(samples_list, pad_frac=0.02):
    """
    Infer a single (global) bounding box that covers all measures' samples.

    samples_list: list of arrays, each (N_i, 2)
    pad_frac: expand bounds by this fraction of range on each side
    quantile_clip: None or tuple (q_low, q_high) to clip outliers before bounds
    """
    all_xy = np.vstack(samples_list)  # (sum N_i, 2)

    lo = all_xy.min(axis=0) 
    hi = all_xy.max(axis=0)

    span = hi - lo
    lo = lo - pad_frac * span
    hi = hi + pad_frac * span
    return lo, hi  # each shape (2,)


def kde_to_grid(samples_xy, x_grid, y_grid, bw_method=None, eps=1e-12):
    """
    KDE (2D) evaluated on a grid, returned as array shape (n2, n1)
    where n1=len(x_grid), n2=len(y_grid).

    bw_method: passed to gaussian_kde (None, 'scott', 'silverman', or float)
    """
    samples_xy = np.asarray(samples_xy)
    if samples_xy.ndim != 2 or samples_xy.shape[1] != 2:
        raise ValueError(f"KDE grid expects samples of shape (N, 2); got {samples_xy.shape}")

    # gaussian_kde expects data as (d, N)
    kde = gaussian_kde(samples_xy.T, bw_method=bw_method)

    # Build evaluation points of shape (2, n1*n2)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")  # X,Y each (n2, n1)
    pts = np.vstack([X.ravel(), Y.ravel()])            # (2, n2*n1)

    Z = kde(pts).reshape(Y.shape)  # (n2, n1)
    s = Z.sum()
    if not np.isfinite(s) or s < eps:
        raise ValueError("KDE produced near-zero or invalid total mass; check samples/bandwidth/support.")
    Z /= s # normalize to sum to 1
    return Z


def build_density_grids_from_measures(
    input_samples_collection,
    n1=2048,
    n2=2048,
    pad_frac=0.02,
    bw_method=None,
    return_grids=True,
):
    """
    input_samples_collection: list-like, length=num_measures, each element is samples (N_i, 2)
    Returns:
      densities: list of arrays, each (n2, n1), normalized so sum = n1*n2
      plus optionally the x_grid,y_grid and bounds
    """
    samples_list = []
    for i in range(num_measures):
        measure_input_samples = input_samples_collection[i]
        samples_list.append(np.asarray(measure_input_samples))

    # Infer global support across ALL measures
    lo, hi = infer_global_support(samples_list, pad_frac=pad_frac)
    print(f"Inferred global support: lo={lo}, hi={hi}")

    # Build common grids
    x_grid = np.linspace(lo[0], hi[0], n1)
    y_grid = np.linspace(lo[1], hi[1], n2)

    # KDE each measure on the SAME grid
    dists = []
    for i, sxy in tqdm(enumerate(samples_list), desc="Building density grids from measures", total=len(samples_list)):
        dens = kde_to_grid(sxy, x_grid, y_grid, bw_method=bw_method)
        dists.append(dens)

    if return_grids:
        return dists, x_grid, y_grid, (lo, hi)
    return dists

def sample_from_density_grid(density_grid, x_grid, y_grid, num_samples, seed=None):
    n2, n1 = density_grid.shape
    flat = density_grid.ravel().astype(float)
    flat /= flat.sum()

    rng = np.random.default_rng(seed)
    idx = rng.choice(n1 * n2, size=num_samples, p=flat)

    y_idx, x_idx = np.unravel_index(idx, (n2, n1))
    samples = np.column_stack([x_grid[x_idx], y_grid[y_idx]])
    return samples

import matplotlib.pyplot as plt
import numpy as np

def plot_density(rd, title="Fréchet mean density", cmap="hot", save_path = None):
    n2, n1 = rd.shape

    x_grid = np.linspace(0.5/n1, 1 - 0.5/n1, n1)
    y_grid = np.linspace(0.5/n2, 1 - 0.5/n2, n2)

    plt.figure(figsize=(6, 5))
    plt.imshow(
        rd,
        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
        origin="lower",
        cmap=cmap,
        aspect="equal"
    )
    plt.colorbar(label="Density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    from pathlib import Path

    num_samples = 1000
    dim = 2
    num_measures = 5
    truncated_radius = 150
    instance_theta = 2000
    n1, n2 = 512, 512
    MC_size = 1

    instance_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}_toy"
    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    SEEDS_PATH = Path(__file__).parent / "seeds.json"
    with open(SEEDS_PATH, "r") as f:
        seeds_dict = json.load(f)

    source_component_seed = seeds_dict["source_components_seed"]
    master_source_rng = np.random.SeedSequence(seeds_dict["master_source_sampling_seed"])

    save_path = f"{instance_dir}/outputs/NCVCC_Kim_outputs"
    os.makedirs(save_path, exist_ok=True)
    
    input_csv_path = f"{instance_dir}/input_samples/csv_files"
    input_sampler = csv_input_sampler_SyntheticGeneration(input_csv_path, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler.set_streamers()

    bary_sample_path = f"{instance_dir}/samples_for_evaluation/bary_samples_collection_dim{dim}_MCsize50_numsamples1000.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}
 
    eval_dir = f"{instance_dir}/samples_for_evaluation"
    input_sampler_for_evaluation = csv_input_sampler_for_evaluation_SyntheticGeneration(eval_dir, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler_for_evaluation.set_streamers()

    input_samples_collection = input_sampler.sample(num_samples)
    
    dists = build_density_grids_from_measures(
        input_samples_collection,
        n1,
        n2,
        pad_frac=0.02,
        bw_method=None,
        return_grids=False
    )
    for i, dist in enumerate(dists):
        print(f"Measure {i} density sum: {dist.sum()}")

    # Run NCVCC Kim
    mu_WGHA = frechet_mean(dists, 500, 'barycenter', save_option = False, return_option = True)
    np.save(f"{save_path}/barycenter_density.npy", mu_WGHA)
    plot_density(mu_WGHA, title="Fréchet mean density via NCVCC", save_path = f"{save_path}/barycenter_density_NCVCC_Kim.png")

    # evaluation
    V_values = []
    W2_distances = []
    V_values_path = os.path.join(save_path, f"V_values_NCVCC_Kim_MCsize{MC_size}.json")
    W2_distances_path = os.path.join(save_path, f"W2_distances_NCVCC_Kim_MCsize{MC_size}.json")

    n2, n1 = mu_WGHA.shape
    x_grid = np.linspace(0.5/n1, 1 - 0.5/n1, n1)
    y_grid = np.linspace(0.5/n2, 1 - 0.5/n2, n2)

    for mc in range(MC_size):
        print(f"Starting MC run {mc+1}/{MC_size} ...")
        bary_samples = bary_samples_collection_loaded[str(mc)]
        input_samples_collection = input_sampler_for_evaluation.sample(num_samples)
        samples_WGHA = sample_from_density_grid(mu_WGHA, x_grid, y_grid, num_samples=10000, seed=mc+100)
        # Evaluate metrics
        V_value = V_value_compute(samples_WGHA, input_samples_collection)
        W2_distance = W2_to_bary_compute(bary_samples, samples_WGHA)

        V_values.append(V_value)
        W2_distances.append(W2_distance)

        with open(V_values_path, 'w') as json_file:
            json.dump(V_values, json_file)
        with open(W2_distances_path, 'w') as json_file:
            json.dump(W2_distances, json_file)

       