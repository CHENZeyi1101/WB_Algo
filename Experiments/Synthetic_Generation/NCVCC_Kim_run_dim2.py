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

def map_in_points(samples_xy, lo, hi):
    """true -> unit"""
    s = np.asarray(samples_xy, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    scale = hi - lo
    if np.any(scale <= 0):
        raise ValueError(f"Invalid bounds: lo={lo}, hi={hi}")
    return (s - lo) / scale

def map_out_points(unit_xy, lo, hi):
    """unit -> true"""
    u = np.asarray(unit_xy, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    return lo + u * (hi - lo)

def unit_grid_centers(n1, n2):
    """Matches your frechet_mean's meshgrid centers."""
    x_u = np.linspace(0.5/n1, 1 - 0.5/n1, n1)
    y_u = np.linspace(0.5/n2, 1 - 0.5/n2, n2)
    return x_u, y_u

def kde_to_unit_grid_mass(samples_unit, n1, n2, bw_method=None, eps=1e-12):
    samples_unit = np.asarray(samples_unit, float)
    if samples_unit.ndim != 2 or samples_unit.shape[1] != 2:
        raise ValueError(f"Expected (N,2), got {samples_unit.shape}")

    kde = gaussian_kde(samples_unit.T, bw_method=bw_method)

    x_u, y_u = unit_grid_centers(n1, n2)
    X, Y = np.meshgrid(x_u, y_u, indexing="xy")
    pts = np.vstack([X.ravel(), Y.ravel()])

    pdf = kde(pts).reshape(Y.shape)  # (n2,n1), units 1/area on unit square

    dx = float(x_u[1] - x_u[0]) if n1 > 1 else 1.0 
    dy = float(y_u[1] - y_u[0]) if n2 > 1 else 1.0 
    
    mass = pdf * dx * dy 
    s = mass.sum() 
    if not np.isfinite(s) or s < eps:
        raise ValueError("KDE mass is invalid; check bw_method/support.") 
    mass /= s 
    
    return mass


def build_unit_mass_grids_from_samples(
    input_samples_collection,
    n1=2048,
    n2=2048,
    pad_frac=0.02,
    bw_method=None
):
    samples_list = []
    for i in range(num_measures):
        measure_input_samples = input_samples_collection[i]
        samples_list.append(np.asarray(measure_input_samples))

    # Infer true support
    lo, hi = infer_global_support(samples_list, pad_frac=pad_frac)
    print(f"[support] lo={lo}, hi={hi}")

    # MAP IN: true -> unit
    unit_samples_list = [map_in_points(s, lo, hi) for s in samples_list]

    # KDE -> unit grid mass
    dists = []
    for s_u in tqdm(unit_samples_list, desc="KDE to unit-grid mass"):
        dists.append(kde_to_unit_grid_mass(s_u, n1=n1, n2=n2, bw_method=bw_method))

    return dists, (lo, hi)

def plot_mass_true_axes(mass_unit, lo, hi, title="NCVCC barycenter", save_path=None):
    n2, n1 = mass_unit.shape
    x_u, y_u = unit_grid_centers(n1, n2)

    # MAP OUT grid centers to true coordinates
    x_phys = lo[0] + x_u * (hi[0] - lo[0])
    y_phys = lo[1] + y_u * (hi[1] - lo[1])

    plt.figure(figsize=(6, 5))
    plt.imshow(
        mass_unit,
        cmap = "hot",
        origin="lower",
        extent=[x_phys[0], x_phys[-1], y_phys[0], y_phys[-1]],
        aspect="auto"
    )
    plt.colorbar(label="probability mass per cell")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def sample_from_unit_mass_grid(mass_unit, lo, hi, num_samples, seed=None):
    mass = np.asarray(mass_unit, float)
    mass = np.clip(mass, 0, None)
    mass /= mass.sum()

    n2, n1 = mass.shape
    rng = np.random.default_rng(seed)

    idx = rng.choice(n1 * n2, size=num_samples, p=mass.ravel())
    y_idx, x_idx = np.unravel_index(idx, (n2, n1))

    x_u, y_u = unit_grid_centers(n1, n2)
    samples_unit = np.column_stack([x_u[x_idx], y_u[y_idx]])

    # MAP OUT: unit -> true
    samples = map_out_points(samples_unit, lo, hi)
    return samples


if __name__ == "__main__":
    
    from pathlib import Path

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]

    # take all items in params
    num_samples = params["num_samples"]
    dim = params["dim"]
    num_measures = params["num_measures"]
    truncated_radius = params["truncated_radius"]
    instance_identifier = params["instance_identifier"]
    num_components = params["num_components"]
    MC_size = params["MC_size"]

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    n1, n2 = 1024, 1024

    source_component_seed = cfg_dict["source_components_seed"]
    master_source_rng = np.random.SeedSequence(cfg_dict["master_source_sampling_seed"])

    outputs_dir = f"{instance_dir}/outputs/NCVCC_Kim_outputs"
    os.makedirs(outputs_dir, exist_ok=True)


    input_csv_dir = f"{instance_dir}/input_samples/csv_files"
    input_sampler = csv_input_sampler_SyntheticGeneration(input_csv_dir, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler.set_streamers()

    bary_sample_path = f"{instance_dir}/samples_for_evaluation/bary_samples_collection_dim{dim}_MCsize{MC_size}_numsamples{num_samples}.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}
 
    eval_dir = f"{instance_dir}/samples_for_evaluation"
    input_sampler_for_evaluation = csv_input_sampler_for_evaluation_SyntheticGeneration(eval_dir, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler_for_evaluation.set_streamers()

    input_samples_collection = input_sampler.sample(num_samples)

    dists_unit_mass, (lo, hi) = build_unit_mass_grids_from_samples(input_samples_collection,
                                                                   n1 = n1, n2 = n2,
                                                                   pad_frac=0.02,
                                                                   bw_method=None)
    
    mu_WGHA_unit = frechet_mean(dists_unit_mass, 500, 'barycenter', save_option = False, return_option = True)
    np.save(f"{outputs_dir}/barycenter_density.npy", mu_WGHA_unit)
    plot_mass_true_axes(mu_WGHA_unit, lo, hi, title="NCVCC barycenter", save_path=f"{outputs_dir}/NCVCC_barycenter.png")

    # evaluation
    V_values = []
    W2_distances = []
    V_values_path = os.path.join(outputs_dir, f"V_values_NCVCC_Kim_MCsize{MC_size}.json")
    W2_distances_path = os.path.join(outputs_dir, f"W2_distances_NCVCC_Kim_MCsize{MC_size}.json")

    for mc in range(MC_size):
        print(f"Starting MC run {mc+1}/{MC_size} ...")
        bary_samples = bary_samples_collection_loaded[str(mc)]
        input_samples_collection = input_sampler_for_evaluation.sample(num_samples)
        samples = sample_from_unit_mass_grid(mu_WGHA_unit, lo, hi, num_samples=1000, seed=mc + 1000)
        # Evaluate metrics
        V_value = V_value_compute(samples, input_samples_collection)
        W2_distance = W2_to_bary_compute(bary_samples, samples)

        V_values.append(V_value)
        W2_distances.append(W2_distance)

        with open(V_values_path, 'w') as json_file:
            json.dump(V_values, json_file)
        with open(W2_distances_path, 'w') as json_file:
            json.dump(W2_distances, json_file)

       