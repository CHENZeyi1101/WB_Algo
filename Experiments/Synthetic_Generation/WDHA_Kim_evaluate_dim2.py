import matplotlib.pyplot as plt

from Experiments.CSV_read import *
from Experiments.metrics_to_compare import evaluate_MC
from Algorithms.data_manage import *

def plot_density_heatmap(dens_xy, lo, hi, title=None, show_colorbar=True, plot_dir = None, filename = None):
    """
    Plot a density heatmap on the physical coordinates [lo,hi]x[lo,hi].

    dens_xy: (n2,n1) density values w.r.t. x,y
    """
    dens_xy = np.asarray(dens_xy, float)
    if dens_xy.ndim != 2:
        raise ValueError(f"Expected 2D array, got {dens_xy.shape}")    

    fig, ax = plt.subplots()
    im = ax.imshow(
        dens_xy,
        origin="lower",
        extent=[lo, hi, lo, hi],   # maps array coordinates to physical coords
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax, label="density")

    plt.tight_layout()
    if plot_dir is not None and filename is not None:
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close(fig)
    else:
        plt.show()
    return fig, ax

def sample_from_density_grid(density_xy, lo, hi, num_samples, seed=None):
    """
    Sample points from a 2D probability density defined on a grid.

    Parameters
    ----------
    density_xy : ndarray, shape (n2, n1)
        Density on [lo,hi] x [lo,hi]
    lo, hi : float
        Domain bounds
    n_samples : int
        Number of samples to draw

    Returns
    -------
    samples : ndarray, shape (n_samples, 2)
        Sampled (x,y) points
    """
    rng = np.random.default_rng(seed)  # <-- controlled RNG

    density_xy = np.asarray(density_xy, float)
    n2, n1 = density_xy.shape
    L = hi - lo

    # Cell area
    dA = (L / n1) * (L / n2)

    # Convert density -> probability mass per cell
    prob = density_xy * dA
    total_mass = prob.sum()

    if not np.isclose(total_mass, 1.0, atol=1e-6):
        raise ValueError(f"Density not normalized, total mass = {total_mass}")

    prob_flat = prob.ravel()

    # Sample cell indices
    idx = rng.choice(
        prob_flat.size,
        size=num_samples,
        p=prob_flat
    )

    # Convert flat indices -> (i,j)
    j, i = np.divmod(idx, n1)

    # Sample uniformly inside each chosen cell
    x = lo + (i + rng.random(num_samples)) * (L / n1)
    y = lo + (j + rng.random(num_samples)) * (L / n2)

    samples = np.column_stack([x, y])
    return samples


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
    eval_num_samples = params["eval_num_samples"]
    num_measures = params["num_measures"]
    MC_size = params["MC_size"]

    hi = truncated_radius
    lo = -truncated_radius
    L = hi - lo

    eval_dir = f"{instance_dir}/samples_for_evaluation"

    bary_sample_path = f"{eval_dir}/bary_samples_collection.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {int(k): np.array(v) for k, v in bary_samples_collection_loaded.items()}

    input_sample_path = f"{eval_dir}/input_samples_collection.json"
    with open(input_sample_path, 'r') as json_file:
        input_samples_collection_loaded = json.load(json_file)
    input_samples_collection_loaded = {int(k): {int(i): np.array(u) for i, u in v.items()}
                                        for k, v in input_samples_collection_loaded.items()}

    outputs_dir = f"{instance_dir}/outputs/WDHA_Kim_outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    V_values_dir = os.path.join(outputs_dir, "V_values")
    W2_to_bary_dir = os.path.join(outputs_dir, "W2_to_bary")
    os.makedirs(V_values_dir, exist_ok=True)
    os.makedirs(W2_to_bary_dir, exist_ok=True)

    # Plotting densities (scaled to unit support)

    plot_dir = f"{outputs_dir}/density_plots"
    os.makedirs(plot_dir, exist_ok=True)

    for i in range(num_measures):
        density_unit_i = np.load(f"{outputs_dir}/density_unit_{i}.npy")
        plot_density_heatmap(density_unit_i, 0, 1, title=f"WDHA input density {i} (unit mass)", show_colorbar=True, plot_dir = plot_dir, filename = f"density_unit_{i}.png")
    
    mu_WGHA_unit = np.load(f"{outputs_dir}/barycenter_density_unit.npy")
    plot_density_heatmap(mu_WGHA_unit, 0, 1, title="WDHA barycenter density (unit mass)", show_colorbar=True, plot_dir = plot_dir, filename = "barycenter_density_unit.png")

    mu_WGHA_scaled = mu_WGHA_unit / (L * L)  # density w.r.t (x,y) coords
    plot_density_heatmap(mu_WGHA_scaled, lo, hi, title="WDHA barycenter density (scaled)", show_colorbar=True, plot_dir = plot_dir, filename = "barycenter_density_scaled.png")

    # sample from the approximated barycenter and evaluate

    sampling_seed_list = [2000 + i for i in range(MC_size)]
    approx_bary_it = [sample_from_density_grid(mu_WGHA_scaled, lo, hi, num_samples=eval_num_samples, seed = sampling_seed_list[i]) for i in range(MC_size)]
    input_measure_samples_collection_it = [{k : input_samples_collection_loaded[i][k][:eval_num_samples] for k in range(num_measures)} for i in range(MC_size)]
    true_bary_samples_it = [bary_samples_collection_loaded[i][:eval_num_samples] for i in range(MC_size)]
    
    V_values_list, W2_to_bary_list = evaluate_MC(approx_bary_it, 
                                                 input_measure_samples_collection_it, 
                                                 true_bary_samples_it, 
                                                 MC_size = MC_size, 
                                                 num_parallel_process = 5, 
                                                 pbar_text = "Evaluation of WDHA_Kim")

    # save V-values and W2_to_bary values
    V_values_dict = {
        "mean": np.mean(V_values_list),
        "std": np.std(V_values_list),
        "values": V_values_list}
    save_json(V_values_dict, V_values_dir, "V_values.json")

    W2_to_bary_dict = {
        "mean": np.mean(W2_to_bary_list),
        "std": np.std(W2_to_bary_list),
        "values": W2_to_bary_list}
    save_json(W2_to_bary_dict, W2_to_bary_dir, "W2_to_bary.json")

    
