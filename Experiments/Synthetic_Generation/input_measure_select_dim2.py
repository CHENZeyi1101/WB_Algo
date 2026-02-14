from pathlib import Path
import json, os
from tqdm import tqdm

from Experiments.Synthetic_Generation.visualize_measures_dim2 import plot_2d_gm_pdf
from Experiments.Synthetic_Generation.MOG import MixtureOfGaussians

if __name__ == "__main__":
    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    
    params = cfg_dict["params_synthetic_generation_dim2"]

    # take all items in params
    dim = params["dim"]
    num_measures = params["num_measures"]
    truncated_radius = params["truncated_radius"]
    instance_identifier = params["instance_identifier"]
    alpha_list = params["alpha_list"]
    theta_list = params["theta_list"]
    gamma = params["gamma"]
    num_components = params["num_components"]

    plot_measure_selection = True

    plot_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    os.makedirs(plot_dir, exist_ok=True)
    
    if plot_measure_selection:  # decision: component_seed = 1009
        # select measures over several random seeds
        for seed in tqdm(range(1000, 1050), desc="Plotting measures for different seeds"):
            source_sampler = MixtureOfGaussians(dim, master_sampling_rng=42, component_seed=seed)
            source_sampler.random_components(num_components = num_components, uniform_weights = True)
            source_sampler.set_truncation(truncated_radius)
            plot_name = f"seed_{seed}_measure.png"
            plot_2d_gm_pdf(source_sampler, truncated_radius, grid_size=1000, plot_contour=False, plot_dirc=f"{plot_dir}/measure_selection", plot_name=plot_name, title = f"Measure (Seed {seed})")
