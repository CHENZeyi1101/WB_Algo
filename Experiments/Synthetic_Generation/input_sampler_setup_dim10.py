from pathlib import Path
import json, os

from Experiments.Synthetic_Generation.input_generate_entropic import entropic_input_sampler, generate_A_matrices, construct_surjective_mapping
from Experiments.CSV_read import *

if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim10"]

    # take all items in params
    dim = params["dim"]

    source_info = {
        "num_components": params["num_components"],
        "component_seed": params["seeds"]["source_components_seed"],
        "master_sampling_rng": np.random.SeedSequence(params["seeds"]["master_source_sampling_seed"])
    }

    auxiliary_info = {
        "num_components": params["num_components"],
        "auxiliary_seeds_list": params["seeds"]["auxiliary_seeds_list"],
        "master_sampling_rng": np.random.SeedSequence(params["seeds"]["master_auxiliary_sampling_seed"])
    }

    truncated_radius = params["truncated_radius"]
    n_k = params["n_k"]
    instance_identifier = params["instance_identifier"]
    alpha_list = params["alpha_list"]
    theta_list = params["theta_list"]
    gamma = params["gamma"]
    num_components = params["num_components"]
    surjective_mapping_seed = params["seeds"]["surjective_mapping_seed"]
    A_matrices_seed = params["seeds"]["A_matrices_seed"]
    num_measures = len(auxiliary_info["auxiliary_seeds_list"])

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    samplers_info_dir = f"{instance_dir}/samplers_info"
    os.makedirs(samplers_info_dir, exist_ok=True)

    surjective_mapping = construct_surjective_mapping(tilde_K = num_measures, num_measures = num_measures, seed = surjective_mapping_seed)
    A_matrices = generate_A_matrices(dim = dim, num_measures = num_measures, seed = A_matrices_seed)

    entropic_sampler = entropic_input_sampler.setup(dim = dim,
                                                    source_info = source_info,
                                                    auxiliary_info = auxiliary_info,
                                                    n_k = n_k,
                                                    alpha_list = alpha_list,
                                                    theta_list = theta_list,
                                                    gamma = gamma,
                                                    truncated_radius = truncated_radius,
                                                    surjective_mapping = surjective_mapping,
                                                    A_matrices = A_matrices,
                                                    maxeig_grid_size = None,
                                                    save_dir = samplers_info_dir)

    
    