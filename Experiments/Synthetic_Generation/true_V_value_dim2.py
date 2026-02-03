import pandas as pd
from pathlib import Path
import json, os
from tqdm import tqdm

from Experiments.CSV_read import *
from Experiments.Synthetic_Generation.samplers import *
from Algorithms.data_manage import *

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
    surjective_mapping = {int(key) : params["surjective_mapping"][key] for key in params["surjective_mapping"]}

    if dim == 2:
        bound_type = "eigen_bound"
    else:
        bound_type = "norm_bound"

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"

    samplers_info_dir = f"{instance_dir}/samplers_info"
    os.makedirs(samplers_info_dir, exist_ok=True)

    source_component_seed = params["seeds"]["source_components_seed"]
    master_source_rng = np.random.SeedSequence(params["seeds"]["true_V_val_source_sampling_seed"])
    auxiliary_seeds_list = params["seeds"]["auxiliary_seeds_list"]
    master_auxiliary_rng = np.random.SeedSequence(params["seeds"]["master_auxiliary_sampling_seed"])

    source_sampler = characterize_source_sampler(dim = dim, 
                                                num_components = num_components, 
                                                master_sampling_rng = master_source_rng,
                                                component_seed = source_component_seed,
                                                truncated_radius = truncated_radius,
                                                save_dir = samplers_info_dir)

    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(dim = dim,
                                                                       num_components = num_components, 
                                                                       master_sampling_rng = master_auxiliary_rng, 
                                                                       auxiliary_seeds_list = auxiliary_seeds_list)
    
    tilde_K = len(auxiliary_measure_sampler_set)

    surjective_mapping_seed = params["seeds"]["surjective_mapping_seed"]
    A_matrices_seed = params["seeds"]["A_matrices_seed"]
    A_matrices_dict = generate_A_matrices(dim = dim, num_measures = num_measures, seed = A_matrices_seed)

    entropic_sampler = entropic_input_sampler(dim = dim, 
                                              num_measures = num_measures, 
                                              auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                              source_sampler = source_sampler, 
                                              n_k = 1000, 
                                              alpha_list = alpha_list,
                                              theta_list = theta_list,
                                              gamma = gamma, 
                                              truncated_radius = truncated_radius,
                                              bound_type = "eigen_bound",
                                              surjective_mapping = surjective_mapping,
                                              A_matrices_dict = A_matrices_dict)
    
    entropic_sampler = load_sampler(samplers_info_dir, entropic_sampler, sampler_type = "entropic")

    MC_sample_size = 10**7
    max_num_saved_samples = 10**4
    [V_mean, V_std, V_vec, distsq_mat] = entropic_sampler.compute_true_V_value(MC_sample_size)

    outputs_dir = f"{instance_dir}/outputs/true_V_value"
    output_dict = {
        "mean": V_mean,
        "std": V_std,
        "sample_size": MC_sample_size,
        "values": V_vec[:max_num_saved_samples].tolist(),
        "dist_values": distsq_mat[:max_num_saved_samples, :].tolist()
    }
    save_json(output_dict, outputs_dir, 'true_V_value.json')