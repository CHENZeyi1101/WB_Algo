from Experiments.Synthetic_Generation.samplers import *
import pandas as pd
from Experiments.CSV_read import *
from pathlib import Path
import json, os
from tqdm import tqdm

if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]

    # take all items in params
    dim = params["dim"]
    num_measures = params["num_measures"]
    truncated_radius = params["truncated_radius"]
    instance_theta = params["instance_theta"]
    num_components = params["num_components"]

    if dim == 2:
        bound_type = "eigen_bound"
    else:
        bound_type = "norm_bound"

    num_samples_in_preparation = 10000

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}"

    samplers_info_dir = f"{instance_dir}/samplers_info"
    os.makedirs(samplers_info_dir, exist_ok=True)

    source_component_seed = cfg_dict["source_components_seed"]
    true_V_val_source_rng = np.random.SeedSequence(cfg_dict["true_V_val_source_sampling_seed"])
    auxiliary_seeds_list = cfg_dict["auxiliary_seeds_list"]
    master_auxiliary_rng = np.random.SeedSequence(cfg_dict["master_auxiliary_sampling_seed"])

    source_sampler = characterize_source_sampler(dim = dim, 
                                                num_components = num_components, 
                                                master_sampling_rng = true_V_val_source_rng,
                                                component_seed = source_component_seed,
                                                truncated_radius = truncated_radius,
                                                save_dir = samplers_info_dir)

    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(dim = dim,
                                                                       num_components = num_components, 
                                                                       master_sampling_rng = master_auxiliary_rng, 
                                                                       auxiliary_seeds_list = auxiliary_seeds_list)
    
    tilde_K = len(auxiliary_measure_sampler_set)

    surjective_mapping_seed = cfg_dict["surjective_mapping_seed"]
    A_matrices_seed = cfg_dict["A_matrices_seed"]
    surjective_mapping = construct_surjective_mapping(tilde_K = tilde_K, num_measures = num_measures, seed = surjective_mapping_seed)
    A_matrices_dict = generate_A_matrices(dim = dim, num_measures = num_measures, seed = A_matrices_seed)

    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                     num_measures = num_measures, 
                                                     auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                     source_sampler = source_sampler,
                                                     truncated_radius = truncated_radius,
                                                     manual = False,
                                                     bound_type = bound_type,
                                                     theta = instance_theta,
                                                     surjective_mapping = surjective_mapping,
                                                     A_matrices_dict = A_matrices_dict)
    
    entropic_sampler = load_sampler(samplers_info_dir, entropic_sampler, sampler_type = "entropic")

    