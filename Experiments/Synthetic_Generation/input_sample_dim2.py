from Experiments.Synthetic_Generation.samplers import *
import pandas as pd
from Experiments.CSV_read import *
from pathlib import Path
import json, os

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

    setup = True # whether to set up the sampler or load existing one

    if dim == 2:
        bound_type = "eigen_bound"
    else:
        bound_type = "norm_bound"

    num_samples_in_preparation = 10**7

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"

    samplers_info_dir = f"{instance_dir}/samplers_info"
    os.makedirs(samplers_info_dir, exist_ok=True)

    source_component_seed = params["seeds"]["source_components_seed"]
    master_source_rng = np.random.SeedSequence(params["seeds"]["master_source_sampling_seed"])
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
                                              A_matrices_dict = A_matrices_dict,
                                              maxeig_grid_size = 500)
    
    if setup:
        entropic_sampler = set_up_entropic_sampler(entropic_sampler, save_dir = samplers_info_dir)
    else:
        entropic_sampler = load_sampler(samplers_info_dir, entropic_sampler, sampler_type = "entropic")

    # Generate input samples
    csv_path = f"{instance_dir}/input_samples/csv_files"
    os.makedirs(csv_path, exist_ok=True)
    
    input_measure_samples = entropic_sampler.sample(num_samples_in_preparation)

    for measure_index in range(num_measures):
        measure_samples = np.asarray(input_measure_samples[measure_index])
        csv_filename = os.path.join(csv_path, f"input_measure_samples_{measure_index}.csv")
        pd.DataFrame(measure_samples).to_csv(csv_filename, index=False, header=False)
    print("Input samples saved to CSV files.")

    
    