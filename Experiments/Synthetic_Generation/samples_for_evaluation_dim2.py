from Experiments.Synthetic_Generation.MOG import *
from Experiments.Synthetic_Generation.samplers import *
from Experiments.CSV_read import *
from Experiments.CSV_shuffle import *
from tqdm import tqdm
import os
import json
from pathlib import Path

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
    num_samples = params["num_samples"]
    surjective_mapping = {int(key) : params["surjective_mapping"][key] for key in params["surjective_mapping"]}
    bary_MC_size = params["MC_size"]

    if dim == 2:
        bound_type = "eigen_bound"
    else:
        bound_type = "norm_bound"
    
    num_samples_in_preparation = 10**7

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"

    samplers_info_dir = f"{instance_dir}/samplers_info"
    os.makedirs(samplers_info_dir, exist_ok=True)

    source_component_seed = cfg_dict["source_components_seed"]
    master_source_rng = np.random.SeedSequence(cfg_dict["evaluation_source_sampling_seed"])
    auxiliary_seeds_list = cfg_dict["auxiliary_seeds_list"]
    master_auxiliary_rng = np.random.SeedSequence(cfg_dict["master_auxiliary_sampling_seed"])

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

    surjective_mapping_seed = cfg_dict["surjective_mapping_seed"]
    A_matrices_seed = cfg_dict["A_matrices_seed"]
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
    entropic_sampler = load_sampler(samplers_info_dir, entropic_sampler, sampler_type = "entropic")

    data_dir = f"{instance_dir}/samples_for_evaluation"
    os.makedirs(data_dir, exist_ok=True)

    # sample from the true barycenter
    bary_samples_collection = {}
    for i in range(bary_MC_size):
        print(f"Generating bary sample of MC step {i+1}/{bary_MC_size} ...")
        bary_samples = source_sampler.sample(num_samples)
        bary_samples_collection[i] = bary_samples
    
    # save as json after changing numpy array to list
    bary_samples_collection_tolist = {k: v.tolist() for k, v in bary_samples_collection.items()}
    json_path = os.path.join(data_dir, f"bary_samples_collection_dim{dim}_MCsize{bary_MC_size}_numsamples{num_samples}.json")
    with open(json_path, 'w') as json_file:
        json.dump(bary_samples_collection_tolist, json_file)

    # Generate input samples for evaluation
    csv_evaluate_dir = f"{instance_dir}/samples_for_evaluation"
    os.makedirs(csv_evaluate_dir, exist_ok=True)
    
    input_measure_samples_for_evaluation = entropic_sampler.sample(num_samples_in_preparation)

    for measure_index in range(num_measures):
        measure_samples = np.asarray(input_measure_samples_for_evaluation[measure_index])
        csv_filename = f"{csv_evaluate_dir}/input_measure_samples_{i}_for_evaluation.csv"
        pd.DataFrame(measure_samples).to_csv(csv_filename, index=False, header=False)
    print("Input samples for evaluation saved to CSV files.")


    # for i in tqdm(range(num_measures), desc="Shuffling CSV files"):
    #     old_csv_path = f"{instance_dir}/input_samples/csv_files/input_measure_samples_{i}.csv"
    #     csv_evaluate_dir = f"{instance_dir}/samples_for_evaluation"
    #     os.makedirs(csv_evaluate_dir, exist_ok=True)
    #     new_csv_path = f"{csv_evaluate_dir}/input_measure_samples_{i}_for_evaluation.csv"
    #     csv_shuffle(old_csv_path, new_csv_path, seed = 200 + i)
    

    
    # read back and change list to numpy array
    # with open(json_path, 'r') as json_file:
    #     bary_samples_collection_loaded = json.load(json_file)
    # bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}
    
