from Experiments.Synthetic_Generation.samplers import *
from tqdm import tqdm
import pandas as pd
import pickle
from Experiments.CSV_read import *
import math
from scipy.linalg import sqrtm, norm
from pathlib import Path
import json, os

if __name__ == "__main__":
    dim = 10
    num_components = 5
    num_measures = 10
    truncated_radius = 5000
    instance_theta = 2000

    if dim == 2:
        bound_type = "eigen_bound"
    else:
        bound_type = "norm_bound"

    num_samples_in_preparation = int(1e7)

    samplers_info_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}/samplers_info"
    os.makedirs(samplers_info_dir, exist_ok=True)

    SEEDS_PATH = Path(__file__).parent / "seeds.json"
    with open(SEEDS_PATH, "r") as f:
        seeds_dict = json.load(f)

    source_component_seed = seeds_dict["source_components_seed"]
    master_source_rng = np.random.SeedSequence(seeds_dict["master_source_sampling_seed"])
    auxiliary_seeds_list = seeds_dict["auxiliary_seeds_list"]
    master_auxiliary_rng = np.random.SeedSequence(seeds_dict["master_auxiliary_sampling_seed"])

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

    surjective_mapping_seed = seeds_dict["surjective_mapping_seed"]
    A_matrices_seed = seeds_dict["A_matrices_seed"]
    surjective_mapping = construct_surjective_mapping(tilde_K = tilde_K, num_measures = num_measures, seed = surjective_mapping_seed)
    A_matrices_dict = generate_A_matrices(dim = dim, num_measures = num_measures, seed = A_matrices_seed)

    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                     num_measures = num_measures, 
                                                     auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                     source_sampler = source_sampler,
                                                     truncated_radius = truncated_radius,
                                                     manual = False,
                                                     bound_type= bound_type,
                                                     theta = instance_theta,
                                                     surjective_mapping = surjective_mapping,
                                                     A_matrices_dict = A_matrices_dict)
    
    entropic_sampler = set_up_entropic_sampler(entropic_sampler, save_dir = samplers_info_dir)
    print("Entropic sampler configured.")

    # Generate input samples
    csv_path = f"../../WB_data/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}/input_samples/csv_files"
    os.makedirs(csv_path, exist_ok=True)
    
    input_measure_samples = entropic_sampler.sample(num_samples_in_preparation)

    for measure_index in range(num_measures):
        measure_samples = np.asarray(input_measure_samples[measure_index])
        csv_filename = os.path.join(csv_path, f"input_measure_samples_{measure_index}.csv")
        pd.DataFrame(measure_samples).to_csv(csv_filename, index=False, header=False)
    print("Input samples saved to CSV files.")

    
    