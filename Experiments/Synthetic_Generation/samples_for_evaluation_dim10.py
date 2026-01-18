from Experiments.Synthetic_Generation.MOG import *
from Experiments.Synthetic_Generation.samplers import *
from Experiments.CSV_read import *
from Experiments.CSV_shuffle import *
from tqdm import tqdm
import os
import json
from pathlib import Path

if __name__ == "__main__":
    dim = 10
    num_measures = 10
    bary_MC_size = 50
    num_samples = 10000
    truncated_radius = 5000
    instance_theta = 2000
    num_components = 5

    if dim == 2:
        bound_type = "eigen_bound"
    else:
        bound_type = "norm_bound"

    SEEDS_PATH = Path(__file__).parent / "seeds.json"
    with open(SEEDS_PATH, "r") as f:
        seeds_dict = json.load(f)

    source_component_seed = seeds_dict["source_components_seed"]

    source_sampler = characterize_source_sampler(dim = dim, 
                                                num_components = num_components, 
                                                master_sampling_rng = 42,
                                                component_seed = source_component_seed,
                                                truncated_radius = truncated_radius,
                                                save_dir = None)
    
    load_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}/samplers_info"
    input_sampler = characterize_entropic_sampler(dim = dim, num_measures = num_measures)
    input_sampler = load_sampler(load_dir, input_sampler, sampler_type = "entropic")

    data_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}_toy/samples_for_evaluation"
    os.makedirs(data_dir, exist_ok=True)

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

    csv_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}_/input_samples/csv_files"
    for i in tqdm(range(num_measures), desc="Shuffling CSV files"):
        old_csv_path = f"{csv_dir}/input_measure_samples_{i}.csv"
        csv_evaluate_dir = f"{csv_dir}/for_evaluation"
        os.makedirs(csv_evaluate_dir, exist_ok=True)
        new_csv_path = f"{csv_evaluate_dir}/input_measure_samples_{i}_for_evaluation.csv"
        csv_shuffle(old_csv_path, new_csv_path, seed = 1000)
    

    
    # read back and change list to numpy array
    # with open(json_path, 'r') as json_file:
    #     bary_samples_collection_loaded = json.load(json_file)
    # bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}
    
