from pathlib import Path
import json, os
import numpy as np

from Algorithms.data_manage import *
from Experiments.metrics_to_compare import evaluate_MC
from Experiments.Synthetic_Generation.MOG import MixtureOfGaussians

if __name__ == "__main__":

    
    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim10"]

    # take all items in params
    num_samples = params["num_samples"]
    eval_num_samples = params["eval_num_samples"]
    dim = params["dim"]
    num_measures = params["num_measures"]
    truncated_radius = params["truncated_radius"]
    instance_identifier = params["instance_identifier"]
    num_components = params["num_components"]
    MC_size = params["MC_size"]

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."
    outputs_dir = f"{instance_dir}/outputs"
    assert os.path.exists(outputs_dir), f"outputs directory {outputs_dir} does not exist."
    save_dir = f"{outputs_dir}/true_via_OT"
    os.makedirs(save_dir, exist_ok=True)

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
    
    source_info = {
        "num_components": params["num_components"],
        "component_seed": params["seeds"]["source_components_seed"], 
        "master_sampling_rng": np.random.SeedSequence(params["seeds"]["master_source_sampling_seed"] - 4000) # use a different seed for the source sampler in the evaluation script to avoid overlap with the samples used in evaluation
    }
    
    source_sampler = MixtureOfGaussians(dim = dim, 
                                        master_sampling_rng = source_info["master_sampling_rng"], 
                                        component_seed = source_info["component_seed"])
    source_sampler.random_components(num_components = source_info["num_components"], 
                                        uniform_weights = True)
    source_sampler.set_truncation(truncated_radius)

    eval_bary_samples_it = []
    for i in range(MC_size):
        source_samples = source_sampler.sample(eval_num_samples)
        eval_bary_samples_it.append(source_samples)

    # true_bary_samples_it = []
    # for i in range(MC_size):
    #     bary_samples = source_sampler.sample(eval_num_samples)
    #     true_bary_samples_it.append(bary_samples)

    # # check
    # print(true_bary_samples_it[0] == eval_bary_samples_it[0]) # should be False since the samples are different, even though they come from the same distribution

    input_measure_samples_collection_it = [{k : input_samples_collection_loaded[i][k][:eval_num_samples] for k in range(num_measures)} for i in range(MC_size)]
    true_bary_samples_it = [bary_samples_collection_loaded[i][:eval_num_samples] for i in range(MC_size)]
    
    V_values_list, W2_to_bary_list = evaluate_MC(eval_bary_samples_it, 
                                   input_measure_samples_collection_it, 
                                   true_bary_samples_it, 
                                   MC_size = MC_size, 
                                   num_parallel_process = None, 
                                   pbar_text = "Computation of the V-value and W2_to_bary via OT")

    # save V-values and W2 values to barycenter
    V_values_dict = {
        "mean": np.mean(V_values_list),
        "std": np.std(V_values_list),
        "values": V_values_list}
    save_json(V_values_dict, save_dir, "true_V_values_OT_check.json")


    W2_to_bary_dict = {
        "mean": np.mean(W2_to_bary_list),
        "std": np.std(W2_to_bary_list),
        "values": W2_to_bary_list}
    save_json(W2_to_bary_dict, save_dir, "true_W2_to_bary_OT_check.json")
