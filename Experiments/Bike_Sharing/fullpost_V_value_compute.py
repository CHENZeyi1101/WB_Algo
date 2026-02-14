from pathlib import Path
import json, os
import numpy as np

from Algorithms.data_manage import *
from Experiments.metrics_to_compare import evaluate_MC

if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)
    
    eval_dir = cfg_dict["samples_for_evaluation_dir"]
    params = cfg_dict["params_posterior_aggregation_dim9"]
    num_measures = params["num_measures"]
    eval_num_samples = params["eval_num_samples"]
    MC_size = params["MC_size"]
    
    outputs_dir = cfg_dict["outputs_dir"]
    save_dir = f"{outputs_dir}/fullpost_V_value"
    os.makedirs(save_dir, exist_ok=True)

    fullpost_sample_path = f"{eval_dir}/bary_samples_collection.json"
    with open(fullpost_sample_path, 'r') as json_file:
        fullpost_samples_collection_loaded = json.load(json_file)
    fullpost_samples_collection_loaded = {int(k): np.array(v) for k, v in fullpost_samples_collection_loaded.items()}

    input_sample_path = f"{eval_dir}/input_samples_collection.json"
    with open(input_sample_path, 'r') as json_file:
        input_samples_collection_loaded = json.load(json_file)
    input_samples_collection_loaded = {int(k): {int(i): np.array(u) for i, u in v.items()}
                                        for k, v in input_samples_collection_loaded.items()}

    fullpost_it = [fullpost_samples_collection_loaded[i][:eval_num_samples] for i in range(MC_size)]
    input_measure_samples_collection_it = [{k : input_samples_collection_loaded[i][k][:eval_num_samples] for k in range(num_measures)} for i in range(MC_size)]
    true_bary_samples_it = [None for _ in range(MC_size)]
    
    V_values_list, _ = evaluate_MC(fullpost_it, 
                                   input_measure_samples_collection_it, 
                                   true_bary_samples_it, 
                                   MC_size = MC_size, 
                                   num_parallel_process = 5, 
                                   pbar_text = "Computation of the V-value of the full data posterior")

    # save V-values
    V_values_dict = {
        "mean": np.mean(V_values_list),
        "std": np.std(V_values_list),
        "values": V_values_list}
    save_json(V_values_dict, save_dir, "fullpost_V_values.json")
