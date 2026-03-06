from pathlib import Path
import json, os
import numpy as np

from Algorithms.data_manage import save_json
from Experiments.metrics_to_compare import evaluate_MC

if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim10"]

    # take all items in params
    dim = params["dim"]
    num_measures = params["num_measures"]
    instance_identifier = params["instance_identifier"]
    MC_size = params["MC_size"]
    eval_num_samples = params["eval_num_samples"]
    eval_num_samples = 10000

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

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
 
    outputs_dir = f"{instance_dir}/outputs/true_V_value_OT"
    os.makedirs(outputs_dir, exist_ok=True)

    V_values_dir = os.path.join(outputs_dir, "V_values")
    W2_to_bary_dir = os.path.join(outputs_dir, "W2_to_bary")
    os.makedirs(V_values_dir, exist_ok=True)
    os.makedirs(W2_to_bary_dir, exist_ok=True)

    approx_bary_it = [bary_samples_collection_loaded[i][:eval_num_samples] for i in range(MC_size)]
    input_measure_samples_collection_it = [{k : input_samples_collection_loaded[i][k][:eval_num_samples] for k in range(num_measures)} for i in range(MC_size)]
    true_bary_samples_it = [None for _ in range(MC_size)]
    
    V_values_list, _ = evaluate_MC(approx_bary_it, 
                                                 input_measure_samples_collection_it, 
                                                 true_bary_samples_it, 
                                                 MC_size = MC_size, 
                                                 num_parallel_process = 5, 
                                                 pbar_text = "Computation of true V-value")

    # save V-values and W2_to_bary values
    V_values_dict = {
        "mean": np.mean(V_values_list),
        "std": np.std(V_values_list),
        "values": V_values_list}
    save_json(V_values_dict, V_values_dir, "V_values.json")