import numpy as np
import json, os
from pathlib import Path

from Algorithms.Fast_Cuturi.free_support_WB import w2_barycenter_free_support_from_samples
from Algorithms.data_manage import save_json
from Experiments.metrics_to_compare import evaluate_MC
from Experiments.CSV_read import csv_posterior_sampler_BikeSharing

if __name__ == "__main__":
    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_posterior_aggregation_dim9"]

    samples_dir = cfg_dict['samples_dir']

    # assert existence
    assert os.path.exists(samples_dir), f"Instance directory {samples_dir} does not exist."

    num_measures = params["num_measures"]
    MC_size = params["MC_size"]
    eval_num_samples = params["eval_num_samples"]
    csv_skip_rows = params["csv_skip_rows"]
    csv_cols_range = range(params["csv_cols_range"][0], params["csv_cols_range"][1])

    # number of atoms to approximate the input measures
    num_samples = 10000
    
    # number of atoms in the discrete supports
    support_size = 10000


    split_posterior_sampler = csv_posterior_sampler_BikeSharing(csv_dir = samples_dir, 
                                                num_measures = num_measures, 
                                                multiplication_factor = 1, 
                                                type = "split",
                                                usecols = csv_cols_range,
                                                skiprows = csv_skip_rows)
    split_posterior_sampler.set_streamers()

    eval_dir = cfg_dict['samples_for_evaluation_dir']

    bary_sample_path = f"{eval_dir}/bary_samples_collection.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {int(k): np.array(v) for k, v in bary_samples_collection_loaded.items()}

    input_sample_path = f"{eval_dir}/input_samples_collection.json"
    with open(input_sample_path, 'r') as json_file:
        input_samples_collection_loaded = json.load(json_file)
    input_samples_collection_loaded = {int(k): {int(i): np.array(u) for i, u in v.items()}
                                        for k, v in input_samples_collection_loaded.items()}

    
 
    outputs_dir = f"{cfg_dict['outputs_dir']}/Fast_Cuturi_outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    V_values_dir = os.path.join(outputs_dir, "V_values")
    W2_to_bary_dir = os.path.join(outputs_dir, "W2_to_bary")
    os.makedirs(V_values_dir, exist_ok=True)
    os.makedirs(W2_to_bary_dir, exist_ok=True)

    input_samples_collection = split_posterior_sampler.sample(num_samples)
    samples_list = [np.array(input_samples_collection[key]) for key in sorted(input_samples_collection.keys())]
    approx_bary = w2_barycenter_free_support_from_samples(
        samples_list,
        k=support_size,
        init="random",
        numItermax=200,
        verbose=True,
        seed=8842,
    )

    # Evaluation
    approx_bary_it = [approx_bary for _ in range(MC_size)]
    input_measure_samples_collection_it = [{k : input_samples_collection_loaded[i][k][:eval_num_samples] for k in range(num_measures)} for i in range(MC_size)]
    true_bary_samples_it = [bary_samples_collection_loaded[i][:eval_num_samples] for i in range(MC_size)]
    
    V_values_list, W2_to_bary_list = evaluate_MC(approx_bary_it, 
                                                 input_measure_samples_collection_it, 
                                                 true_bary_samples_it, 
                                                 MC_size = MC_size, 
                                                 num_parallel_process = 5, 
                                                 pbar_text = "Evaluation of Fast_Cuturi")

    # save V-values and W2_to_bary values
    V_values_dict = {
        "mean": np.mean(V_values_list),
        "std": np.std(V_values_list),
        "values": V_values_list}
    save_json(V_values_dict, V_values_dir, "V_values.json")

    W2_to_bary_dict = {
        "mean": np.mean(W2_to_bary_list),
        "std": np.std(W2_to_bary_list),
        "values": W2_to_bary_list}
    save_json(W2_to_bary_dict, W2_to_bary_dir, "W2_to_bary.json")