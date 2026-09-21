from pathlib import Path
import json, os
import numpy as np

from Algorithms.data_manage import *
from Experiments.metrics_to_compare import evaluate_MC
from Experiments.CSV_read import StreamingCSVSamples

if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)
    
    eval_dir = cfg_dict["samples_for_evaluation_dir"]
    params = cfg_dict["params_posterior_aggregation_dim8"]
    num_measures = params["num_measures"]
    eval_num_samples = params["eval_num_samples"]
    MC_size = params["MC_size"]

    samples_dir = cfg_dict['samples_dir']
    csv_skip_rows = params["csv_skip_rows"]
    csv_cols_range = range(params["csv_cols_range"][0], params["csv_cols_range"][1])

    
    outputs_dir = cfg_dict["outputs_dir"]
    save_dir = f"{outputs_dir}/fullpost_via_OT"
    os.makedirs(save_dir, exist_ok=True)

    n_skip_rows = eval_num_samples * (MC_size + 1) # skip the samples used for evaluation to avoid overlap between the samples used in evaluation and the samples used in the full data posterior V-value computation

    full_data_posterior_reader = StreamingCSVSamples(csv_filename = f"{samples_dir}/posterior_full.csv",
                                                     skiprows = csv_skip_rows + n_skip_rows, 
                                                     usecols = csv_cols_range,
                                                     has_header = False)
    
    fullpost_samples_it = []
    for i in range(MC_size):
        source_samples = full_data_posterior_reader.take(eval_num_samples)
        fullpost_samples_it.append(source_samples)

    bary_sample_path = f"{eval_dir}/bary_samples_collection.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {int(k): np.array(v) for k, v in bary_samples_collection_loaded.items()}

    input_sample_path = f"{eval_dir}/input_samples_collection.json"
    with open(input_sample_path, 'r') as json_file:
        input_samples_collection_loaded = json.load(json_file)
    input_samples_collection_loaded = {int(k): {int(i): np.array(u) for i, u in v.items()}
                                        for k, v in input_samples_collection_loaded.items()}

    input_measure_samples_collection_it = [{k : input_samples_collection_loaded[i][k][:eval_num_samples] for k in range(num_measures)} for i in range(MC_size)]
    true_bary_samples_it = [bary_samples_collection_loaded[i][:eval_num_samples] for i in range(MC_size)]
    
    V_values_list, W2_to_bary_list = evaluate_MC(fullpost_samples_it, 
                                   input_measure_samples_collection_it, 
                                   true_bary_samples_it, 
                                   MC_size = MC_size, 
                                   num_parallel_process = None, 
                                   pbar_text = "Computation of the V-value of the full data posterior")

    # save V-values
    V_values_dict = {
        "mean": np.mean(V_values_list),
        "std": np.std(V_values_list),
        "values": V_values_list}
    save_json(V_values_dict, save_dir, "fullpost_V_values.json")

    W2_to_bary_dict = {
        "mean": np.mean(W2_to_bary_list),
        "std": np.std(W2_to_bary_list),
        "values": W2_to_bary_list}
    save_json(W2_to_bary_dict, save_dir, "fullpost_W2_to_bary_OT.json")
