import os
import json
from pathlib import Path

from Experiments.CSV_read import StreamingCSVSamples

if __name__ == "__main__":
    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)
    samples_dir = cfg_dict["samples_dir"]

    params = cfg_dict["params_posterior_aggregation_dim9"]
    num_measures = params["num_measures"]
    eval_MC_size = params["MC_size"]
    eval_num_samples = params["eval_num_samples"]
    csv_skip_rows = params["csv_skip_rows"]
    csv_cols_range = range(params["csv_cols_range"][0], params["csv_cols_range"][1])

    eval_dir = cfg_dict["samples_for_evaluation_dir"]
    os.makedirs(eval_dir, exist_ok=True)

    full_data_posterior_reader = StreamingCSVSamples(csv_filename = f"{samples_dir}/posterior_full.csv",
                                                     skiprows = csv_skip_rows, 
                                                     usecols = csv_cols_range,
                                                     has_header = False)
    
    split_data_posterior_eval_reader_list = [None] * num_measures

    for k in range(num_measures):
        split_data_posterior_eval_reader_list[k] = StreamingCSVSamples(csv_filename = f"{samples_dir}/posterior_for_evaluation_split_{k}.csv",
                                                                       skiprows = csv_skip_rows, 
                                                                       usecols = csv_cols_range,
                                                                       has_header = False)

    # sample from the true barycenter and the input measures
    bary_samples_collection = {}
    input_samples_collection = {}
    for i in range(eval_MC_size):
        print(f"Generating bary sample of MC step {i+1}/{eval_MC_size} ...")
        bary_samples_collection[i] = full_data_posterior_reader.take(eval_num_samples).tolist()
        input_samples_collection[i] = {k : split_data_posterior_eval_reader_list[k].take(eval_num_samples).tolist() for k in range(num_measures)}
    
    # save as json
    bary_json_path = os.path.join(eval_dir, f"bary_samples_collection.json")
    with open(bary_json_path, 'w') as json_file:
        json.dump(bary_samples_collection, json_file)
    
    input_json_path = os.path.join(eval_dir, f"input_samples_collection.json")
    with open(input_json_path, 'w') as json_file:
        json.dump(input_samples_collection, json_file)