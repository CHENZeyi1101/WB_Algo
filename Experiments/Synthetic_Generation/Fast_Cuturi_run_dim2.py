import os
os.environ["PYKEOPS_VERBOSE"] = "0"
import numpy as np
import ot
from tqdm import tqdm
from multiprocessing import Pool
import json
from pathlib import Path

from Experiments.Synthetic_Generation.samplers import *
from Experiments.Synthetic_Generation.metrics_to_compare import evaluate_zipped
from Experiments.Synthetic_Generation.input_generate_entropic import *
from Algorithms.Fast_Cuturi.free_support_WB import w2_barycenter_free_support_from_samples
from Experiments.CSV_read import *
from Algorithms.data_manage import *

if __name__ == "__main__":
    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]

    # take all items in params
    dim = params["dim"]
    num_measures = params["num_measures"]
    instance_identifier = params["instance_identifier"]
    MC_size = params["MC_size"]
    num_samples = params["num_samples"]
    eval_num_samples = params["eval_num_samples"]

    # number of atoms in the discrete supports
    support_size = 10000

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    input_csv_path = f"{instance_dir}/input_samples/csv_files"
    input_sampler = csv_input_sampler_SyntheticGeneration(input_csv_path, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler.set_streamers()

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

    
 
    outputs_dir = f"{instance_dir}/outputs/Fast_Cuturi_outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    V_values_dir = os.path.join(outputs_dir, "V_values")
    W2_to_bary_dir = os.path.join(outputs_dir, "W2_to_bary")
    os.makedirs(V_values_dir, exist_ok=True)
    os.makedirs(W2_to_bary_dir, exist_ok=True)

    V_values_list = []
    W2_to_bary_list = []

    input_samples_collection = input_sampler.sample(num_samples)
    samples_list = [np.array(input_samples_collection[key]) for key in sorted(input_samples_collection.keys())]
    approx_bary = w2_barycenter_free_support_from_samples(
        samples_list,
        k=support_size,
        init="random",
        numItermax=200,
        verbose=True,
        seed=42,
    )

    # Evaluation
    approx_bary_it = [approx_bary for _ in range(MC_size)]
    input_measure_samples_collection_it = [{k : input_samples_collection_loaded[i][k][:eval_num_samples] for k in range(num_measures)} for i in range(MC_size)]
    true_bary_samples_it = [bary_samples_collection_loaded[i][:eval_num_samples] for i in range(MC_size)]
    
    with Pool(processes = 5) as pool, tqdm(total = MC_size) as pbar:
        for V_value, W2_to_bary in pool.imap(evaluate_zipped, 
                                zip(approx_bary_it, 
                                    input_measure_samples_collection_it, 
                                    true_bary_samples_it)):
            V_values_list.append(V_value)
            W2_to_bary_list.append(W2_to_bary)
            pbar.update(1)
            pbar.refresh()

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