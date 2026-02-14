import numpy as np
import os, json
from pathlib import Path
from tqdm import tqdm

from Algorithms.CWB_Li.cwb.tests.comparison.common import *
from Experiments.metrics_to_compare import evaluate_zipped
from multiprocessing import Pool
from Algorithms.data_manage import save_json

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

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    outputs_dir = f"{instance_dir}/outputs/CWB_Li_outputs"
    assert os.path.exists(outputs_dir), f"CWB_Li outputs directory {outputs_dir} does not exist."

    # g_base_dir = f"{instance_dir}/outputs/CWB_Li_outputs"

    result_dir = get_result_nd_dir(dim, outputs_dir)
    result_filename = get_result_filename("cwb", 0)
    result_npz_path = os.path.join(result_dir, result_filename)
    cwb_data = np.load(result_npz_path)
    print(cwb_data.shape) # (MC_size * num_samples, dim)

    approx_bary_it = list(cwb_data.reshape(MC_size, num_samples, dim))


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

    V_values_dir = os.path.join(outputs_dir, "V_values")
    W2_to_bary_dir = os.path.join(outputs_dir, "W2_to_bary")
    os.makedirs(V_values_dir, exist_ok=True)
    os.makedirs(W2_to_bary_dir, exist_ok=True)

    V_values_list = []
    W2_to_bary_list = []

    input_measure_samples_collection_it = [{k : input_samples_collection_loaded[i][k][:eval_num_samples] for k in range(num_measures)} for i in range(MC_size)]
    true_bary_samples_it = [bary_samples_collection_loaded[i][:eval_num_samples] for i in range(MC_size)]

    # for args in tqdm(
    #         zip(approx_bary_it,
    #             input_measure_samples_collection_it,
    #             true_bary_samples_it),
    #         total=MC_size):

    #     V_value, W2_to_bary = evaluate_zipped(args)

    #     V_values_list.append(V_value)
    #     W2_to_bary_list.append(W2_to_bary)

    with Pool(processes = 2) as pool, tqdm(total = MC_size) as pbar:
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





# arr = np.load("file.npy")
