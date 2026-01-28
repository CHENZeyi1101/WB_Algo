import numpy as np
import ot
from Experiments.Synthetic_Generation.samplers import *
from Experiments.Synthetic_Generation.metrics_to_compare import *
from Experiments.Synthetic_Generation.input_generate_entropic import *
import json, os
from pathlib import Path
from Algorithms.Fast_Cuturi.free_support_WB import w2_barycenter_free_support_from_samples
from Experiments.CSV_read import *

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
    input_sampler_for_evaluation = csv_input_sampler_for_evaluation_SyntheticGeneration(eval_dir, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler_for_evaluation.set_streamers()

    bary_sample_path = f"{instance_dir}/samples_for_evaluation/bary_samples_collection_dim{dim}_MCsize{MC_size}_numsamples{num_samples}.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}
 
    outputs_dir = f"{instance_dir}/outputs/Fast_Cuturi_outputs_numsamples{num_samples}_support{support_size}"
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

    for i in range(MC_size):
        print(f"Computing barycenter sample {i+1}/{MC_size}...")

        bary_samples = bary_samples_collection_loaded[str(i)]
        input_samples_collection_for_evaluation = input_sampler_for_evaluation.sample(num_samples)

        # compute V-value
        V_value = 0
        for measure_index in range(num_measures):
            input_samples = np.array(input_samples_collection_for_evaluation[measure_index])
            V_value += W2_pot(input_samples, approx_bary)
        V_value /= num_measures
        V_values_list.append(V_value)
        print(f"V-value for barycenter sample {i}: {V_value}")

        # compute W2 to barycenter samples
        W2_sq = W2_pot(approx_bary, bary_samples)
        W2_to_bary_list.append(W2_sq)
        print(f"W2 squared to barycenter samples for barycenter sample {i}: {W2_sq}")

        # save V-values and W2_to_bary values
        V_values_path = os.path.join(V_values_dir, f"V_values.json")
        V_values_dict = {
            "mean": np.mean(np.array(V_values_list)),
            "std": np.std(np.array(V_values_list)),
            "values": V_values_list}
        with open(V_values_path, 'w') as json_file:
            json.dump(V_values_dict, json_file) 

        W2_to_bary_path = os.path.join(W2_to_bary_dir, f"W2_to_bary.json")
        W2_to_bary_dict = {
            "mean": np.mean(np.array(W2_to_bary_list)),
            "std": np.std(np.array(W2_to_bary_list)),
            "values": W2_to_bary_list}
        with open(W2_to_bary_path, 'w') as json_file:
            json.dump(W2_to_bary_dict, json_file)

    

   






    