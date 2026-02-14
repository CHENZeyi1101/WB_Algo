import os
import json
from pathlib import Path
import numpy as np

from Algorithms.Stochastic_FP.entropic_iterative_scheme import entropic_iterative_scheme
from Experiments.CSV_read import csv_input_sampler_SyntheticGeneration

if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim10"]

    # take all items in params
    dim = params["dim"]
    num_measures = params["num_measures"]
    instance_identifier = params["instance_identifier"]

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"

    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    input_csv_path = f"{instance_dir}/input_samples/csv_files"
    input_sampler = csv_input_sampler_SyntheticGeneration(input_csv_path, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler.set_streamers()

    eval_MC_size = params["MC_size"]
    eval_num_samples = params["eval_num_samples"]

    num_iters = 10
    rand_state = np.random.RandomState(seed = 97777)
    init_method = {"type": "moment", "sample_size": 10000}
    truncate_radius = params["truncated_radius"]
    sample_size_scheme = np.rint(np.exp(np.linspace(np.log(10000), np.log(80000), num_iters)) + 1).astype(int).tolist()
    reg_param_scheme = [10] * num_iters
    # sinkhorn_impl = "ott"
    # warm_start = {"type": "first-order"}

    sinkhorn_impl = "geomloss"
    warm_start = None

    outputs_dir = f"{instance_dir}/outputs/stochastic_FP_outputs"
    os.makedirs(outputs_dir, exist_ok=True)

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

    # Set up the entropic iterative computer
    entropic_iterative_computer = entropic_iterative_scheme(
        dim = dim,
        num_iters = num_iters,
        input_sampler = input_sampler,
        rand_state = rand_state,
        init_method = init_method,
        truncate_radius = truncate_radius,
        sinkhorn_impl = sinkhorn_impl,
        sample_size_scheme = sample_size_scheme,
        reg_param_scheme = reg_param_scheme,
        warm_start = warm_start,
        bary_samples_collection = bary_samples_collection_loaded, 
        input_samples_for_evaluation = input_samples_collection_loaded,
        eval_num_samples = eval_num_samples,
        eval_MC_size = eval_MC_size,
        num_parallel = 5
    )

    entropic_iterative_computer.converge(logger = {'sample_logger': None, 'map_logger': None}, data_dir = outputs_dir)
    