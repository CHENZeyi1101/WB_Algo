import numpy as np
import json, os
from pathlib import Path

from Algorithms.data_manage import *
from Algorithms.Stochastic_FP.entropic_iterative_scheme import entropic_iterative_scheme
from Experiments.metrics_to_compare import evaluate_MC
from Experiments.CSV_read import *

if __name__ == "__main__":
    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_posterior_aggregation_dim9"]

    samples_dir = cfg_dict['samples_dir']

    # assert existence
    assert os.path.exists(samples_dir), f"Instance directory {samples_dir} does not exist."

    num_measures = params["num_measures"]
    eval_MC_size = params["MC_size"]
    eval_num_samples = params["eval_num_samples"]
    csv_skip_rows = params["csv_skip_rows"]
    csv_cols_range = params["cvs_cols_range"]

    num_iters = 10
    rand_state = np.random.RandomState(seed = 88888)
    init_method = {"type": "moment", "sample_size": 10000}
    truncate_radius = params["truncated_radius"]
    sample_size_scheme = [5000, 5000, 10000, 10000, 20000, 20000, 40000, 40000, 80000, 80000]
    reg_param_scheme = [1e-8] * num_iters
    # sinkhorn_impl = "ott"
    # warm_start = {"type": "first-order"}

    sinkhorn_impl = "geomloss"
    warm_start = None

    ##### Set up the samplers for Bike Sharing data #####
    split_posterior_sampler = csv_posterior_sampler_BikeSharing(csv_dir = samples_dir, 
                                                num_measures = num_measures, 
                                                multiplication_factor = 1, 
                                                type = "split",
                                                usecols = csv_cols_range,
                                                skiprows = csv_skip_rows)
    split_posterior_sampler.set_streamers()

    ##### Set up the samples for evaluation #####
    eval_dir = cfg_dict["samples_for_evaluation_dir"]

    bary_sample_path = f"{eval_dir}/bary_samples_collection.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {int(k): np.array(v) for k, v in bary_samples_collection_loaded.items()}

    input_sample_path = f"{eval_dir}/input_samples_collection.json"
    with open(input_sample_path, 'r') as json_file:
        input_samples_collection_loaded = json.load(json_file)
    input_samples_collection_loaded = {int(k): {int(i): np.array(u) for i, u in v.items()}
                                        for k, v in input_samples_collection_loaded.items()}

    
    ##### Set up the entropic iterative computer #####
    entropic_iterative_computer = entropic_iterative_scheme(
        dim = dim,
        num_iters = num_iters,
        input_sampler = split_posterior_sampler,
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
    

    outputs_dir = f"{cfg_dict['outputs_dir']}/stochastic_FP_outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    ##### Run the stochastic FP algorithm with entropic OT map estimation #####
    entropic_iterative_computer.converge(logger = {'sample_logger': None, 'map_logger': None}, data_dir = outputs_dir)