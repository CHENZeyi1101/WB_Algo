import os
os.environ["PYKEOPS_VERBOSE"] = "0"

from Algorithms.Stochastic_FP.modified_entropic_iterative_scheme import modified_entropic_iterative_scheme
from Algorithms.Stochastic_FP.modified_entropic_iterative_scheme2 import modified_entropic_iterative_scheme2
from Algorithms.Stochastic_FP.entropic_iterative_scheme import entropic_iterative_scheme
from Algorithms.data_manage import *
from Experiments.Synthetic_Generation.samplers import *
from Experiments.Synthetic_Generation.visualize_measures_dim2 import *
from Experiments.Synthetic_Generation.input_generate_entropic import *
from Experiments.Synthetic_Generation.MOG import *
import json
from pathlib import Path

if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]

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

    num_iters = 8
    rand_state = np.random.RandomState(seed = 7777)
    init_method = {"type": "moment", "sample_size": 10000}
    truncate_radius = params["truncated_radius"]
    sample_size_scheme = [20000, 20000, 40000, 80000, 160000, 320000, 320000, 320000]
    reg_param_scheme = [2, 2, 2, 2, 2, 2, 2, 2]
    # sinkhorn_impl = "ott"
    # warm_start = {"type": "first-order"}

    sinkhorn_impl = "geomloss"
    warm_start = None

    bary_sample_path = f"{instance_dir}/samples_for_evaluation/bary_samples_collection_dim{dim}_MCsize{eval_MC_size}_numsamples{eval_num_samples}.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}

    outputs_dir = f"{instance_dir}/outputs/stochastic_FP_outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    eval_dir = f"{instance_dir}/samples_for_evaluation"
    input_sampler_for_evaluation = csv_input_sampler_for_evaluation_SyntheticGeneration(eval_dir, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler_for_evaluation.set_streamers()

    # Set up the entropic iterative computer
    entropic_iterative_computer = modified_entropic_iterative_scheme2(
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
        bary_sample_collection = bary_samples_collection_loaded, 
        input_sampler_for_evaluation = input_sampler_for_evaluation,
        eval_num_samples = eval_num_samples,
        eval_MC_size = eval_MC_size,
        num_parallel = 10
    )

    entropic_iterative_computer.converge(logger = {'sample_logger': None, 'map_logger': None}, data_dir = outputs_dir)
    