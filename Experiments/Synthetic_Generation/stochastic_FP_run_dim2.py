from Algorithms.Stochastic_FP.entropic_iterative_scheme import *
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
    num_samples = params["num_samples"]
    dim = params["dim"]
    num_measures = params["num_measures"]
    truncated_radius = params["truncated_radius"]
    instance_identifier = params["instance_identifier"]
    num_components = params["num_components"]
    MC_size = params["MC_size"]

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_identifier}"
    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    source_component_seed = params["seeds"]["source_components_seed"]
    master_source_rng = np.random.SeedSequence(params["seeds"]["master_source_sampling_seed"])
    source_sampler = characterize_source_sampler(dim = dim, 
                                                num_components = num_components, 
                                                master_sampling_rng = master_source_rng,
                                                component_seed = source_component_seed,
                                                truncated_radius = truncated_radius,
                                                save_dir = None)

    input_csv_path = f"{instance_dir}/input_samples/csv_files"
    input_sampler = csv_input_sampler_SyntheticGeneration(input_csv_path, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler.set_streamers()

    # Set up the entropic iterative computer
    entropic_iterative_computer = entropic_iterative_scheme(dim = dim, 
                                                            num_measures = num_measures, 
                                                            bary_sampler = source_sampler, 
                                                            input_sampler = input_sampler, 
                                                            truncate_radius = truncated_radius)


    bary_sample_path = f"{instance_dir}/samples_for_evaluation/bary_samples_collection_dim{dim}_MCsize{MC_size}_numsamples{num_samples}.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}
 
    # print(bary_samples_collection_loaded["0"])

    epsilon = 1

    outputs_dir = f"{instance_dir}/outputs/stochastic_FP_outputs_epsilon{epsilon}"
    os.makedirs(outputs_dir, exist_ok=True)

    eval_dir = f"{instance_dir}/samples_for_evaluation"
    input_sampler_for_evaluation = csv_input_sampler_for_evaluation_SyntheticGeneration(eval_dir, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler_for_evaluation.set_streamers()

    entropic_iterative_computer.converge(bary_samples_collection_loaded,
                                        input_sampler_for_evaluation,
                                        max_iter = 5,
                                        num_samples = num_samples,
                                        epsilon = epsilon,
                                        MC_size = MC_size,
                                        logger = {'sample_logger': None, 'map_logger': None},
                                        data_dir = outputs_dir,
                                        warm_start = False
                                        )
    