from Algorithms.Stochastic_FP.entropic_iterative_scheme import *
from Algorithms.data_manage import *
from Experiments.Synthetic_Generation.samplers import *
from Experiments.Synthetic_Generation.visualize_measures_dim2 import *
from Experiments.Synthetic_Generation.input_generate_entropic import *
from Experiments.Synthetic_Generation.MOG import *
import json
from pathlib import Path

if __name__ == "__main__":
    dim = 2
    num_components = 5
    num_samples = 1000
    num_measures = 5
    truncated_radius = 150
    instance_theta = 2000
    MC_size = 1

    instance_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}_toy"
    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    SEEDS_PATH = Path(__file__).parent / "seeds.json"
    with open(SEEDS_PATH, "r") as f:
        seeds_dict = json.load(f)

    source_component_seed = seeds_dict["source_components_seed"]
    master_source_rng = np.random.SeedSequence(seeds_dict["master_source_sampling_seed"])

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
    
    input_samples_collection = entropic_iterative_computer.input_sampling(num_samples)

    bary_sample_path = f"{instance_dir}/samples_for_evaluation/bary_samples_collection_dim{dim}_MCsize50_numsamples1000.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}
 
    # print(bary_samples_collection_loaded["0"])

    data_dir = f"{instance_dir}/outputs/stochastic_FP_outputs"
    os.makedirs(data_dir, exist_ok=True)

    eval_dir = f"{instance_dir}/samples_for_evaluation"
    input_sampler_for_evaluation = csv_input_sampler_for_evaluation_SyntheticGeneration(eval_dir, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler_for_evaluation.set_streamers()

    entropic_iterative_computer.converge(bary_samples_collection_loaded,
                                        input_sampler_for_evaluation,
                                        max_iter = 5,
                                        num_samples = num_samples,
                                        epsilon = 10,
                                        MC_size = MC_size,
                                        logger = {'sample_logger': None, 'map_logger': None},
                                        data_dir = data_dir,
                                        warm_start = False
                                        )
    