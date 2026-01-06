from Algorithms.Stochastic_FP.entropic_iterative_scheme import *
from Algorithms.data_manage import *
from Experiments.Synthetic_Generation.samplers import *
from Experiments.Synthetic_Generation.visualize_measures_dim2 import *
from Experiments.Synthetic_Generation.input_generate_entropic import *

if __name__ == "__main__":
    dim = 2
    num_components = 5
    num_samples = 10000
    num_measures = 5
    truncated_radius = 150
    source_sampler_seed = 1009
    instance_theta = 2000
    csv_sampler = True

    if csv_sampler:
        source_csv_file = f"../../WB_data/Synthetic_Generation/dim{dim}_data/source_samples/csv_files/source_measure_samples.csv"
        source_sampler = csv_source_sampler_SyntheticGeneration(source_csv_file, 
                                                    multiplication_factor=1,
                                                    usecols=None,
                                                    skiprows=0)
        source_sampler.set_streamer()

        input_csv_path = f"../../WB_data/Synthetic_Generation/dim{dim}_data/input_samples/csv_files_InstanceTheta2000"
        input_sampler = csv_input_sampler_SyntheticGeneration(input_csv_path, 
                                                    num_measures, 
                                                    multiplication_factor=1)
        input_sampler.set_streamers()

    else:
        load_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/samplers_info"
        source_sampler = MixtureOfGaussians(dim)
        source_sampler.random_components(num_components=5, uniform_weights = True, seed = source_sampler_seed)
        source_sampler.set_truncation(truncated_radius)
        with open(f"{load_dir}/source_sampler_info.pkl", "rb") as f:
            loaded_data_source_sampler = pickle.load(f)
        source_sampler.__dict__.update(loaded_data_source_sampler)

        auxiliary_csv_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/auxiliary_samples/csv_files"
        auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(csv_dir = auxiliary_csv_dir, auxiliary_seeds_list = [1010, 1018, 1014, 1016, 1003])

        entropic_sampler = characterize_entropic_sampler(dim = dim, num_measures = num_measures)                
        with open(f"{sampler_load_dir}/entropic_sampler_info.pkl", "rb") as f:
            loaded_data_entropic_sampler = pickle.load(f)
        manual_params = {
        "auxiliary_measure_sampler_set": auxiliary_measure_sampler_set,
        "source_sampler": source_sampler
        }
        config = {**loaded_data_entropic_sampler, **manual_params}
        # update entropic sampler
        entropic_sampler.__dict__.update(config)
        input_sampler = entropic_sampler


    # Set up the entropic iterative computer
    entropic_iterative_computer = entropic_iterative_scheme(dim = dim, 
                                                            num_measures = num_measures, 
                                                            bary_sampler = source_sampler, 
                                                            input_sampler = input_sampler, # alternative: entropic_sampler
                                                            truncate_radius = truncated_radius)
    # bary_samples = entropic_iterative_computer.bary_sampling(num_samples = num_samples)
    input_samples_collection = entropic_iterative_computer.input_sampling(num_samples)

    bary_sample_path = f"../../WB_Data/Synthetic_Generation/dim{dim}_data/bary_samples_collection/bary_samples_collection_dim{dim}_MCsize50_numsamples10000.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}

    # print(bary_samples_collection_loaded["0"])

    data_dir = f"../../WB_Data/Synthetic_Generation/dim{dim}_data/Outputs_InstanceTheta{instance_theta}/stochastic_FP_outputs"
    os.makedirs(data_dir, exist_ok=True)

    entropic_iterative_computer.converge(bary_samples_collection_loaded,
                                        # input_samples_collection,
                                        max_iter = 5,
                                        num_samples = num_samples,
                                        epsilon = 10,
                                        MC_size = 20,
                                        logger = {'sample_logger': None, 'map_logger': None},
                                        data_dir = data_dir,
                                        warm_start = False
                                        )
    