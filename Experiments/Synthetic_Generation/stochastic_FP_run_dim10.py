from ...Algorithms.Stochastic_FP.entropic_iterative_scheme import *
from ...Algorithms.data_manage import *
from .samplers_dim2 import *
from .visualize_measures_dim2 import *
from .input_generate_entropic import *

if __name__ == "__main__":
    dim = 10
    num_components = 5
    num_samples = 10000
    num_measures = 10
    truncated_radius = 5000
    seed = 1009

    load_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/samplers_info"

    csv_path = f"../WB_data/Synthetic_Generation/dim{dim}_data/input_samples/csv_files"
    csv_sampler = csv_input_sampler(dim = dim, num_measures = num_measures, csv_path = csv_path)

    # Load the samplers
    source_sampler = MixtureOfGaussians(dim)
    source_sampler = load_sampler(load_dir, source_sampler, sampler_type="source")

    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(dim, num_components)
    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                     num_measures = num_measures, 
                                                     auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                     source_sampler = source_sampler,
                                                     truncated_radius = truncated_radius,
                                                     manual = False)
    entropic_sampler = load_sampler(load_dir, entropic_sampler, sampler_type="entropic")
    print("done")
    
    # Set up the entropic iterative computer
    entropic_iterative_computer = entropic_iterative_scheme(dim = dim, 
                                                            num_measures = num_measures, 
                                                            bary_sampler = source_sampler, 
                                                            input_sampler = csv_sampler, # alternative: entropic_sampler
                                                            truncate_radius = truncated_radius)
    # bary_samples = entropic_iterative_computer.bary_sampling(num_samples = num_samples)
    input_samples_collection = entropic_iterative_computer.input_sampling(num_samples = num_samples)

    bary_sample_path = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/bary_samples_collection/bary_samples_collection_dim{dim}_MCsize50_numsamples10000.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}

    # print(bary_samples_collection_loaded["0"])

    data_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/stochastic_FP_outputs"
    os.makedirs(data_dir, exist_ok=True)

    entropic_iterative_computer.converge(bary_samples_collection_loaded,
                                        # input_samples_collection,
                                        max_iter = 5,
                                        num_samples = num_samples,
                                        epsilon = 10,
                                        MC_size = 20,
                                        logger = {'sample_logger': None, 'map_logger': None},
                                        data_dir = data_dir,
                                        warm_start = True
                                        )