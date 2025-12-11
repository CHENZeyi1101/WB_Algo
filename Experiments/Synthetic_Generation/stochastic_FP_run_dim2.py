from ...Algorithms.Stochastic_FP.entropic_iterative_scheme import *
from ...Algorithms.data_manage import *
from .samplers_dim2 import *
from .visualize_measures_dim2 import *

if __name__ == "__main__":
    dim = 2
    num_components = 5
    num_samples = 5000
    num_measures = 5
    truncated_radius = 150
    seed = 1009

    load_dir = "./WB_Algo/Experiments/Synthetic_Generation/dim2_data/samplers_info"

    # Load the samplers
    source_sampler = MixtureOfGaussians(dim)
    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(dim, num_components)
    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                     num_measures = num_measures, 
                                                     auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                     source_sampler = source_sampler,
                                                     truncated_radius = truncated_radius,
                                                     manual = True)
    
    source_sampler = load_sampler(load_dir, source_sampler, sampler_type="source")
    entropic_sampler = load_sampler(load_dir, entropic_sampler, sampler_type="entropic")
    print("done")
    
    # Set up the entropic iterative computer
    entropic_iterative_computer = entropic_iterative_scheme(dim = dim, 
                                                            num_measures = num_measures, 
                                                            bary_sampler = source_sampler, 
                                                            input_sampler = entropic_sampler, 
                                                            truncate_radius = truncated_radius)
    bary_samples = entropic_iterative_computer.bary_sampling(num_samples = num_samples)
    input_samples_collection = entropic_iterative_computer.input_sampling(num_samples = num_samples)

    data_dir = "./WB_Algo/Experiments/Synthetic_Generation/dim2_data/stochastic_FP_outputs"
    os.makedirs(data_dir, exist_ok=True)

    entropic_iterative_computer.converge(bary_samples,
                                        input_samples_collection,
                                        max_iter = 5,
                                        num_samples = num_samples,
                                        epsilon = 10,
                                        MC_size = 30,
                                        logger = {'sample_logger': None, 'map_logger': None},
                                        data_dir = data_dir,
                                        warm_start = True
                                        )