from ...Algorithms.Stochastic_FP.entropic_iterative_scheme import *
from ...Algorithms.data_manage import *
from .posterior_sampler import *
from .visualize_posteriors import *

if __name__ == "__main__":
    dim = 8
    num_samples = 2000
    num_measures = 5
    truncated_radius = 500
    multiplication_factor = 10

    DATA_DIR = os.path.dirname(__file__)
    MODEL_DIR = os.path.join(DATA_DIR, "models_meta")

    total_posterior_path = os.path.join(MODEL_DIR, "model_total.meta.pkl")
    
    total_posterior_sampler = posterior_sampler(model_path=total_posterior_path, num_measures=1, multiplication_factor=multiplication_factor)
    subset_posterior_sampler = posterior_sampler(model_path=MODEL_DIR, num_measures=num_measures, multiplication_factor=multiplication_factor)
    
    # Set up the entropic iterative computer
    entropic_iterative_computer = entropic_iterative_scheme(dim = dim, 
                                                            num_measures = num_measures, 
                                                            bary_sampler = total_posterior_sampler, 
                                                            input_sampler = subset_posterior_sampler, 
                                                            truncate_radius = truncated_radius)
    bary_samples = entropic_iterative_computer.bary_sampling(num_samples = num_samples)
    input_samples_collection = entropic_iterative_computer.input_sampling(num_samples = num_samples)

    data_dir = "./WB_Algo/Experiments/Bike_Sharing/data_outputs/stochastic_FP_outputs"
    os.makedirs(data_dir, exist_ok=True)

    entropic_iterative_computer.converge(bary_samples,
                                        input_samples_collection,
                                        max_iter = 5,
                                        num_samples = num_samples,
                                        epsilon = 10,
                                        MC_size = 30,
                                        logger = {'sample_logger': None, 'map_logger': None},
                                        data_dir = data_dir,
                                        warm_start = False
                                        )