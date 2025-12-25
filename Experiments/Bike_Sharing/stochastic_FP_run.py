from ...Algorithms.Stochastic_FP.entropic_iterative_scheme import *
from ...Algorithms.data_manage import *
from .posterior_sampler import *
from .visualize_posteriors import *

'''
Running command from terminal: python -m WB_Algo.Experiments.Bike_Sharing.stochastic_FP_run
'''

if __name__ == "__main__":
    dim = 9
    num_samples = 10000
    num_measures = 5
    truncated_radius = 500
    multiplication_factor = 10
    MC_size = 20

    # DATA_DIR = os.path.dirname(__file__)
    # MODEL_DIR = os.path.join(DATA_DIR, "models_meta")
    # total_posterior_path = os.path.join(MODEL_DIR, "model_total.meta.pkl")
    
    # total_posterior_sampler = posterior_sampler(model_path=total_posterior_path, num_measures=1, multiplication_factor=multiplication_factor)
    # subset_posterior_sampler = posterior_sampler(model_path=MODEL_DIR, num_measures=num_measures, multiplication_factor=multiplication_factor)

    posterior_csv_dir = f"../WB_data/Bike_Sharing"
    total_posterior_sampler = csv_posterior_sampler(csv_dir=posterior_csv_dir, num_measures=1, multiplication_factor=multiplication_factor, type="full")
    split_posterior_sampler = csv_posterior_sampler(csv_dir=posterior_csv_dir, num_measures=num_measures, multiplication_factor=multiplication_factor, type="split")
    
    # Set up the entropic iterative computer
    entropic_iterative_computer = entropic_iterative_scheme(dim = dim, 
                                                            num_measures = num_measures, 
                                                            bary_sampler = total_posterior_sampler, 
                                                            input_sampler = split_posterior_sampler,
                                                            truncate_radius = truncated_radius)
    
    # input_samples_collection = entropic_iterative_computer.input_sampling(num_samples = num_samples)

    bary_sample_path = f"./WB_Algo/Experiments/Bike_Sharing/bary_samples_collection/bary_samples_collection_dim{dim}_MCsize50_numsamples10000.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}

    # print(bary_samples_collection_loaded["0"])

    data_dir = f"./WB_Algo/Experiments/Bike_Sharing/data_outputs/stochastic_FP_outputs"
    os.makedirs(data_dir, exist_ok=True)

    entropic_iterative_computer.converge(bary_samples_collection_loaded,
                                        # input_samples_collection,
                                        max_iter = 5,
                                        num_samples = num_samples,
                                        epsilon = 10,
                                        MC_size = MC_size,
                                        logger = {'sample_logger': None, 'map_logger': None},
                                        data_dir = data_dir,
                                        warm_start = True
                                        )