from Algorithms.Stochastic_FP.entropic_iterative_scheme import *
from Algorithms.data_manage import *
from Experiments.Bike_Sharing.visualize_posteriors import *
from Experiments.CSV_read import *

if __name__ == "__main__":
    dim = 9
    num_samples = 10000
    num_measures = 5
    truncated_radius = 1000
    multiplication_factor = 1
    MC_size = 20
    max_iter = 5
    epsilon = 10 ** -8
    warm_start = False
    data_dir = f"../../WB_Data/Bike_Sharing/data_outputs/stochastic_FP_outputs"
    os.makedirs(data_dir, exist_ok=True)

    ##### Set up the samplers for Bike Sharing data #####
    csv_dir = f"../../WB_data/Bike_Sharing"
    print("CSV directory exists.")
    total_posterior_sampler = csv_posterior_sampler_BikeSharing(csv_dir=csv_dir, 
                                                    num_measures=num_measures, 
                                                    multiplication_factor=multiplication_factor, 
                                                    type="full",
                                                    usecols=range(7, 16),
                                                    skiprows=52)
    total_posterior_sampler.set_streamers()
    print("Total posterior sampler set up.")
    split_posterior_sampler = csv_posterior_sampler_BikeSharing(csv_dir, 
                                                num_measures, 
                                                multiplication_factor, 
                                                type="split",
                                                usecols=range(7, 16),
                                                skiprows=52)
    split_posterior_sampler.set_streamers()
    print("Split posterior sampler set up.")

    split_posterior_sampler_for_evaluation = csv_posterior_sampler_BikeSharing(csv_dir, 
                                                num_measures, 
                                                multiplication_factor, 
                                                type="split",
                                                usecols=range(7, 16),
                                                skiprows=6000000) # adjust skiprows for the samples used for evaluation
    split_posterior_sampler_for_evaluation.set_streamers()
    print("Split posterior sampler for evaluation set up.")

    
    ##### Set up the entropic iterative computer #####
    entropic_iterative_computer = entropic_iterative_scheme(dim = dim, 
                                                            num_measures = num_measures, 
                                                            bary_sampler = total_posterior_sampler, 
                                                            input_sampler = split_posterior_sampler,
                                                            truncate_radius = truncated_radius)
    
    ##### Load the barycenter samples (for MC comparison) #####
    bary_sample_path = f"../../WB_Data/Bike_Sharing/bary_samples_collection/bary_samples_collection_dim{dim}_MCsize50_numsamples10000.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}

    ##### Run the stochastic FP algorithm with entropic OT map estimation #####
    entropic_iterative_computer.converge(bary_samples_collection_loaded,
                                        split_posterior_sampler_for_evaluation,
                                        max_iter = max_iter,
                                        num_samples = num_samples,
                                        epsilon = epsilon,
                                        MC_size = MC_size,
                                        logger = {'sample_logger': None, 'map_logger': None},
                                        data_dir = data_dir,
                                        warm_start = warm_start
                                        )