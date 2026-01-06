import numpy as np
import ot
# from .posterior_sampler import *
from Experiments.Bike_Sharing.metrics_to_compare import *
import json, os
from Algorithms.Fast_Cuturi.free_support_WB import w2_barycenter_free_support_from_samples
from Experiments.CSV_read import *

if __name__ == "__main__":
    dim = 9
    num_samples = 10000
    num_measures = 5
    truncated_radius = 1000
    multiplication_factor = 1
    MC_size = 20
    support_size = 500
    print("Setting up posterior samplers...")

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

    # posterior_csv_dir = f"../WB_data/Bike_Sharing"
    # total_posterior_sampler = csv_posterior_sampler(csv_dir=posterior_csv_dir, num_measures=1, multiplication_factor=multiplication_factor, type="full")
    # split_posterior_sampler = csv_posterior_sampler(csv_dir=posterior_csv_dir, num_measures=num_measures, multiplication_factor=multiplication_factor, type="split")

    bary_sample_path = f"../../WB_Data/Bike_Sharing/bary_samples_collection/bary_samples_collection_dim{dim}_MCsize50_numsamples10000.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}

    data_dir = f"../../WB_Data/Bike_Sharing/data_outputs/Fast_Cuturi_outputs/SupportSize{support_size}_NumSamples{num_samples}"
    os.makedirs(data_dir, exist_ok=True)
    V_values_dir = os.path.join(data_dir, "V_values")
    W2_to_bary_dir = os.path.join(data_dir, "W2_to_bary")
    os.makedirs(V_values_dir, exist_ok=True)
    os.makedirs(W2_to_bary_dir, exist_ok=True)

    V_values_path = os.path.join(V_values_dir, f"V_values.json")
    W2_to_bary_path = os.path.join(W2_to_bary_dir, f"W2_to_bary.json")

    V_values_list = []
    W2_to_bary_list = []
    for i in range(MC_size):
        print(f"Computing barycenter sample {i+1}/{MC_size}...")
        input_samples_collection = split_posterior_sampler.sample(num_samples)
        samples_list = [np.array(input_samples_collection[key]) for key in sorted(input_samples_collection.keys())]
        approx_bary = w2_barycenter_free_support_from_samples(
            samples_list,
            k=support_size,
            init="random",
            numItermax=200,
            verbose=True,
            seed=42,
        )

        input_samples_collection_for_evaluation = split_posterior_sampler_for_evaluation.sample(num_samples)
        bary_samples = bary_samples_collection_loaded[str(i)]

        # compute V-value
        V_value = 0
        for measure_index in range(num_measures):
            input_samples = np.array(input_samples_collection_for_evaluation[measure_index])
            V_value += W2_pot(input_samples, approx_bary)
        V_value /= num_measures
        V_values_list.append(V_value)
        print(f"V-value for barycenter sample {i}: {V_value}")

        # compute W2 to barycenter samples
        W2_sq = W2_pot(approx_bary, bary_samples)
        W2_to_bary_list.append(W2_sq)
        print(f"W2 squared to barycenter samples for barycenter sample {i}: {W2_sq}")

        with open(V_values_path, 'w') as json_file:
            json.dump(V_values_list, json_file) 

        with open(W2_to_bary_path, 'w') as json_file:
            json.dump(W2_to_bary_list, json_file)

    

   






    