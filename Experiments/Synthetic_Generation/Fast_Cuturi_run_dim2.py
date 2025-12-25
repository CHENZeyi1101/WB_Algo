import numpy as np
import ot
from .samplers_dim2 import *
from .metrics_to_compare import *
from .input_generate_entropic import *
import json, os
from ...Algorithms.Fast_Cuturi.free_support_WB import w2_barycenter_free_support_from_samples

if __name__ == "__main__":
    dim = 2
    num_samples = 10000
    num_measures = 5
    truncated_radius = 150
    # multiplication_factor = 10
    MC_size = 20

    load_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/samplers_info"
    source_sampler = MixtureOfGaussians(dim)
    source_sampler = load_sampler(load_dir, source_sampler, sampler_type="source")
    csv_path = f"../WB_data/Synthetic_Generation/dim{dim}_data/input_samples/csv_files"
    csv_sampler = csv_input_sampler(dim = dim, num_measures = num_measures, csv_path = csv_path)

    bary_sample_path = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/bary_samples_collection/bary_samples_collection_dim{dim}_MCsize50_numsamples10000.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}

    data_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/Fast_Cuturi_outputs"
    os.makedirs(data_dir, exist_ok=True)
    V_values_dir = os.path.join(data_dir, "V_values")
    W2_to_bary_dir = os.path.join(data_dir, "W2_to_bary")
    os.makedirs(V_values_dir, exist_ok=True)
    os.makedirs(W2_to_bary_dir, exist_ok=True)

    V_values_list = []
    W2_to_bary_list = []
    for i in range(MC_size):
        print(f"Computing barycenter sample {i+1}/{MC_size}...")
        input_samples_collection = csv_sampler.sample(num_samples)
        samples_list = [np.array(input_samples_collection[key]) for key in sorted(input_samples_collection.keys())]
        approx_bary = w2_barycenter_free_support_from_samples(
            samples_list,
            k=10000,
            init="kmeans",
            numItermax=300,
            verbose=True,
            seed=42,
        )
        bary_samples = bary_samples_collection_loaded[str(i)]

        # compute V-value
        V_value = 0
        for measure_index in range(num_measures):
            input_samples = np.array(input_samples_collection[measure_index])
            V_value += W2_pot(input_samples, approx_bary)
        V_value /= num_measures
        V_values_list.append(V_value)
        print(f"V-value for barycenter sample {i}: {V_value}")

        # compute W2 to barycenter samples
        W2_sq = W2_pot(approx_bary, bary_samples)
        W2_to_bary_list.append(W2_sq)
        print(f"W2 squared to barycenter samples for barycenter sample {i}: {W2_sq}")

        # save V-values and W2_to_bary values
        V_values_path = os.path.join(V_values_dir, f"V_values.json")
        with open(V_values_path, 'w') as json_file:
            json.dump(V_values_list, json_file) 

        W2_to_bary_path = os.path.join(W2_to_bary_dir, f"W2_to_bary.json")
        with open(W2_to_bary_path, 'w') as json_file:
            json.dump(W2_to_bary_list, json_file)

    

   






    