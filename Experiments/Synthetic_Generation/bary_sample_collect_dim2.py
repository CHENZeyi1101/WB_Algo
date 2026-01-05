from Experiments.Synthetic_Generation.true_WB import *
from Experiments.Synthetic_Generation.samplers import *
import os
import json

if __name__ == "__main__":
    dim = 2 
    bary_MC_size = 50
    num_samples = 10000
    truncated_radius = 150
    source_sampler_seed = 1009

    source_sampler = MixtureOfGaussians(dim)
    source_sampler.random_components(num_components=5, uniform_weights = True, seed = source_sampler_seed)
    source_sampler.set_truncation(truncated_radius)

    bary_samples_collection = {}
    for i in range(bary_MC_size):
        bary_samples = source_sampler.sample(num_samples, seed = i+2000)
        bary_samples_collection[i] = bary_samples
    data_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_data/bary_samples_collection"
    os.makedirs(data_dir, exist_ok=True)

    # save as json after changing numpy array to list
    bary_samples_collection_list = {k: v.tolist() for k, v in bary_samples_collection.items()}
    json_path = os.path.join(data_dir, f"bary_samples_collection_dim{dim}_MCsize{bary_MC_size}_numsamples{num_samples}.json")
    with open(json_path, 'w') as json_file:
        json.dump(bary_samples_collection_list, json_file)

    # read back and change list to numpy array
    # with open(json_path, 'r') as json_file:
    #     bary_samples_collection_loaded = json.load(json_file)
    # bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}
    
