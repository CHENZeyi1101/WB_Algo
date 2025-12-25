from .true_WB import *
from .samplers_dim2 import *
import os
import json

if __name__ == "__main__":
    dim = 2 
    bary_MC_size = 50
    num_samples = 10000

    source_sampler = MixtureOfGaussians(dim)
    load_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/samplers_info"
    source_sampler = load_sampler(load_dir, source_sampler, sampler_type="source")

    bary_samples_collection = {}
    for i in range(bary_MC_size):
        bary_samples = source_sampler.sample(num_samples)
        bary_samples_collection[i] = bary_samples
    data_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/bary_samples_collection"
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
    
