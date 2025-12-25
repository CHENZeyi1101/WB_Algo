import os
import json
from .posterior_sampler import *
if __name__ == "__main__":
    dim = 8
    bary_MC_size = 50
    num_samples = 10000

    posterior_csv_dir = f"../WB_data/Bike_Sharing"
    total_posterior_sampler = csv_posterior_sampler(csv_dir=posterior_csv_dir, num_measures=1, multiplication_factor=1, type="full")

    bary_samples_collection = {}
    for i in range(bary_MC_size):
        bary_samples = total_posterior_sampler.sample(num_samples)
        bary_samples_collection[i] = bary_samples
    data_dir = f"./WB_Algo/Experiments/Bike_Sharing/bary_samples_collection"
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
    
