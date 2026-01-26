import os
import json
from Experiments.CSV_read import *
# from .posterior_sampler import *
if __name__ == "__main__":
    dim = 9
    bary_MC_size = 20
    num_samples = 10000
    num_measures = 5
    multiplication_factor = 1

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
    bary_samples_collection = {}
    for i in range(bary_MC_size):
        bary_samples = total_posterior_sampler.sample(num_samples)
        bary_samples_collection[i] = bary_samples

    data_dir = f"../../WB_Data/Bike_Sharing/bary_samples_collection"
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
    
