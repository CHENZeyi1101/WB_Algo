from Experiments.Synthetic_Generation.MOG import *
from Experiments.Synthetic_Generation.samplers import *
from Experiments.CSV_read import *
from Experiments.CSV_shuffle import *
from tqdm import tqdm
import os
import json
from pathlib import Path

if __name__ == "__main__":
    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]
    dim = params["dim"]
    instance_identifier = params["instance_identifier"]
    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"

    input_sampler = entropic_input_sampler.load_from_file(load_dir = f"{instance_dir}/samplers_info")
    source_sampler = input_sampler.source_sampler
    source_sampler.set_master_rng(np.random.SeedSequence(params["seeds"]["evaluation_source_sampling_seed"]))

    eval_MC_size = params["MC_size"]
    eval_num_samples = params["eval_num_samples"]

    data_dir = f"{instance_dir}/samples_for_evaluation"
    os.makedirs(data_dir, exist_ok=True)

    # sample from the true barycenter and the input measures
    bary_samples_collection = {}
    input_samples_collection = {}
    for i in range(eval_MC_size):
        print(f"Generating bary sample of MC step {i+1}/{eval_MC_size} ...")
        bary_samples = source_sampler.sample(eval_num_samples)
        input_samples = input_sampler.sample(eval_num_samples)
        bary_samples_collection[i] = bary_samples.tolist()
        input_samples_collection[i] = {k : v.tolist() for k, v in input_samples.items()}
    
    # save as json
    bary_json_path = os.path.join(data_dir, f"bary_samples_collection.json")
    with open(bary_json_path, 'w') as json_file:
        json.dump(bary_samples_collection, json_file)
    
    input_json_path = os.path.join(data_dir, f"input_samples_collection.json")
    with open(input_json_path, 'w') as json_file:
        json.dump(input_samples_collection, json_file)
    
