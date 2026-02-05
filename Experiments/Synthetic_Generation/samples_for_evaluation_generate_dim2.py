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
    num_samples = params["eval_num_samples"]
    instance_identifier = params["instance_identifier"]
    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"

    entropic_sampler = entropic_input_sampler.load_from_file(load_dir = f"{instance_dir}/samplers_info")
    source_sampler = entropic_sampler.source_sampler
    source_sampler.set_master_rng(np.random.SeedSequence(params["seeds"]["evaluation_source_sampling_seed"]))

    bary_MC_size = params["MC_size"]
    num_samples_in_preparation = 10**7

    data_dir = f"{instance_dir}/samples_for_evaluation"
    os.makedirs(data_dir, exist_ok=True)

    # sample from the true barycenter
    bary_samples_collection = {}
    for i in range(bary_MC_size):
        print(f"Generating bary sample of MC step {i+1}/{bary_MC_size} ...")
        bary_samples = source_sampler.sample(num_samples)
        bary_samples_collection[i] = bary_samples
    
    # save as json after changing numpy array to list
    bary_samples_collection_tolist = {k: v.tolist() for k, v in bary_samples_collection.items()}
    json_path = os.path.join(data_dir, f"bary_samples_collection_dim{dim}_MCsize{bary_MC_size}_numsamples{num_samples}.json")
    with open(json_path, 'w') as json_file:
        json.dump(bary_samples_collection_tolist, json_file)

    # Generate input samples for evaluation
    csv_evaluate_dir = f"{instance_dir}/samples_for_evaluation"
    os.makedirs(csv_evaluate_dir, exist_ok=True)
    
    input_measure_samples_for_evaluation = entropic_sampler.sample(num_samples_in_preparation)

    for measure_index in range(entropic_sampler.num_measures):
        measure_samples = np.asarray(input_measure_samples_for_evaluation[measure_index])
        csv_filename = f"{csv_evaluate_dir}/input_measure_samples_{measure_index}_for_evaluation.csv"
        pd.DataFrame(measure_samples).to_csv(csv_filename, index=False, header=False)
    print("Input samples for evaluation saved to CSV files.")
    
