import pandas as pd
from Experiments.Synthetic_Generation.input_generate_entropic import entropic_input_sampler
from Experiments.CSV_read import *
from pathlib import Path
import json, os

if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]
    dim = params["dim"]
    instance_identifier = params["instance_identifier"]
    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"

    entropic_sampler = entropic_input_sampler.load_from_file(load_dir = f"{instance_dir}/samplers_info")
    entropic_sampler.source_sampler.set_master_rng(np.random.SeedSequence(params["seeds"]["master_source_sampling_seed"]))
    num_samples_in_preparation = 10**7

    # Generate input samples
    csv_path = f"{instance_dir}/input_samples/csv_files"
    os.makedirs(csv_path, exist_ok=True)
    
    input_measure_samples = entropic_sampler.sample(num_samples_in_preparation)

    for measure_index in range(entropic_sampler.num_measures):
        measure_samples = np.asarray(input_measure_samples[measure_index])
        csv_filename = os.path.join(csv_path, f"input_measure_samples_{measure_index}.csv")
        pd.DataFrame(measure_samples).to_csv(csv_filename, index=False, header=False)
    print("Input samples saved to CSV files.")

    
    