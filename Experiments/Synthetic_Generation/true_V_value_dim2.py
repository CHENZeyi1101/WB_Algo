from pathlib import Path
import json, os
import numpy as np

from Experiments.Synthetic_Generation.input_generate_entropic import entropic_input_sampler
from Algorithms.data_manage import save_json

if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]
    dim = params["dim"]
    instance_identifier = params["instance_identifier"]
    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"

    entropic_sampler = entropic_input_sampler.load_from_file(load_dir = f"{instance_dir}/samplers_info")
    entropic_sampler.source_sampler.set_master_rng(np.random.SeedSequence(params["seeds"]["true_V_val_source_sampling_seed"]))

    MC_sample_size = 10**7
    max_num_saved_samples = 10**4
    [V_mean, V_std, V_vec, distsq_mat] = entropic_sampler.compute_true_V_value(MC_sample_size)

    outputs_dir = f"{instance_dir}/outputs/true_V_value"
    os.makedirs(outputs_dir, exist_ok=True)
    output_dict = {
        "mean": V_mean,
        "std": V_std,
        "sample_size": MC_sample_size,
        "values": V_vec[:max_num_saved_samples].tolist(),
        "dist_values": distsq_mat[:max_num_saved_samples, :].tolist()
    }

    save_json(output_dict, outputs_dir, 'true_V_value.json')

    
    # sample_size = 10000
    # num_rep = 20

    # [V_mean, V_std, V_vec] = entropic_sampler.compute_true_V_value_via_OT(sample_size=sample_size, num_rep=num_rep)

    # outputs_dir = f"{instance_dir}/outputs/true_V_value_OT"
    # os.makedirs(outputs_dir, exist_ok=True)
    # output_dict = {
    #     "mean": V_mean,
    #     "std": V_std,
    #     "sample_size": sample_size,
    #     "values": V_vec.tolist()
    # }

    # save_json(output_dict, outputs_dir, 'true_V_value_OT.json')


    