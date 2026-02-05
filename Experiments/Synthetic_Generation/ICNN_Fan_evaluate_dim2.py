from Algorithms.ICNN_Fan.CNX.cfg import CNXCfgCustom as Cfg_class
import Algorithms.ICNN_Fan.optimal_transport_modules.pytorch_utils as PTU
import Algorithms.ICNN_Fan.optimal_transport_modules.plot_utils as PLU
import Algorithms.ICNN_Fan.optimal_transport_modules.data_utils as DTU
import Algorithms.ICNN_Fan.CNX.compare_dist_results as CDR
from Experiments.Synthetic_Generation.metrics_to_compare import *
from Experiments.CSV_read import *
from pathlib import Path
import json, os
from tqdm import tqdm
import pandas as pd
import numpy as np

Cfg_PATH = Path(__file__).parent / "cfg.json"
with open(Cfg_PATH, "r") as f:
    cfg_dict = json.load(f)

params = cfg_dict["params_synthetic_generation_dim2"]

# take all items in params
num_samples = params["num_samples"]
dim = params["dim"]
num_measures = params["num_measures"]
truncated_radius = params["truncated_radius"]
instance_theta = params["instance_theta"]
num_components = params["num_components"]
MC_size = params["MC_size"]
instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}_toy"
# assert existence
assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

cfg = Cfg_class(DIM = dim, NUM_DISTRIBUTION=num_measures)

outputs_dir = f"{instance_dir}/outputs/ICNN_Fan_outputs"
assert os.path.exists(outputs_dir), f"Outputs directory {outputs_dir} does not exist."

evaluation_dir = os.path.join(outputs_dir, "evaluation_results")
os.makedirs(evaluation_dir, exist_ok=True)

# generate samples from approximated barycenter and save to csv
PTU.set_gpu_mode(False, 0)
for epoch_to_load in range(1, 101):
    for i in tqdm(range(MC_size), desc=f"Epoch {epoch_to_load}"):
        barycenter_samples = CDR.barycenter_sampler(
            cfg, PTU.device, outputs_dir, load_epoch=epoch_to_load
        )
        epoch_evaluation_save_path = f"{evaluation_dir}/csv_files/outputs_{epoch_to_load}"
        os.makedirs(epoch_evaluation_save_path, exist_ok=True)

        df = pd.DataFrame(barycenter_samples.detach().numpy())
        df.to_csv(f"{epoch_evaluation_save_path}/outputs_NWBFanTaghvaeiChen_samples_epoch_{epoch_to_load}_MCSample_{i}.csv",index=False, header=False)

# compute V values and W2 to true barycenter
bary_sample_path = f"{instance_dir}/samples_for_evaluation/bary_samples_collection_dim{dim}_MCsize50_numsamples{num_samples}.json"
with open(bary_sample_path, 'r') as json_file:
    bary_samples_collection_loaded = json.load(json_file)
bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}

# print(bary_samples_collection_loaded["0"])

input_sampler_for_evaluation = csv_input_sampler_for_evaluation_SyntheticGeneration(f"{instance_dir}/samples_for_evaluation", 
                                            num_measures, 
                                            multiplication_factor=1)
input_sampler_for_evaluation.set_streamers()

V_values_dict = {}
W2_to_true_bary_dict = {}

for epoch_to_load in range(1, 101, 5):
    # compute V values
    epoch_evaluation_save_path = f"{evaluation_dir}/csv_files/outputs_{epoch_to_load}"
    epoch_V_value_dict = {}
    epoch_W2_sq_dict = {}

    for i in tqdm(range(MC_size), desc=f"Epoch {epoch_to_load}"):
        approximated_bary_samples = pd.read_csv(f"{epoch_evaluation_save_path}/outputs_NWBFanTaghvaeiChen_samples_epoch_{epoch_to_load}_MCSample_{i}.csv", header = None).to_numpy()

        input_samples_collection = input_sampler_for_evaluation.sample(num_samples)
        bary_samples = bary_samples_collection_loaded[str(i)]

        V_value = V_value_compute(samples, input_samples_collection)
        W2_sq = W2_to_bary_compute(bary_samples, samples)

        epoch_V_value_dict[f"MCSample_{i}"] = V_value
        epoch_W2_sq_dict[f"MCSample_{i}"] = W2_sq

    V_values_dict[f"epoch_{epoch_to_load}"] = epoch_V_value_dict
    W2_to_true_bary_dict[f"epoch_{epoch_to_load}"] = epoch_W2_sq_dict

    V_value_dir = f"{evaluation_dir}/V_values"
    os.makedirs(V_value_dir, exist_ok=True)

    V_values_path = os.path.join(V_value_dir, f"V_values_epoch_{epoch_to_load}.json")

    with open(V_values_path, 'w') as json_file:
            json.dump(V_values_dict[f"epoch_{epoch_to_load}"], json_file)

    W2_to_bary_dir = f"{evaluation_dir}/W2_to_true_bary"
    os.makedirs(W2_to_bary_dir, exist_ok=True)

    W2_to_bary_path = os.path.join(W2_to_bary_dir, f"W2_to_true_bary_epoch_{epoch_to_load}.json")

    with open(W2_to_bary_path, 'w') as json_file:
            json.dump(W2_to_true_bary_dict[f"epoch_{epoch_to_load}"], json_file)
