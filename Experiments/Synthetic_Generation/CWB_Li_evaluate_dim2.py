import numpy as np
import os, json
from pathlib import Path

from Algorithms.CWB_Li.cwb.tests.comparison.common import *

Cfg_PATH = Path(__file__).parent / "cfg.json"
with open(Cfg_PATH, "r") as f:
    cfg_dict = json.load(f)

params = cfg_dict["params_synthetic_generation_dim2"]

# take all items in params
num_samples = params["num_samples"]
dim = params["dim"]
num_measures = params["num_measures"]
truncated_radius = params["truncated_radius"]
instance_identifier = params["instance_identifier"]
num_components = params["num_components"]
MC_size = params["MC_size"]

instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
# assert existence
assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

g_base_dir = f"{instance_dir}/outputs/CWB_Li_outputs/"

result_dir = get_result_nd_dir(dim, g_base_dir)
result_filename = get_result_filename("cwb", 0)

result_npz_path = os.path.join(result_dir, result_filename)

cwb_data = np.load(result_npz_path)
print(cwb_data.shape)

'''
To do: evaluate V-values and W2.
'''


# arr = np.load("file.npy")
