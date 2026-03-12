import matplotlib.pyplot as plt
import math
import os, json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from Experiments.Synthetic_Generation.visualize_measures_dim2 import plot_2d_measures_kde, combine_images_row
from Experiments.visualize_evaluations import plot_v_values, plot_w2_values

if __name__ == "__main__":
    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim10"]

    # take all items in params
    dim = params["dim"]
    num_measures = params["num_measures"]
    instance_identifier = params["instance_identifier"]

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    outputs_dir = f"{instance_dir}/outputs/stochastic_FP_outputs"
    assert os.path.exists(outputs_dir), f"Outputs directory {outputs_dir} does not exist."

    plot_dir = f"{outputs_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)


    '''
    V-value and W2-to-bary visualization
    '''
    v_values_path = f"{outputs_dir}/V_values/V_values_iter9.json"
    assert os.path.exists(v_values_path), f"V values file {v_values_path} does not exist."
    true_v_path = f"{instance_dir}/outputs/true_V_value/true_V_value.json"
    assert os.path.exists(true_v_path), f"True V value file {true_v_path} does not exist."

    W2_to_bary_path = f"{outputs_dir}/W2_to_bary/W2_to_bary_iter9.json"
    assert os.path.exists(W2_to_bary_path), f"W2-to-bary file {W2_to_bary_path} does not exist."

    plot_v_values(v_values_path, true_v_path, plot_dir=plot_dir, plot_name="V_values_iters")
    plot_w2_values(W2_to_bary_path, plot_dir=plot_dir, plot_name="W2_to_bary_iters")

    
