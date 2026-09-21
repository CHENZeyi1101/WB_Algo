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

    params = cfg_dict["params_posterior_aggregation_dim8"]

    outputs_dir = f"{cfg_dict['outputs_dir']}/stochastic_FP_outputs"
    assert os.path.exists(outputs_dir), f"Outputs directory {outputs_dir} does not exist."

    plot_dir = f"{outputs_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    '''
    V-value and W2-to-bary visualization
    '''
    v_values_path = f"{outputs_dir}/V_values/V_values_iter9.json"
    assert os.path.exists(v_values_path), f"V values file {v_values_path} does not exist."
    true_v_value_OT_path = f"{cfg_dict['outputs_dir']}/fullpost_via_OT/fullpost_V_values.json"
    assert os.path.exists(true_v_value_OT_path), f"True V value file {true_v_value_OT_path} does not exist."

    W2_to_bary_path = f"{outputs_dir}/W2_to_bary/W2_to_bary_iter9.json"
    assert os.path.exists(W2_to_bary_path), f"W2-to-bary file {W2_to_bary_path} does not exist."
    true_w2_to_bary_OT_path = f"{cfg_dict['outputs_dir']}/fullpost_via_OT/fullpost_W2_to_bary_OT.json"
    assert os.path.exists(true_w2_to_bary_OT_path), f"True W2-to-bary file {true_w2_to_bary_OT_path} does not exist."

    plot_v_values(v_values_path, true_v_OT_path=true_v_value_OT_path, plot_dir=plot_dir, plot_name="V_values_iters_maxmin", BS=True)
    plot_w2_values(W2_to_bary_path, true_OT_path=true_w2_to_bary_OT_path, plot_dir=plot_dir, plot_name="W2_to_bary_iters_maxmin", BS=True)
