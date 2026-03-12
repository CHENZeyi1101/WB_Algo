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

    params = cfg_dict["params_synthetic_generation_dim2"]

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

    # fetch approximated barycenter samples across iterations
    G_samples_dir = os.path.join(outputs_dir, "G_samples")
    assert os.path.exists(G_samples_dir), f"G_samples directory {G_samples_dir} does not exist."
    n_iters = len(os.listdir(G_samples_dir))

    '''
    Measure visualization
    '''

    plot_radius = params["truncated_radius"]
    
    for i in range(n_iters):
        with open(f'{G_samples_dir}/G_samples_iter{i}.json', 'r') as f:
            samples_dict = json.load(f)
        approx_samples = np.array(samples_dict[f"iteration_{i}"][0]) # there are MC_size runs, we take the first one for visualization
        print(f"Starts plotting Iter_{i}")
        plot_2d_measures_kde(approx_samples, bins = 400, plot_radius = plot_radius, scatter=False, plot_dirc=plot_dir, plot_name=f"approx_bary_kde_iter{i}", title=f"Iteration {i}")
        print(f"Iter_{i} plotted.")

    image_paths = [f"{plot_dir}/approx_bary_kde_iter{i}.png" for i in range(5)]
    
    combine_images_row(image_paths, save_path= f"{plot_dir}/approx_bary_kde_iters_combined.png", figsize=(24, 12))
    print("Combined approximated barycenter KDEs.")


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

    # images_path_evaluation = [f"{plot_dir}/V_values_iters.png", f"{plot_dir}/W2_to_bary_iters.png"]
    # combine_images_row(images_path_evaluation, save_path= f"{plot_dir}/evaluation_iters_combined.pdf", figsize=(24, 12))
    # print("Combined evaluation plots across iterations.")


