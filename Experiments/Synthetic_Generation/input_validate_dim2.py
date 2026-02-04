from Experiments.Synthetic_Generation.samplers import *
import pandas as pd
from Experiments.CSV_read import *
from pathlib import Path
import json, os
from tqdm import tqdm

if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]
    dim = params["dim"]
    truncated_radius = params["truncated_radius"]
    instance_identifier = params["instance_identifier"]
    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"

    entropic_sampler = entropic_input_sampler.load_from_file(load_dir = f"{instance_dir}/samplers_info")
    num_measures = entropic_sampler.num_measures

    print(entropic_sampler.smoothness_param_dict)

    # Generate grid
    grid_size_x = 400
    grid_size_y = 400
    xx = np.linspace(-truncated_radius, truncated_radius, grid_size_x)
    yy = np.linspace(-truncated_radius, truncated_radius, grid_size_y)
    grid_x, grid_y = np.meshgrid(xx, yy)
    input_mat = np.array([grid_x.flatten('F'), grid_y.flatten('F')]).T
    Brenier_grad_mat_list = [np.zeros_like(input_mat) for _ in range(num_measures)]
    component_grad_mat_list = [np.zeros_like(input_mat) for _ in range(num_measures * 2)]

    Brenier_sc_list = np.zeros(num_measures)
    Brenier_sm_list = np.zeros(num_measures)
    component_sc_list = np.zeros(num_measures * 2)
    component_sm_list = np.zeros(num_measures * 2)
    OT_diff_mat = np.zeros((input_mat.shape[0], dim))

    for i in tqdm(range(input_mat.shape[0]), f"Computing OT map"):
        Brenier_grad, component_grad = entropic_sampler.generate_input_measure_sample(input_mat[i, :])

        OT_diff = -input_mat[i, :]
        
        for k in range(num_measures):
            Brenier_grad_mat_list[k][i, :] = Brenier_grad[k]
            OT_diff += Brenier_grad[k] / num_measures
        
        for tilde_k in range(num_measures * 2):
            component_grad_mat_list[tilde_k][i, :] = component_grad[tilde_k]
        
        OT_diff_mat[i, :] = OT_diff


    for k in range(num_measures): 
        strong_convexity_LB = np.inf
        smoothness_UB = -np.inf

        for i in tqdm(range(input_mat.shape[0]), f"Checking OT map {k}"):
            norm_sq = np.sum((input_mat - input_mat[i, :]) ** 2, axis=1)
            norm_sq[i] = 1
            innerprod_vec = np.sum((input_mat - input_mat[i, :]) * (Brenier_grad_mat_list[k] - Brenier_grad_mat_list[k][i, :]), axis=1) / norm_sq
            innerprod_vec[i] = np.inf
            strong_convexity_LB = np.minimum(strong_convexity_LB, np.min(innerprod_vec))
            innerprod_vec[i] = -np.inf
            smoothness_UB = np.maximum(smoothness_UB, np.max(innerprod_vec))
        
        Brenier_sc_list[k] = strong_convexity_LB
        Brenier_sm_list[k] = smoothness_UB
        print(f"Strong convexity of OT map {k}: {strong_convexity_LB}")
        print(f"Smoothness of OT map {k}: {smoothness_UB}")
    
    for tilde_k in range(num_measures * 2): 
        strong_convexity_LB = np.inf
        smoothness_UB = -np.inf

        for i in tqdm(range(input_mat.shape[0]), f"Checking component map {tilde_k}"):
            norm_sq = np.sum((input_mat - input_mat[i, :]) ** 2, axis=1)
            norm_sq[i] = 1
            innerprod_vec = np.sum((input_mat - input_mat[i, :]) * (component_grad_mat_list[tilde_k] - component_grad_mat_list[tilde_k][i, :]), axis=1) / norm_sq
            innerprod_vec[i] = np.inf
            strong_convexity_LB = np.minimum(strong_convexity_LB, np.min(innerprod_vec))
            innerprod_vec[i] = -np.inf
            smoothness_UB = np.maximum(smoothness_UB, np.max(innerprod_vec))
        
        component_sc_list[tilde_k] = strong_convexity_LB
        component_sm_list[tilde_k] = smoothness_UB
        print(f"Strong convexity of component map {tilde_k}: {strong_convexity_LB}")
        print(f"Smoothness of component map {tilde_k}: {smoothness_UB}")
    
    print('\n')
    print('Estimated strong convexity parameters of the Brenier potentials:')
    print(Brenier_sc_list)

    print('Estimated smoothness parameters of the Brenier potentials:')
    print(Brenier_sm_list)
    print('\n')

    print('Estimated strong convexity parameters of the component potentials:')
    print(component_sc_list)

    print('Estimated smoothness parameters of the component potentials:')
    print(component_sm_list)
    print('\n')

    OT_diff_norm_max = np.max(np.linalg.norm(OT_diff_mat, axis=1))

    print(f"Maximum norm of difference between the weighted sum of OT maps and the identity: {OT_diff_norm_max}")

    if np.all(Brenier_sc_list >= 0):
        if OT_diff_norm_max <= 1e-5:
            print('Convexity conditions and the optimality condition are satisfied, hence the synthetically generated instance is VALID')
        else:
            print('Convexity conditions are satisfied, but the optimality condition is not satisfied, hence the synthetically generated instance is INVALID')
    else:
        print('Convexity conditions are not satisfied, hence the synthetically generated instance is INVALID')