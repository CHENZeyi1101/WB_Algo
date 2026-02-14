import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from multiprocessing import Pool
from Algorithms.data_manage import save_json
from Experiments.metrics_to_compare import evaluate_zipped

from Algorithms.WIN_Korotin.src.icnn import DenseICNN_U
from Algorithms.WIN_Korotin.src.plotters import plot_training_phase
from Algorithms.WIN_Korotin.src.tools import ewma, score_gen, freeze, unfreeze
from Algorithms.WIN_Korotin.src.fid_score import calculate_frechet_distance
from Algorithms.WIN_Korotin.src import distributions
from Algorithms.WIN_Korotin.src import bar_benchmark

def load_models_from_iteration(model_save_dir, iteration, 
                               G, Ts, Ds, Ts_inv=None, Ds_inv=None,
                               device="cpu"):

    model_path = os.path.join(
        model_save_dir,
        f"trained_models_iter_{iteration}.pth"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No checkpoint found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    G.load_state_dict(checkpoint["G"])

    for k in checkpoint["Ts"]:
        Ts[int(k)].load_state_dict(checkpoint["Ts"][k])

    for k in checkpoint["Ds"]:
        Ds[int(k)].load_state_dict(checkpoint["Ds"][k])

    if Ts_inv is not None:
        for k in checkpoint["Ts_inv"]:
            Ts_inv[int(k)].load_state_dict(checkpoint["Ts_inv"][k])

    if Ds_inv is not None:
        for k in checkpoint["Ds_inv"]:
            Ds_inv[int(k)].load_state_dict(checkpoint["Ds_inv"][k])

    G.to(device).eval()
    for k in range(len(Ts)):
        Ts[k].to(device).eval()
        Ds[k].to(device).eval()
        if Ts_inv is not None:
            Ts_inv[k].to(device).eval()
        if Ds_inv is not None:
            Ds_inv[k].to(device).eval()

    print(f"Loaded models from iteration {iteration}")

    return G, Ts, Ds, Ts_inv, Ds_inv


if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]

    # take all items in params
    num_samples = params["num_samples"]
    eval_num_samples = params["eval_num_samples"]
    dim = params["dim"]
    num_measures = params["num_measures"]
    truncated_radius = params["truncated_radius"]
    instance_identifier = params["instance_identifier"]
    num_components = params["num_components"]
    MC_size = params["MC_size"]

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    outputs_dir = f"{instance_dir}/outputs/WIN_Korotin_outputs"
    assert os.path.exists(outputs_dir), f"WIN_Korotin outputs directory {outputs_dir} does not exist."
    # define the save path
    model_save_dir = f"{outputs_dir}/trained_models"
    assert os.path.exists(model_save_dir), f"Model save directory {model_save_dir} does not exist."

    DEVICE = cfg_dict["devices"]["WIN_Korotin"]


    # load samples for evaluation
    eval_dir = f"{instance_dir}/samples_for_evaluation"

    bary_sample_path = f"{eval_dir}/bary_samples_collection.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {int(k): np.array(v) for k, v in bary_samples_collection_loaded.items()}

    input_sample_path = f"{eval_dir}/input_samples_collection.json"
    with open(input_sample_path, 'r') as json_file:
        input_samples_collection_loaded = json.load(json_file)
    input_samples_collection_loaded = {int(k): {int(i): np.array(u) for i, u in v.items()}
                                        for k, v in input_samples_collection_loaded.items()}


    '''
    Neural networks setup
    '''
    '''''
    Discriminator setup
    '''''
    D = nn.Sequential(
    nn.Linear(dim, max(100, 2*dim)),
    nn.ReLU(True),
    nn.Linear(max(100, 2*dim), max(100, 2*dim)),
    nn.ReLU(True),
    nn.Linear(max(100, 2*dim), max(100, 2*dim)),
    nn.ReLU(True),
    nn.Linear(max(100, 2*dim), 1)
    ).to(DEVICE)

    T = nn.Sequential(
        nn.Linear(dim, max(100, 2*dim)),
        nn.ReLU(True),
        nn.Linear(max(100, 2*dim), max(100, 2*dim)),
        nn.ReLU(True),
        nn.Linear(max(100, 2*dim), max(100, 2*dim)),
        nn.ReLU(True),
        nn.Linear(max(100, 2*dim), dim)
    ).to(DEVICE)

    Ds = nn.ModuleList([deepcopy(D) for _ in range(num_measures)]).to(DEVICE) #.cuda()
    Ts = nn.ModuleList([deepcopy(T) for _ in range(num_measures)]).to(DEVICE) #.cuda()

    Ds_inv = nn.ModuleList([deepcopy(D) for _ in range(num_measures)]).to(DEVICE) #.cuda()
    Ts_inv = nn.ModuleList([deepcopy(T) for _ in range(num_measures)]).to(DEVICE) #.cuda()

    '''
    Generator setup
    '''
    Z_sampler = distributions.StandardNormalSampler(dim=dim, device=DEVICE)
    # Z_sampler = distributions.StandardNormalSampler(dim=dim, device='cuda')

    G = nn.Sequential(
    nn.Linear(dim, max(100, 2*dim)),
    nn.ReLU(True),
    nn.Dropout(0.005),
    nn.Linear(max(100, 2*dim), max(100, 2*dim)),
    nn.ReLU(True),
    nn.Dropout(0.005),
    nn.Linear(max(100, 2*dim), max(100, 2*dim)),
    nn.ReLU(True),
    nn.Linear(max(100, 2*dim), dim)
    ).to(DEVICE)



    ###########################################

    evaluation_dir = os.path.join(outputs_dir, "evaluation_results")
    os.makedirs(evaluation_dir, exist_ok=True)
    V_values_dir = os.path.join(evaluation_dir, "V_values")
    W2_to_bary_dir = os.path.join(evaluation_dir, "W2_to_bary")
    os.makedirs(V_values_dir, exist_ok=True)
    os.makedirs(W2_to_bary_dir, exist_ok=True)

    iterations_to_load = range(1000, 100001, 1000)

    input_measure_samples_collection_it = [{k : input_samples_collection_loaded[i][k][:eval_num_samples] for k in range(num_measures)} for i in range(MC_size)]
    true_bary_samples_it = [bary_samples_collection_loaded[i][:eval_num_samples] for i in range(MC_size)]

    for iteration in iterations_to_load:
        iter_evaluation_dir = f"{evaluation_dir}/outputs_iteration_{iteration}"
        model_path = os.path.join(
            model_save_dir,
            f"trained_models_iter_{iteration}.pth"
        )

        if not os.path.exists(model_path):
            print(f"Checkpoint {iteration} not found — skipping")
            continue

        checkpoint = torch.load(model_path, map_location="cpu")

        G.load_state_dict(checkpoint["G"])
        for k in checkpoint["Ts"]:
            Ts[int(k)].load_state_dict(checkpoint["Ts"][k])
        for k in checkpoint["Ds"]:
            Ds[int(k)].load_state_dict(checkpoint["Ds"][k])

        G.eval()
        for k in range(len(Ts)):
            Ts[k].eval()
            Ds[k].eval()

        print(f"Loaded iteration {iteration}")

        seed_list = [iteration * 100 + i for i in range(MC_size)]
        approx_bary_it = []

        for i in range(MC_size):
            accepted_G_samples = (
                        G(Z_sampler.sample(size = eval_num_samples, seed = seed_list[i]))
                        .detach()
                        .cpu()
                        .numpy()
                    )
            approx_bary_it.append(accepted_G_samples)

            df = pd.DataFrame(accepted_G_samples)
            df.to_csv(f"{iter_evaluation_dir}/outputs_WIN_samples_iteration_{iteration}_MCSample_{i}.csv",index=False, header=False)

        V_values_list = []
        W2_to_bary_list = []
        with Pool(processes = 5) as pool, tqdm(total = MC_size) as pbar:
            for V_value, W2_to_bary in pool.imap(evaluate_zipped, 
                                    zip(approx_bary_it, 
                                        input_measure_samples_collection_it, 
                                        true_bary_samples_it)):
                V_values_list.append(V_value)
                W2_to_bary_list.append(W2_to_bary)
                pbar.update(1)
                pbar.refresh()

        # save V-values and W2_to_bary values
        V_values_dict = {
            "mean": np.mean(V_values_list),
            "std": np.std(V_values_list),
            "values": V_values_list}
        save_json(V_values_dict, V_values_dir, f"V_values_iter{iteration}.json")

        W2_to_bary_dict = {
            "mean": np.mean(W2_to_bary_list),
            "std": np.std(W2_to_bary_list),
            "values": W2_to_bary_list}
        save_json(W2_to_bary_dict, W2_to_bary_dir, f"W2_to_bary_iter{iteration}.json")

            
            



