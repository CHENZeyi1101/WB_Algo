import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from .input_sample_dim2 import *
from tqdm import tqdm, tqdm_notebook

import warnings
warnings.filterwarnings('ignore')
from IPython.display import clear_output

import os, sys
sys.path.append("..")

import torch
from torch import nn
from torch.nn import functional as F

from scipy.stats import ortho_group
from scipy.linalg import sqrtm

from Algorithms.data_manage import *
from Algorithms.WIN_Korotin.src.icnn import DenseICNN_U
from Algorithms.WIN_Korotin.src.plotters import plot_training_phase
from Algorithms.WIN_Korotin.src.tools import ewma, score_gen, freeze, unfreeze
from Algorithms.WIN_Korotin.src.fid_score import calculate_frechet_distance
from Algorithms.WIN_Korotin.src import distributions
from Algorithms.WIN_Korotin.src import bar_benchmark
from Experiments.Synthetic_Generation.metrics_to_compare import W2_pot
import itertools

import gc
from sklearn.decomposition import PCA

from copy import deepcopy

import ot

if __name__ == "__main__":

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
    
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())

    GPU_DEVICE = 0 # GPU index starting from 0
    BATCH_SIZE = 256 #1024

    LAMBDA = 10
    G_LR = 1e-4
    D_LR = 1e-3
    MAX_ITER = 10001

    D_ITERS = 50
    T_ITERS = 10
    G_ITERS = 50

    PLOT_FREQ = 499
    SCORE_FREQ = 499

    # Parameters for input distributions
    NUM = 5 # we have 5 input measures
    ALPHAS = np.array([1. / NUM for _ in range(NUM)])

    CASE = {
        'type' : 'EigWarp', 
        'sampler' : 'Rectangles', #'Gaussians', #'SwissRoll',# , #
        'params' : {'num' : NUM, 'alphas' : ALPHAS, 'min_eig' : .5, 'max_eig' : 2}
    }

    OUTPUT_SEED = 0xBADBEEF

    # assert torch.cuda.is_available()
    # torch.cuda.set_device(GPU_DEVICE)

    # np.random.seed(OUTPUT_SEED)
    # torch.manual_seed(OUTPUT_SEED)
    # ---------------- DEVICE SETUP ----------------
    DEVICE = torch.device("cpu")

    print("Using device:", DEVICE)

    torch.manual_seed(OUTPUT_SEED)
    np.random.seed(OUTPUT_SEED)


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
    Z_sampler = distributions.StandardNormalSampler(dim=dim, device='cpu')
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

    G_opt = torch.optim.Adam(G.parameters(), lr=1e-4, weight_decay=1e-8)
    loss = np.inf

    G.train(True)

    for iteration in tqdm_notebook(range(10000)):
        Z = Z_sampler.sample(BATCH_SIZE).detach() * 3
        loss = F.mse_loss(Z, G(Z))
        loss.backward()
        G_opt.step(); G_opt.zero_grad()
        print(loss.item())
        if loss.item() < 1e-2:
            break

    print(loss)

    '''
    Main training loop
    '''

    G_opt = torch.optim.Adam(G.parameters(), lr=G_LR, weight_decay=1e-10)
    Ts_opt, Ds_opt = [], []
    Ts_inv_opt, Ds_inv_opt = [], []
    for k in range(num_measures):
        Ts_opt.append(torch.optim.Adam(Ts[k].parameters(), lr=D_LR, weight_decay=1e-10))
        Ds_opt.append(torch.optim.Adam(Ds[k].parameters(), lr=D_LR, weight_decay=1e-10))
        Ts_inv_opt.append(torch.optim.Adam(Ts_inv[k].parameters(), lr=D_LR, weight_decay=1e-10))
        Ds_inv_opt.append(torch.optim.Adam(Ds_inv[k].parameters(), lr=D_LR, weight_decay=1e-10))
    G_loss_history = []
    G_UVP_history = []

    # if hasattr(benchmark, 'gauss_bar_cost'):
    #     print('Gaussian Barycenter Cost:', benchmark.gauss_bar_cost)

    it = 0
    last_plot_it = -1
    last_score_it = -1

    # load_dir = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/samplers_info"
    # source_sampler = MixtureOfGaussians(dim)
    # source_sampler = load_sampler(load_dir, source_sampler, sampler_type="source")
    # source_measure_samples = source_sampler.sample(num_samples)

    # # Load the input measures samplers
    # csv_path = f"./WB_Algo/Experiments/Synthetic_Generation/dim{dim}_data/input_samples/csv_files"
    # csv_sampler = csv_input_sampler(dim = dim, num_measures = num_measures, csv_path = csv_path)

    source_component_seed = cfg_dict["source_components_seed"]
    master_source_rng = np.random.SeedSequence(cfg_dict["master_source_sampling_seed"])
    source_sampler = characterize_source_sampler(dim = dim, 
                                                num_components = num_components, 
                                                master_sampling_rng = master_source_rng,
                                                component_seed = source_component_seed,
                                                truncated_radius = truncated_radius,
                                                save_dir = None)

    input_csv_path = f"{instance_dir}/input_samples/csv_files"
    input_sampler = csv_input_sampler_SyntheticGeneration(input_csv_path, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler.set_streamers()

    bary_sample_path = f"{instance_dir}/samples_for_evaluation/bary_samples_collection_dim{dim}_MCsize{MC_size}_numsamples{num_samples}.json"
    with open(bary_sample_path, 'r') as json_file:
        bary_samples_collection_loaded = json.load(json_file)
    bary_samples_collection_loaded = {k: np.array(v) for k, v in bary_samples_collection_loaded.items()}
 
    eval_dir = f"{instance_dir}/samples_for_evaluation"
    input_sampler_for_evaluation = csv_input_sampler_for_evaluation_SyntheticGeneration(eval_dir, 
                                                num_measures, 
                                                multiplication_factor=1)
    input_sampler_for_evaluation.set_streamers()


    outputs_dir = f"{instance_dir}/outputs/WIN_Korotin_outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    # define the save path
    model_save_dir = f"{outputs_dir}/trained_models"
    os.makedirs(model_save_dir, exist_ok=True)

    G_samples_dict = {}
    V_values_dict = {}
    W2_to_true_bary_dict = {}

    while it < MAX_ITER:
        freeze(G)
        input_measure_samples_for_D = input_sampler.sample(BATCH_SIZE * D_ITERS)
        input_measure_samples_for_T_inv = input_sampler.sample(BATCH_SIZE * D_ITERS * T_ITERS)
        input_measure_samples_for_D_inv = input_sampler.sample(BATCH_SIZE * D_ITERS)
        # this is a dictionary with k keys, pointing to the samples collected from each measure
        for k in range(num_measures):
            # D and T optimization cycle
            for d_iter in tqdm(range(D_ITERS)):
                it += 1

                # T optimization
                unfreeze(Ts[k]); freeze(Ds[k])
                for t_iter in range(T_ITERS): 
                    with torch.no_grad():
                        X = G(Z_sampler.sample(BATCH_SIZE))
                    Ts_opt[k].zero_grad()
                    T_X = Ts[k](X)
                    T_loss = F.mse_loss(X, T_X).mean() - Ds[k](T_X).mean()
                    T_loss.backward(); Ts_opt[k].step()
                del T_loss, T_X, X
                gc.collect()

                # D optimization
                with torch.no_grad():
                    X = G(Z_sampler.sample(BATCH_SIZE))
                # Y = torch.tensor(input_measure_samples_for_D[k][d_iter * BATCH_SIZE : (d_iter + 1) * BATCH_SIZE]).float()
                Y = torch.tensor(
                    input_measure_samples_for_D[k][d_iter * BATCH_SIZE : (d_iter + 1) * BATCH_SIZE],
                    dtype=torch.float32,
                    device=DEVICE
                )

                
                unfreeze(Ds[k]); freeze(Ts[k])
                T_X = Ts[k](X).detach()
                Ds_opt[k].zero_grad()
                D_loss = Ds[k](T_X).mean() - Ds[k](Y).mean()
                D_loss.backward(); Ds_opt[k].step()
                del D_loss, Y, X, T_X
                gc.collect()
                # torch.cuda.empty_cache()
                
                # T inv optimization
                unfreeze(Ts_inv[k]); freeze(Ds_inv[k])
                for t_iter in range(T_ITERS): 
                    Y = torch.tensor(input_measure_samples_for_T_inv[k][d_iter * T_ITERS * BATCH_SIZE + t_iter * BATCH_SIZE : d_iter * T_ITERS * BATCH_SIZE + (t_iter + 1) * BATCH_SIZE]).float()
                    Ts_inv_opt[k].zero_grad()
                    T_inv_Y = Ts_inv[k](Y)
                    T_inv_loss = F.mse_loss(Y, T_inv_Y).mean() - Ds_inv[k](T_inv_Y).mean()
                    T_inv_loss.backward(); Ts_inv_opt[k].step()
                del T_inv_loss, T_inv_Y, Y
                gc.collect()
                # torch.cuda.empty_cache()

                # D inv optimization
                Y = torch.tensor(input_measure_samples_for_D_inv[k][d_iter * BATCH_SIZE : (d_iter + 1) * BATCH_SIZE]).float()
                with torch.no_grad():
                    X = G(Z_sampler.sample(BATCH_SIZE))
                
                unfreeze(Ds_inv[k]); freeze(Ts_inv[k])
                T_inv_Y = Ts_inv[k](Y).detach()
                Ds_inv_opt[k].zero_grad()
                D_inv_loss = Ds_inv[k](T_inv_Y).mean() - Ds_inv[k](X).mean()
                D_inv_loss.backward(); Ds_inv_opt[k].step()
                del D_inv_loss, Y, X, T_inv_Y
                gc.collect()
                torch.cuda.empty_cache()

            
        # G optimization
        if G_ITERS > 0:
            for k in range(num_measures):
                freeze(Ts[k])
            G_old = deepcopy(G); freeze(G_old)
            unfreeze(G)
            for g_iter in range(G_ITERS):
                it += 1
                Z = Z_sampler.sample(BATCH_SIZE)
                with torch.no_grad():
                    G_old_Z = G_old(Z)
                    T_G_old_Z = torch.zeros_like(G_old(Z))
                G_old_Z.requires_grad_(True)
                for k in range(num_measures):
                    T_G_old_Z += ALPHAS[k] * Ts[k](G_old_Z)

                G_opt.zero_grad()
                G_loss = .5 * F.mse_loss(G(Z), T_G_old_Z)
                G_loss.backward(); G_opt.step() 

                G_loss_history.append(G_loss.item())

            model_save_path = f"{model_save_dir}/trained_models_iter_{it}.pth"
            # os.makedirs(save_path, exist_ok=True)
            models_to_save = {
                "G": G.state_dict(),
                "Ts": {k: Ts[k].state_dict() for k in range(num_measures)},
                "Ds": {k: Ds[k].state_dict() for k in range(num_measures)},
                # Save the inverse transforms if needed
                "Ts_inv": {k: Ts_inv[k].state_dict() for k in range(num_measures)},
                "Ds_inv": {k: Ds_inv[k].state_dict() for k in range(num_measures)},
            }

            # Save the dictionary
            torch.save(models_to_save, model_save_path)
            print(f"Models saved to {model_save_path}")

                # Log G_loss_history to a local file
            with open("G_loss_history.log", "a") as f:
                f.write(f"Iteration {it}, G_loss: {G_loss.item()}\n")

            del G_old, G_loss, T_G_old_Z, Z
            gc.collect()


            for MC_iter in range(MC_size):
                print(f"Computing metrics for MC sample {MC_iter+1}/{MC_size} at iteration {it}...")
                # Save the generated samples from the G-mapping at each iteration
                # accepted_G_samples = G(Z_sampler.sample(num_samples)).cuda().detach().numpy() #.cpu()
                accepted_G_samples = (
                    G(Z_sampler.sample(num_samples))
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Save the generated samples from the G-mapping at each iteration;
                G_samples_dict[f"iteration_{iter}"] = accepted_G_samples
                G_samples_json = {str(k): v.tolist() for k, v in G_samples_dict.items()}
                G_sample_dir = f"{outputs_dir}/G_samples"
                os.makedirs(G_sample_dir, exist_ok=True)
                with open(f"{G_sample_dir}/G_samples.json", 'w') as f:
                    json.dump(G_samples_json, f, indent=4)

                # Compute the V-value
                input_samples_collection_for_evaluation = input_sampler_for_evaluation.sample(num_samples)
                V_value = 0
                for measure_index in range(num_measures):
                    input_samples = np.array(input_samples_collection_for_evaluation[measure_index])
                    V_value += W2_pot(input_samples, accepted_G_samples)
                # normalize the V_value by the number of input measures
                V_value /= num_measures
                V_values_dict[f"iteration_{it}"] = V_value
                V_value_dir = f"{outputs_dir}/V_values"
                os.makedirs(V_value_dir, exist_ok=True)
                with open(f"{V_value_dir}/V_values.json", 'w') as f:
                    json.dump(V_values_dict, f, indent=4) 

                bary_samples = bary_samples_collection_loaded[str(MC_iter)]
                W2_sq = W2_pot(accepted_G_samples, bary_samples)
                W2_to_true_bary_dict[f"iteration_{it}"] = W2_sq
                W2_to_true_bary_json = W2_to_true_bary_dict
                W2_to_true_bary_dir = f"{outputs_dir}/W2_to_true_bary"
                os.makedirs(W2_to_true_bary_dir, exist_ok=True)
                with open(f"{W2_to_true_bary_dir}/W2_to_true_bary.json", 'w') as f:
                    json.dump(W2_to_true_bary_json, f, indent=4)


        print(f"Iteration {it} completed.")