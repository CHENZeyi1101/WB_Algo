from Algorithms.ICNN_Fan.CNX.cfg import CNXCfgCustom as Cfg_class
import Algorithms.ICNN_Fan.optimal_transport_modules.pytorch_utils as PTU
import Algorithms.ICNN_Fan.optimal_transport_modules.plot_utils as PLU
import Algorithms.ICNN_Fan.optimal_transport_modules.data_utils as DTU
import Algorithms.ICNN_Fan.CNX.compare_dist_results as CDR
from Experiments.CSV_read import *
from pathlib import Path
import json, os
from tqdm import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool
from Algorithms.data_manage import save_json
from Experiments.metrics_to_compare import evaluate_zipped
from Experiments.metrics_to_compare import evaluate_MC
from Experiments.Synthetic_Generation.visualize_measures_dim2 import plot_2d_measures_kde

if __name__ == "__main__":

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]

    # take all items in params
    dim = params["dim"]
    num_measures = params["num_measures"]
    instance_identifier = params["instance_identifier"]
    MC_size = params["MC_size"]
    num_samples = params["num_samples"]
    eval_num_samples = params["eval_num_samples"]


    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    cfg = Cfg_class(DIM = dim, NUM_DISTRIBUTION=num_measures, N_TEST = eval_num_samples)

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

    outputs_dir = f"{instance_dir}/outputs/ICNN_Fan_outputs_cpu"
    assert os.path.exists(outputs_dir), f"Outputs directory {outputs_dir} does not exist."

    evaluation_dir = os.path.join(outputs_dir, "evaluation_results")
    os.makedirs(evaluation_dir, exist_ok=True)
    V_values_dir = os.path.join(evaluation_dir, "V_values")
    W2_to_bary_dir = os.path.join(evaluation_dir, "W2_to_bary")
    os.makedirs(V_values_dir, exist_ok=True)
    os.makedirs(W2_to_bary_dir, exist_ok=True)

    # generate samples from approximated barycenter and save to csv
    max_epoch = 500
    PTU.set_gpu_mode(False, 0)

    input_measure_samples_collection_it = [{k : input_samples_collection_loaded[i][k][:eval_num_samples] for k in range(num_measures)} for i in range(MC_size)]
    true_bary_samples_it = [bary_samples_collection_loaded[i][:eval_num_samples] for i in range(MC_size)]

 
    epoch_to_load = 100
    MC_size = 20
    epoch_evaluation_dir = f"{evaluation_dir}/outputs_{epoch_to_load}"
    os.makedirs(epoch_evaluation_dir, exist_ok=True)
    approx_bary_it = []
    for i in tqdm(range(MC_size), desc=f"Epoch {epoch_to_load}"):
        barycenter_samples = CDR.barycenter_sampler(
            cfg, PTU.device, outputs_dir, load_epoch=epoch_to_load
        )
        barycenter_samples_np = barycenter_samples.detach().numpy()
        approx_bary_it.append(barycenter_samples_np)
        df = pd.DataFrame(barycenter_samples_np)
        df.to_csv(f"{epoch_evaluation_dir}/outputs_NWBFanTaghvaeiChen_samples_epoch_{epoch_to_load}_MCSample_{i}.csv",index=False, header=False)

    V_values_list, W2_to_bary_list = evaluate_MC(approx_bary_it, 
                                                input_measure_samples_collection_it, 
                                                true_bary_samples_it, 
                                                MC_size = MC_size, 
                                                num_parallel_process = None, 
                                                pbar_text = "Evaluation of ICNN_Fan")
    
    V_values_dict = {
        "mean": np.mean(V_values_list),
        "std": np.std(V_values_list),
        "values": V_values_list}
    save_json(V_values_dict, V_values_dir, f"V_values_epoch{epoch_to_load}.json")

    W2_to_bary_dict = {
        "mean": np.mean(W2_to_bary_list),
        "std": np.std(W2_to_bary_list),
        "values": W2_to_bary_list}
    save_json(W2_to_bary_dict, W2_to_bary_dir, f"W2_to_bary_epoch{epoch_to_load}.json")

    # plot_dir = f"{evaluation_dir}/plots"
    # os.makedirs(plot_dir, exist_ok=True)

    # plot_2d_measures_kde(approx_bary_it[0], bins = 400, plot_radius = 150, scatter=False, plot_dirc=f"{plot_dir}/epoch_{epoch_to_load}", plot_name=f"ICNN_Fan_epoch_{epoch_to_load}_measures_kde.png")
    # print(f"Measure from Epoch {epoch_to_load} KDE plotted.")






    














