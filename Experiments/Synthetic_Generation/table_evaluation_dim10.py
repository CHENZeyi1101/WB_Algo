import math
import os, json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from Experiments.visualize_evaluations import export_latex_table
from Algorithms.data_manage import save_json, read_json

if __name__ == "__main__":
    '''
    Experiment: 10d synthetic generation 
    '''

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim10"]
    dim = params["dim"]
    instance_identifier = params["instance_identifier"]
    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."
    all_outputs_dir = f"{instance_dir}/outputs"
    assert os.path.exists(all_outputs_dir), f"Outputs directory {all_outputs_dir} does not exist."


    stochastic_FP_dir = f"{all_outputs_dir}/stochastic_FP_outputs"
    ICNN_Fan_dir = f"{all_outputs_dir}/ICNN_Fan_outputs_cpu"
    WIN_Korotin_dir = f"{all_outputs_dir}/WIN_Korotin_outputs_cpu"
    CWB_Li_dir = f"{all_outputs_dir}/CWB_Li_outputs"
    Fast_Cuturi_dir = f"{all_outputs_dir}/Fast_Cuturi_outputs"


    V_value_stochastic_FP_path = f"{stochastic_FP_dir}/V_values/V_values_iter9.json"
    V_value_ICNN_Fan_path = f"{ICNN_Fan_dir}/evaluation_results/V_values/V_values_epoch100.json"
    V_value_WIN_Korotin_path = f"{WIN_Korotin_dir}/evaluation_results/V_values/V_values_iter100100.json"
    V_value_CWB_li_path = f"{CWB_Li_dir}/V_values/V_values.json"
    V_value_Fast_Cuturi_path = f"{Fast_Cuturi_dir}/V_values/V_values.json"
    assert os.path.exists(V_value_stochastic_FP_path), f"File {V_value_stochastic_FP_path} does not exist."
    assert os.path.exists(V_value_ICNN_Fan_path), f"File {V_value_ICNN_Fan_path} does not exist."
    assert os.path.exists(V_value_WIN_Korotin_path), f"File {V_value_WIN_Korotin_path} does not exist."
    assert os.path.exists(V_value_CWB_li_path), f"File {V_value_CWB_li_path} does not exist."
    assert os.path.exists(V_value_Fast_Cuturi_path), f"File {V_value_Fast_Cuturi_path} does not exist."
    

    W2_to_bary_stochastic_FP_path = f"{stochastic_FP_dir}/W2_to_bary/W2_to_bary_iter9.json"
    W2_to_bary_ICNN_Fan_path = f"{ICNN_Fan_dir}/evaluation_results/W2_to_bary/W2_to_bary_epoch100.json"
    W2_to_bary_WIN_Korotin_path = f"{WIN_Korotin_dir}/evaluation_results/W2_to_bary/W2_to_bary_iter100100.json"
    W2_to_bary_CWB_li_path = f"{CWB_Li_dir}/W2_to_bary/W2_to_bary.json"
    W2_to_bary_Fast_Cuturi_path = f"{Fast_Cuturi_dir}/W2_to_bary/W2_to_bary.json"
    assert os.path.exists(W2_to_bary_stochastic_FP_path), f"File {W2_to_bary_stochastic_FP_path} does not exist."
    assert os.path.exists(W2_to_bary_ICNN_Fan_path), f"File {W2_to_bary_ICNN_Fan_path} does not exist."
    assert os.path.exists(W2_to_bary_WIN_Korotin_path), f"File {W2_to_bary_WIN_Korotin_path} does not exist."
    assert os.path.exists(W2_to_bary_CWB_li_path), f"File {W2_to_bary_CWB_li_path} does not exist."
    assert os.path.exists(W2_to_bary_Fast_Cuturi_path), f"File {W2_to_bary_Fast_Cuturi_path} does not exist."
    

    # read files and organize results for table
    with open(V_value_stochastic_FP_path, "r") as f:
        V_value_stochastic_FP = json.load(f)['iteration_9']
    with open(V_value_ICNN_Fan_path, "r") as f:
        V_value_ICNN_Fan = json.load(f)
    with open(V_value_WIN_Korotin_path, "r") as f:
        V_value_WIN_Korotin = json.load(f)
    with open(V_value_CWB_li_path, "r") as f:
        V_value_CWB_Li = json.load(f)   
    with open(V_value_Fast_Cuturi_path, "r") as f:
        V_value_Fast_Cuturi = json.load(f)
    

    with open(W2_to_bary_stochastic_FP_path, "r") as f:
        W2_to_bary_stochastic_FP = json.load(f)['iteration_9']
    with open(W2_to_bary_ICNN_Fan_path, "r") as f:
        W2_to_bary_ICNN_Fan = json.load(f)
    with open(W2_to_bary_WIN_Korotin_path, "r") as f:
        W2_to_bary_WIN_Korotin = json.load(f)
    with open(W2_to_bary_CWB_li_path, "r") as f:    
        W2_to_bary_CWB_Li = json.load(f)
    with open(W2_to_bary_Fast_Cuturi_path, "r") as f:    
        W2_to_bary_Fast_Cuturi = json.load(f)
    


    results = {
    'stochastic_FP': [
        V_value_stochastic_FP, 
        W2_to_bary_stochastic_FP
        ],
    'ICNN_Fan': [
        V_value_ICNN_Fan, 
        W2_to_bary_ICNN_Fan
        ],
    'WIN_Korotin': [
        V_value_WIN_Korotin, 
        W2_to_bary_WIN_Korotin
        ],
    'CWB_Li': [
        V_value_CWB_Li, 
        W2_to_bary_CWB_Li
        ],
    'Fast_Cuturi': [
        V_value_Fast_Cuturi, 
        W2_to_bary_Fast_Cuturi
        ]
    }

    true_V_value_dir = f"{all_outputs_dir}/true_V_value"
    true_V_value_path = f"{true_V_value_dir}/true_V_value.json"
    with open(true_V_value_path, "r") as f:
        true_V_value = json.load(f)

    true_v = true_V_value["mean"]

    export_latex_table(
        results,
        output_path=f"{all_outputs_dir}/10d_results_table.txt",
        metric_names = [
            f"V-values ($V(\\bar{{\\mu}}) = {true_v:.4f}$)",
            r"2-Wasserstein distance to $\bar{\mu}$"
        ],
        caption="Performances of algorithms in Experiment [SG-10d]",
        label="tab:results"
    )