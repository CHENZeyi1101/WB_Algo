from Algorithms.CWB_Li.cwb.tests.comparison.common import *
from Algorithms.CWB_Li.cwb.tests.comparison.batch import batch_run_exp
from Algorithms.CWB_Li.cwb.tests.comparison import validate
from Experiments.CSV_read import *
import json
from pathlib import Path

import argparse
import tensorflow as tf
import os
import subprocess
import numpy as np
import time
import pickle

'''
Caveat: check customized_template.yaml in Algorithms/CWB_Li/cwb/templates to ensure the parameters are set correctly.
'''


if __name__ == '__main__':
    
    # ---- Reproducibility ----
    np.random.seed(44)
    tf.random.set_seed(44)

    # ---- Reduce TF logging ----
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


    ##############################

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

    dim_range = [dim]
    repeat_range = range(1)

    exp = "SyntheticGeneration"

    input_csv_path = f"{instance_dir}/input_samples/csv_files"

    g_base_dir = f"{instance_dir}/outputs/CWB_Li_outputs"
    os.makedirs(g_base_dir, exist_ok=True)

    sample_count = eval_num_samples * MC_size

    batch_run_exp(exp, "cwb", repeat_range=repeat_range, input_csv_path=input_csv_path, num_measures=num_measures, g_base_dir=g_base_dir, dim_range=dim_range, sample_count=sample_count)

    