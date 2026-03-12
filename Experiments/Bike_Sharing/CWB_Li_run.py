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


if __name__ == '__main__':
    np.random.seed(44)
    tf.random.set_seed(44)

    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(gpu_devices) > 0:
    #     tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    # ---- FORCE CPU ONLY ----
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # ---- Reproducibility ----
    np.random.seed(44)
    tf.random.set_seed(44)

    # ---- Reduce TF logging ----
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # ---- (Optional) verify ----
    print("GPUs visible to TF:", tf.config.list_physical_devices("GPU"))
    print("CPUs visible to TF:", tf.config.list_physical_devices("CPU"))

    ##############################

    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_posterior_aggregation_dim8"]

    samples_dir = cfg_dict['samples_dir']
    # assert existence
    assert os.path.exists(samples_dir), f"Instance directory {samples_dir} does not exist."

    # take all items in params
    dim = params["dim"]
    num_measures = params["num_measures"]
    instance_identifier = params["instance_identifier"]
    MC_size = params["MC_size"]
    eval_num_samples = params["eval_num_samples"]

    instance_dir = cfg_dict['data_dir']
    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    dim_range = [dim]
    repeat_range = range(1)

    exp = "BikeSharing"

    input_csv_path = samples_dir

    g_base_dir = f"{instance_dir}/outputs/CWB_Li_outputs"
    os.makedirs(g_base_dir, exist_ok=True)

    sample_count = eval_num_samples * MC_size

    batch_run_exp(exp, "cwb", repeat_range=repeat_range, input_csv_path=input_csv_path, num_measures=num_measures, g_base_dir=g_base_dir, dim_range=dim_range, sample_count=sample_count, params=params)

    