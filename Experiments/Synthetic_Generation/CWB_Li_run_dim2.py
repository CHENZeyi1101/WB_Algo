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
    params = cfg_dict["params_synthetic_generation_dim2"]

    dim = params["dim"]
    dim_range = [dim]
    num_measures = params["num_measures"]
    sample_count = params["num_samples"]
    instance_theta = params["instance_theta"]

    repeat_range = range(1)

    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/InstanceTheta{instance_theta}_toy"
    # assert existence
    assert os.path.exists(instance_dir), f"Instance directory {instance_dir} does not exist."

    exp = "SyntheticGeneration"

    input_csv_path = f"{instance_dir}/input_samples/csv_files"

    g_base_dir = f"{instance_dir}/outputs/CWB_Li_outputs"

    batch_run_exp(exp, "cwb", repeat_range=repeat_range, input_csv_path=input_csv_path, num_measures=num_measures, g_base_dir=g_base_dir, dim_range=dim_range, sample_count=10)

    