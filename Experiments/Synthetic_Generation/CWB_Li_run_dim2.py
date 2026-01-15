from Algorithms.CWB_Li.cwb.tests.comparison.common import *
from Algorithms.CWB_Li.cwb.tests.comparison.batch import batch_run_exp
from Algorithms.CWB_Li.cwb.tests.comparison import validate
from Experiments.CSV_read import *

import argparse
import tensorflow as tf
import os
import subprocess
import numpy as np
import time
import pickle


if __name__ == '__main__':
    # np.random.seed(44)
    # tf.random.set_seed(44)

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
    exp = "SyntheticGeneration"
    dim = 2
    dim_range = [dim]
    num_measures = 5

    input_csv_path = f"../../WB_data/Synthetic_Generation/dim{dim}_data/input_samples/csv_files_InstanceTheta2000"

    batch_run_exp(exp, "cwb", repeat_range=range(1), input_csv_path=input_csv_path, num_measures=num_measures)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('exp', type=str)
    # parser.add_argument('--dims', nargs='+', type=int, required=True)
    # parser.add_argument('--gen_data', action='store_true')
    # parser.add_argument('--adapt_h5', action='store_true')
    # parser.add_argument('--run', type=str)
    # parser.add_argument('--validate', type=str)
    # parser.add_argument('--evolve', type=str)
    # parser.add_argument('--repeat_start', type=int, default=0)
    # parser.add_argument('--repeat_times', type=int, default=1)
    # parser.add_argument('--reseed', action='store_true')

    # args = parser.parse_args()

    # if args.reseed:
    #     t = int(time.time() * 1000.0) & 0xffffffff
    #     np.random.seed(t)
    #     tf.random.set_seed(t)

    # exp = args.exp
    # repeat_start = args.repeat_start
    # repeat_times = args.repeat_times
    # repeat_range = range(repeat_start, repeat_start + repeat_times)

    # if args.dims:
    #     dim_range = args.dims

    # if args.gen_data:
    #     batch_gen_data(exp)

    # if args.adapt_h5:
    #     batch_adapt_data_to_h5(exp)

    # if args.run is not None:
    #     method = args.run
    #     batch_run_exp(exp, method, repeat_range)

    # if args.validate is not None:
    #     method = args.validate
    #     batch_validate_exp(exp, method, repeat_range)

    # if args.evolve is not None:
    #     method = args.evolve
    #     batch_evolve_exp(exp, method, repeat_range)
