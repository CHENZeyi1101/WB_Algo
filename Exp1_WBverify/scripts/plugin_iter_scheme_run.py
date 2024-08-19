import numpy as np
import gurobipy as gp
from gurobipy import GRB
import logging

import sys
import os

# Add the parent directory (Exp1_WBverify) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from classes.true_WB import *
from classes.input_generate import *
from classes.plugin_estimate_OT import *
from classes.ADMM import *
from classes.plugin_iterative_scheme import *
from classes.config_log import *
from classes.measure_visualize import *

seed = 1000
np.random.seed(seed)
data_path = "/Users/zeyichen/GitHub/Repo/WB_Algo/Exp1_WBverify/input_samples_6measures"
input_samples_dict = {k: np.array(v) for k, v in read_data(data_path, "input_samples_dict.json").items()}
source_samples = np.array(read_data(data_path, "source_samples.json"))
print(len(input_samples_dict))
print(source_samples.shape)

lambda_lower, lambda_upper = 0.001, 1000
smoothing = 'BA'
ADMM = False

dim = 2
num_samples = 50
num_measures = 6
iter = 0

source_sampler = MixtureOfGaussians(dim)
source_sampler.random_components(5)
source_sampler.set_truncation(100)

iteration = plugin_iterative_scheme(dim, 
                             num_measures, 
                             source_sampler, 
                             input_samples_dict, 
                             lambda_lower, 
                             lambda_upper,
                             ADMM, 
                             smoothing = smoothing, 
                             log = True)
iteration.converge(iter, num_samples)
