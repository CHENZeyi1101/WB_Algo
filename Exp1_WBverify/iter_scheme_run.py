import numpy as np
import gurobipy as gp
from gurobipy import GRB
import logging

from true_WB import *
from input_generate import *
from estimate_OT import *
from ADMM import QCQP_ADMM
from iterative_scheme import *
from config_log import *

data_path = "input_samples"
input_samples_dict = {k: np.array(v) for k, v in read_data(data_path, "input_samples_dict_BA.json").items()}
# print(input_samples_dict["measure_0"].shape)

lambda_lower, lambda_upper = 0.01, 1000
smoothing = 'BA'
ADMM = False

dim = 2
num_samples = 50
num_measures = 6
iter = 0

source_sampler = MixtureOfGaussians(dim)
source_sampler.random_components(5)
source_sampler.set_truncation(100)

iteration = iterative_scheme(dim, 
                             num_measures, 
                             source_sampler, 
                             input_samples_dict, 
                             lambda_lower, 
                             lambda_upper,
                             ADMM, 
                             smoothing = smoothing, 
                             log = True)
iteration.converge(iter, num_samples)
