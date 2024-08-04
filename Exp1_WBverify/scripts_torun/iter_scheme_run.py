import numpy as np
import gurobipy as gp
from gurobipy import GRB
import logging

from ..classes.true_WB import *
from ..classes.input_generate import *
from ..classes.estimate_OT import *
from ..classes.ADMM import *
from ..classes.iterative_scheme import *
from ..classes.config_log import *
from ..classes.measure_visualize import *

seed = 1000
np.random.seed(seed)
data_path = "input_samples_1"
input_samples_dict = {k: np.array(v) for k, v in read_data(data_path, "input_samples_dict_BA.json").items()}
source_samples = np.array(read_data(data_path, "source_samples_BA.json"))
print(len(input_samples_dict))
print(source_samples.shape)

# breakpoint()
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
