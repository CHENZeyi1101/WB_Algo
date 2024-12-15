import numpy as np
import gurobipy as gp
from gurobipy import GRB
import logging

# parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# sys.path.append(parent_directory)

import sys
import os

# Add the parent directory (Exp1_WBverify) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from classes.true_WB import *
from Exp1_WBverify.classes.input_generate_plugin import *
from classes.plugin_estimate_OT import *
from classes.ADMM import *
from classes.plugin_iterative_scheme import *
from classes.config_log import *
from classes.measure_visualize import *
from classes.data_operate import *
from classes.feed_sample import *


dim = 2
num_measures = 5
log = True
num_samples = 50000

seed = 500
np.random.seed(seed)

##################### BA generate samples #####################

input_func_logger_BA, input_measure_logger_BA, input_func_log_file_BA, input_measure_log_file_BA = input_logger(log, "BA")
source_sampler_BA, input_measure_sampler_BA = feed_samples(num_measures = num_measures,
                                                    num_samples = num_samples,
                                                    dim = dim,
                                                    log = log,
                                                    smoothing = "BA",
                                                    input_func_logger = input_func_logger_BA,
                                                    input_measure_logger = input_measure_logger_BA,
                                                    input_func_log_file = input_func_log_file_BA,
                                                    plot = True,
                                            )
source_samples_BA = input_measure_sampler_BA.source_samples
input_samples_dict_BA = input_measure_sampler_BA.sample_collection
# save both data
data_path = "input_samples_num5"
os.makedirs(data_path, exist_ok=True)
source_samples_BA_json = source_samples_BA.tolist()
save_data(source_samples_BA_json, data_path, "source_samples.json")
input_samples_dict_BA_json = {str(k): v.tolist() for k, v in input_samples_dict_BA.items()}
save_data(input_samples_dict_BA_json, data_path, "input_samples_dict.json")
dist_visualize(input_samples_dict_BA, source_samples_BA, oneplot = False, save = True, savefile = "measure_visualization_num5.png")





# commands to convert back
# source_samples = np.array(read_data(data_path, "source_samples.json"))
# input_samples_dict = {int(k): np.array(v) for k, v in read_data(data_path, "input_samples_dict.json").items()}

##################### KS generate samples #####################

# input_func_logger_KS, input_measure_logger_KS, input_func_log_file_KS, input_measure_log_file_KS = input_logger(log, "KS")
# source_sampler_KS, input_measure_sampler_KS = feed_samples(num_measures,
#                                                     num_samples = 10000,
#                                                     dim = dim,
#                                                     log = log,
#                                                     smoothing = "KS",
#                                                     input_func_logger = input_func_logger_KS,
#                                                     input_measure_logger = input_measure_logger_KS,
#                                                     input_func_log_file = input_func_log_file_KS,
#                                                     plot = True,
#                                                     plot_savefile = "input_measures.png"
#                                             )
# source_samples_KS = input_measure_sampler_KS.source_samples
# input_samples_dict_KS = input_measure_sampler_KS.sample_collection

# # save both data
# source_samples_KS_json = source_samples_KS.tolist()
# save_data(source_samples_KS_json, data_path, "source_samples_KS.json")
# input_samples_dict_KS_json = {str(k): v.tolist() for k, v in input_samples_dict_KS.items()}
# save_data(input_samples_dict_KS_json, data_path, "input_samples_dict_KS.json")

################# check done: KS and BA generate same results #################













    