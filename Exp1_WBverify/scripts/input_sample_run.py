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
from classes.input_generate import *
from Exp1_WBverify.classes.plugin_estimate_OT import *
from classes.ADMM import *
from Exp1_WBverify.classes.plugin_iterative_scheme import *
from classes.config_log import *
from classes.measure_visualize import *
from classes.data_operate import *


def feed_samples(num_measures, 
                 num_samples,
                 dim, 
                 log = False, 
                 smoothing = "BA", 
                 input_func_logger = None, 
                 input_measure_logger = None, 
                 input_func_log_file = None,
                 plot = False,
                 ):
    
    # FUNCTIONALITY:
    # 1. "Feed" suffiently many input samples to the input_measyre_sampler before the iterative scheme starts.
    # 2. To reduce dependency, the input samples of each measure would be reordered randomly.
    # 2. In each iteration of the algorithm, one may just extract samples of each input measure.

    raw_func_list = []
    for i in range(num_measures):
        x_samples = np.random.uniform(low = -50, high = 50, size=(100, dim))

        # log the sample points for generating the convex function
        if log:
            input_func_logger.info(f"Sample points for CvxFunction_{i}: {x_samples}")

        cvxfunc_generator = convex_function(x_samples, num_functions = num_measures, log = log, logger = input_func_logger)

        if i % 2 == 0:
            x_values, x_gradients, max_indices = cvxfunc_generator.generate_quadratic_sqrt()
        else:
            x_values, x_gradients, max_indices = cvxfunc_generator.generate_quadratic_sq()
        # x_values, x_gradients, max_indices = cvxfunc_generator.generate_quadratic_sqrt()
        # x_values, x_gradients, max_indices = cvxfunc_generator.generate_quadratic_sq()
        if plot:
            plot_dir = "cvx_func_plots"
            os.makedirs(plot_dir, exist_ok=True)
            cvxfunc_generator.plot_func(x_values, max_indices, name = f"{plot_dir}/cvx_func_{i}.png")
        cvx_otmap_generator = cvx_based_OTmap(x_samples, x_values, x_gradients, log = log)

        # initialize parameters of cvx_otmap_generator
        input_func_logger.info(f"####### Shape and Interpolation Parameters for CvxFunction_{i} #######")
        # cvx_otmap_generator.shape_paras(seed = 5 + i, logger = input_func_logger) #4, 5
        cvx_otmap_generator.shape_paras(logger = input_func_logger)
        cvx_otmap_generator.interp_paras(logger = input_func_logger)
        raw_func_list.append(cvx_otmap_generator)

    if log:
        input_func_logger.info("######### Finished generating raw functions #########")

    source_sampler = MixtureOfGaussians(dim)
    source_sampler.random_components(5)
    source_sampler.set_truncation(100)
    # source_samples = source_sampler.sample(num_samples, seed = 43)
    source_samples = source_sampler.sample(int(num_samples))


    input_measure_sampler = input_sampler(raw_func_list, 
                                          source_samples, 
                                          log = log, 
                                          func_logger = input_func_logger, 
                                          measure_logger = input_measure_logger,
                                          func_log_file_path = input_func_log_file)
    input_measure_sampler.base_function_sample(smoothing)
    input_measure_sampler.measure_sample()

    # shuffle the samples of each measure
    for key in input_measure_sampler.sample_collection:
        np.random.shuffle(input_measure_sampler.sample_collection[key])

    # if plot:
    #     input_measure_sampler.dist_visualize(oneplot = False, save = True, savefile = plot_savefile)

    return source_sampler, input_measure_sampler

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
data_path = "input_samples"
os.makedirs(data_path, exist_ok=True)
source_samples_BA_json = source_samples_BA.tolist()
save_data(source_samples_BA_json, data_path, "source_samples_BA.json")
input_samples_dict_BA_json = {str(k): v.tolist() for k, v in input_samples_dict_BA.items()}
save_data(input_samples_dict_BA_json, data_path, "input_samples_dict_BA.json")

dist_visualize(input_samples_dict_BA, source_samples_BA, oneplot=False, save=True, savefile="measure_visualization.png")





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













    