import numpy as np
import gurobipy as gp
from gurobipy import GRB
import logging

from true_WB import *
from input_generate import *
from estimate_OT import *
from ADMM import *
from iterative_scheme import *
from config_log import *

def save_data(data, pathname = None, filename = None):
    output_file = os.path.join(pathname, filename)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4) 

def read_data(pathname = None, filename = None):
    file_path = os.path.join(pathname, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# current_path = os.getcwd()

# Set up logging
def input_logger(log, smoothing):
    if log:
        input_log_dir = "input_logs"
        os.makedirs(input_log_dir, exist_ok=True)
        if smoothing == "BA":
            input_func_logger, input_func_log_file = configure_logging('input_function_BA', input_log_dir, 'input_function_BA.log')
            input_measure_logger, input_measure_log_file = configure_logging('input_measure_BA', input_log_dir, 'input_measure_BA.log')
        elif smoothing == "KS":
            input_func_logger, input_func_log_file = configure_logging('input_function_KS', input_log_dir, 'input_function_KS.log')
            input_measure_logger, input_measure_log_file = configure_logging('input_measure_KS', input_log_dir, 'input_measure_KS.log')
    else:
        input_func_logger = None
        input_measure_logger = None

    return input_func_logger, input_measure_logger, input_func_log_file, input_measure_log_file

def feed_samples(num_measures, 
                 num_samples,
                 dim, 
                 log = False, 
                 smoothing = "BA", 
                 input_func_logger = None, 
                 input_measure_logger = None, 
                 input_func_log_file = None,
                 plot = False,
                 plot_savefile = None):

    raw_func_list = []
    for i in range(num_measures):
        x_samples = np.random.uniform(low = -100, high = 100, size=(5, dim))

        # log the sample points for generating the convex function
        if log:
            input_func_logger.info(f"Sample points for CvxFunction_{i}: {x_samples}")

        cvxfunc_generator = convex_function(x_samples, num_functions = num_measures, log = log, logger = input_func_logger)

        x_values, x_gradients, _ = cvxfunc_generator.generate_quadratic_sqrt(seed = 250 + i) # 100, 200
        cvx_otmap_generator = cvx_based_OTmap(x_samples, x_values, x_gradients, log = log)

        # initialize parameters of cvx_otmap_generator
        input_func_logger.info(f"####### Shape and Interpolation Parameters for CvxFunction_{i} #######")
        cvx_otmap_generator.shape_paras(seed = 5 + i, logger = input_func_logger) #4, 5
        cvx_otmap_generator.interp_paras(logger = input_func_logger)
        raw_func_list.append(cvx_otmap_generator)

    if log:
        input_func_logger.info("######### Finished generating raw functions #########")

    source_sampler = MixtureOfGaussians(dim)
    source_sampler.random_components(5)
    source_sampler.set_truncation(100)
    source_samples = source_sampler.sample(num_samples, seed = 43)

    input_measure_sampler = input_sampler(raw_func_list, 
                                          source_samples, 
                                          log = log, 
                                          func_logger = input_func_logger, 
                                          measure_logger = input_measure_logger,
                                          func_log_file_path = input_func_log_file)
    input_measure_sampler.base_function_sample(smoothing)
    input_measure_sampler.measure_sample()

    if plot:
        input_measure_sampler.dist_visualize(oneplot = True, save = True, savefile = plot_savefile)

    return source_sampler, input_measure_sampler

dim = 2
num_measures = 6
log = True

##################### BA generate samples #####################

input_func_logger_BA, input_measure_logger_BA, input_func_log_file_BA, input_measure_log_file_BA = input_logger(log, "BA")
source_sampler_BA, input_measure_sampler_BA = feed_samples(num_measures,
                                                    num_samples = 10000,
                                                    dim = dim,
                                                    log = log,
                                                    smoothing = "BA",
                                                    input_func_logger = input_func_logger_BA,
                                                    input_measure_logger = input_measure_logger_BA,
                                                    input_func_log_file = input_func_log_file_BA,
                                                    plot = True,
                                                    plot_savefile = "input_measures.png"
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













    