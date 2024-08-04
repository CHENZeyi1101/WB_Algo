import os
import json

from .config_log import *

# Read and save data as json files
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