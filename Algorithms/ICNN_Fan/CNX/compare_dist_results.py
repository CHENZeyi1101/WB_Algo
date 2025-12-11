from __future__ import print_function
import ot
from scipy.stats import entropy
import numpy as np
import scipy.linalg as ln
from numpy import linalg as LA
import torch
import CNX.generate_NN
from optimal_transport_modules.log_utils import ResultsLog
import optimal_transport_modules.generate_data as g_data
import optimal_transport_modules.generate_NN as g_NN
import optimal_transport_modules.data_utils as DTU
import CNX.generate_data


#! generate barycenter data


def barycenter_sampler(cfg, device, results_save_path=None, load_epoch=None):
    if results_save_path is None:
        results_save_path = cfg.get_save_path()

    
    epsilon = CNX.generate_data.torch_normal_gaussian(
        cfg.INPUT_DIM, N_TEST=cfg.N_TEST, device=device)
    generator_h = g_NN.generate_FixedWeight_h_NN(cfg)

    if load_epoch is None:
        generator_h = CNX.generate_NN.load_generator_h(
            results_save_path, generator_h, epochs=cfg.epochs, device=device)
    else:
        generator_h = CNX.generate_NN.load_generator_h(
            results_save_path, generator_h, epochs=load_epoch, device=device)
    miu = generator_h(epsilon)
    return miu