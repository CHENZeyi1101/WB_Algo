import os
import torch.utils.data
import torch
from math import isclose
import numpy as np
from scipy.stats import multivariate_normal
from torchvision import datasets, transforms
import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.data_utils as DTU
import jacinle.io as io

#! gaussian

def torch_normal_gaussian(INPUT_DIM, **kwargs):
    N_TEST = kwargs.get('N_TEST')
    device = kwargs.get('device')
    kernel_size = kwargs.get('kernel_size')
    if N_TEST is None:
        epsilon_test = torch.randn(INPUT_DIM)
    elif kernel_size is None:
        epsilon_test = torch.randn(N_TEST, INPUT_DIM)
    else:
        epsilon_test = torch.randn(N_TEST, INPUT_DIM, kernel_size, kernel_size)
    # return epsilon_test.cuda(device)
    return epsilon_test.cpu()


def torch_samples_generate_Gaussian(n, mean, cov, **kwargs):
    device = kwargs.get('device')
    # return torch.from_numpy(
    #     np.random.multivariate_normal(mean, cov, n)).float().cuda(device)
    return torch.from_numpy(
            np.random.multivariate_normal(mean, cov, n)).float().cpu()
# * numpy type


def repeat_list(ndarray, repeat_times):
    return [ndarray] * repeat_times


def np_samples_generate_Gaussian(mean, cov, n):
    Gaussian_sample = np.random.multivariate_normal(mean, cov, n)
    return Gaussian_sample


def np_PDF_generate_multi_normal_NN_1(pos_n_n_2, mean, cov):
    rv = multivariate_normal(mean, cov)
    multi_normal_nn_1 = rv.pdf(pos_n_n_2)
    return multi_normal_nn_1


def np_PDF_generate_multi_normal_N_N(pos_n_n_2, mean, cov):
    multi_normal_n_n = np_PDF_generate_multi_normal_NN_1(
        pos_n_n_2, mean, cov).reshape(-1, 1)[:, 0]
    return multi_normal_n_n


def np_generate_kde_NN_1(pos_nn_2, kde_analyzer):
    kde_nn_1 = kde_analyzer.score_samples(pos_nn_2)
    kde_nn_1 = np.exp(kde_nn_1)
    return kde_nn_1
