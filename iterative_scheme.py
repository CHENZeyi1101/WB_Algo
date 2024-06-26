import numpy as np
import gurobipy as gp
from gurobipy import GRB
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import psutil
import math
import time
import scipy as sp
from scipy.linalg import sqrtm, pinv, norm, inv, solve
import pdb
from scipy.interpolate import griddata

from measure_generate import *
from OT_estimator import *

class Iterative_Scheme:
    def __init__(self, barycenter, OT_estimator, input_measure_sampler, num_samples):
        self.barycenter = barycenter # python class object
        self.OT_Map_Estimator = OT_estimator # python class object
        self.Input_Measure_Sampler = input_measure_sampler # python class object
        self.num_samples = num_samples
        
    def generate_samples(self, iter, smoothing = 'KS', truncate_radius = 10):
        num_samples = self.num_samples
        dim = self.barycenter.dim
        count = 0
        accepted = np.zeros((num_samples, self.barycenter.dim))
        num_measures = self.Input_Measure_Sampler.num_measures
        self.barycenter.set_truncation(radius = 10)

        while count < num_samples:
            sample = self.barycenter.sample(1, seed = iter * num_samples + count)
            for t in range(iter):
                # breakpoint()
                sum_sample = np.zeros(dim)
                for measure_index in range(num_measures):
                    print(f"iter: {iter}")
                    print(f"Measure index: {measure_index}")
                    # breakpoint()
                    if smoothing == 'KS':
                        _, sub_sample = self.OT_Map_Estimator.KS_estimate(t, measure_index, sample)
                    if smoothing == 'SM':
                        _, sub_sample = self.OT_Map_Estimator.SM_estimate(t, measure_index, sample)

                    sum_sample += sub_sample
                sample = sum_sample / num_measures
            if np.linalg.norm(sample) < truncate_radius:
                accepted[count] = sample
                count += 1
        return accepted
    
    def W2_square(self, BX, BY):
        W2_square = self.OT_Map_Estimator.solve_OptCoupling_matrix(BX, BY)[1]
        return W2_square
    
    def construct_mapping(self, iter, BX, lambda_lower = 0.1, lambda_upper = 100, radi = 10, ADMM = False):
        num_measures = self.Input_Measure_Sampler.num_measures
        num_samples = self.num_samples
        base_samples = self.barycenter.sample(num_samples, seed = 42 + iter)
        input_measure_samples = self.Input_Measure_Sampler.measure_sampling(base_samples) # return a dictionary
        V_value = 0
        for measure_index in range(num_measures):
            self.OT_Map_Estimator.label(iter, measure_index)
            BY = input_measure_samples[f"measure_{measure_index}"]
            self.OT_Map_Estimator.solve_opt_tuples(BX, BY, lambda_lower, lambda_upper, radi, ADMM)
            print(self.OT_Map_Estimator.estimator_info[f'Iteration_{iter}_Measure_{measure_index}']['tilde_g_star'])
            breakpoint()
            # self.OT_Map_Estimator.solve_opt_tuples(BX, BY, lambda_lower, lambda_upper, radi, ADMM = True)
            # print(self.OT_Map_Estimator.estimator_info[f'Iteration_{iter}_Measure_{measure_index}']['tilde_g_star'])
            # breakpoint()
            V_value += self.W2_square(BX, BY) / num_measures

        return V_value

    def convergence(self, max_diff = 1e-3, max_iter = 5, smoothing = 'KS'):
        iter = 0
        V_list = [math.inf]
        difference = math.inf
        while difference > max_diff and iter < max_iter:
            BX = self.generate_samples(iter, smoothing = smoothing)
            breakpoint()
            V_value = self.construct_mapping(iter, BX, ADMM = False)
            difference = abs(V_value - V_list[-1])
            V_list.append(V_value)
            iter += 1
            breakpoint()
        print(V_list)
        breakpoint()
        return BX
    
dim = 2
barycenter = MixtureOfGaussians(dim)
barycenter.random_components(4, seed = 42)

OT_estimator = OT_Map_Estimator(dim)
input_measure_sampler = Input_Measure_Sampling()
input_measure_sampler.generate_cvx_functions(dim=2, num_x=50, num_measures=3)
input_measure_sampler.index_permutation()

num_samples = 50
# breakpoint()
iter_scheme = Iterative_Scheme(barycenter, OT_estimator, input_measure_sampler, num_samples)
approx_barycenter = iter_scheme.convergence(smoothing = 'SM')
print(approx_barycenter)
breakpoint()
bary_sample = barycenter.sample(10)
breakpoint()


        

    
