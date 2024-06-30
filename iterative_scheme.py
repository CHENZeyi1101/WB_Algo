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
        
    def generate_samples(self, iter, smoothing = 'KS', truncate_radius = 100):
        num_samples = self.num_samples
        dim = self.barycenter.dim
        count = 0
        accepted = np.zeros((num_samples, dim))
        num_measures = self.Input_Measure_Sampler.num_measures
        start_time = time.time()
        while count < num_samples:
            mean = np.zeros(dim) + 10
            cov = np.eye(dim) * 10
            truncate_radius = norm(mean) * 5
            sample = np.random.multivariate_normal(mean, cov)
            # sample = self.barycenter.sample(1, seed = iter * num_samples + count)
            # breakpoint()
            
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
                print(f"count: {count}")
                # breakpoint()
        print("time: ", time.time() - start_time)
        # breakpoint()
        return accepted
    
    def W2_square(self, BX, BY):
        W2_square = self.OT_Map_Estimator.solve_OptCoupling_matrix(BX, BY)[1]
        return W2_square
    
    def construct_mapping(self, iter, BX, lambda_lower = 0.01, lambda_upper = 1000, radi = 1000, ADMM = False):
        num_measures = self.Input_Measure_Sampler.num_measures
        num_samples = self.num_samples

        base_samples = self.barycenter.sample(num_samples, seed = 100 + iter)
       
        # input_measure_samples = self.Input_Measure_Sampler.measure_sampling(num_samples)
        input_measure_samples = self.Input_Measure_Sampler.measure_sampling(base_samples, smoothing = 'KS')
         # return a dictionary

        V_value = 0
        for measure_index in range(num_measures):
            self.OT_Map_Estimator.label(iter, measure_index)
            BY = input_measure_samples[f"measure_{measure_index}"]
            print("BY:, ", BY)
            # breakpoint()
            # radi = np.max(norm(BY, axis = 1)) * 10
            self.OT_Map_Estimator.solve_opt_tuples(BX, BY, lambda_lower, lambda_upper, radi, ADMM)
            print(self.OT_Map_Estimator.estimator_info[f'Iteration_{iter}_Measure_{measure_index}']['tilde_g_star'])
            # breakpoint()
            V_value += self.W2_square(BX, BY) / num_measures

        return V_value

    def convergence(self, max_diff = 1e-3, max_iter = 20, smoothing = 'KS', ADMM = False):
        iter = 0
        V_list = [math.inf]
        difference = math.inf
        while difference > max_diff and iter < max_iter:
            BX = self.generate_samples(iter, smoothing = smoothing)
            V_value= self.construct_mapping(iter, BX, ADMM = ADMM)
            difference = abs(V_value - V_list[-1])
            V_list.append(V_value)
            iter += 1
            print(f"iteration: {iter}")
            print(f"V_value: {V_value}")
            print(f"BX: {BX}")
            # store V_list as json file in folder "records"
            with open(f"test_records_KS_ADMM/V_list_{smoothing}_{iter}.json", "w") as f:
                json.dump(V_list, f)
            # with open(f"test_records_KS_solver/V_list_{smoothing}_{iter}.json", "w") as f:
            #     json.dump(V_list, f)
            # store BX as json file in folder "records"
            with open(f"test_records_KS_ADMM/BX_{smoothing}_{iter}.json", "w") as f:
                json.dump(BX.tolist(), f)
            # with open(f"test_records_KS_solver/BX_{smoothing}_{iter}.json", "w") as f:
            #     json.dump(BX.tolist(), f)
            # breakpoint()
        print(V_list)
        # breakpoint()
        return BX
    
dim = 2
num_measures = 3
barycenter = MixtureOfGaussians(dim)
barycenter.random_components(1, seed = 42)
barycenter.set_truncation(radius = 100)
print(barycenter.gaussians)
# print(barycenter.sample(10))
breakpoint()

OT_estimator = OT_Map_Estimator(dim)
# Gaussian
# input_measure_sampler = Gaussian_Measure(dim, num_measures, 'gaussian')

# Barycenter Generate
input_measure_sampler = Input_Measure_Sampling()
input_measure_sampler.generate_cvx_functions(dim, num_x=100, num_measures=num_measures)
input_measure_sampler.index_permutation()

# print(input_measure_sampler.permutation)
# breakpoint()
num_samples = 70
# breakpoint()
iter_scheme = Iterative_Scheme(barycenter, OT_estimator, input_measure_sampler, num_samples)

start_time = time.time()
base_samples = barycenter.sample(num_samples, seed = 10)
input_measure_samples = input_measure_sampler.measure_sampling(base_samples, smoothing = 'KS')
# input_measure_samples = input_measure_sampler.measure_sampling(num_samples)

V_bary_list = []
bary_sample_list = []
for t in range(20):
    V_bary = 0
    bary_samples = barycenter.sample(num_samples, seed = 20 + t)
    for measure_index in range(num_measures):
        BY = input_measure_samples[f"measure_{measure_index}"]
        print("BY:, ", BY)
        print("bary_samples: ", bary_samples)
        # breakpoint()
        V_bary += iter_scheme.W2_square(bary_samples, BY) / num_measures
    V_bary_list.append(V_bary)
    bary_sample_list.append(bary_samples.tolist())

# breakpoint()

# store V_bary as json file
with open("test_records_KS_ADMM/V_bary.json", "w") as f:
    json.dump(V_bary_list, f)
# with open("test_records_KS_solver/V_bary.json", "w") as f:
#     json.dump(V_bary_list, f)
# store bary_samples as json file  
with open("test_records_KS_ADMM/bary_samples.json", "w") as f:
    json.dump(bary_samples.tolist(), f)
# with open("test_records_KS_solver/bary_samples.json", "w") as f:
#     json.dump(bary_samples.tolist(), f)
print("saved")
breakpoint()

approx_barycenter1 = iter_scheme.convergence(smoothing = 'SM', ADMM = False)

# store both approx_barycenter as json file
with open("test_records_KS_ADMM/approx_barycenter1.json", "w") as f:
    json.dump(approx_barycenter1.tolist(), f)
with open("test_records_KS_solver/approx_barycenter1.json", "w") as f:
    json.dump(approx_barycenter1.tolist(), f)
# with open("approx_barycenter2.json", "w") as f:
#     json.dump(approx_barycenter2.tolist(), f)
# print(approx_barycenter)


        

    
