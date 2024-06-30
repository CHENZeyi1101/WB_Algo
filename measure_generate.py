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
import pickle

class MixtureOfGaussians:
### For generating samples from a mixture of Gaussian distributions (underlying barycenter measure) ###

    def __init__(self, dim, weights=None):
        self.truncation = False
        # Default weights if not provided (equally distributed)
        if weights is None:
            self.weights = []
        else:
            self.weights = weights
            self.weights /= np.sum(self.weights)
        
        # Initialize list to hold parameters for each Gaussian component
        self.gaussians = []
        self.dim = dim

    def add_gaussian(self, mean, cov):
        self.gaussians.append((mean, cov))
        
    def set_weights(self, weights):
        self.weights = weights
        self.weights /= np.sum(self.weights)

    def set_truncation(self, radius):
        self.truncation = True
        self.radius = radius

    def random_components(self, num_components, seed = 42):
        dim = self.dim
        np.random.seed(seed)
        for _ in range(num_components):
            mean = (np.random.rand(dim) - 0.5) * 10
            A = np.random.rand(dim, dim) - 0.5
            cov = np.dot(A, A.T) + np.eye(dim) * 10
            self.add_gaussian(mean, cov)
        weights = np.random.rand(num_components)
        self.set_weights(weights)

    def sample(self, n, seed = None):
        dim = self.dim
        count = 0
        samples = np.zeros((n, dim))
        np.random.seed(seed)
        while count < n:
            choice = np.random.choice(len(self.gaussians), p=self.weights)
            mean, cov = self.gaussians[choice]
            sample = np.random.multivariate_normal(mean, cov)
            if not self.truncation or np.linalg.norm(sample) <= self.radius:
                samples[count] = sample
                count += 1
        return samples

class convex_function:
# For generating different types of convex functions 

    def __init__(self, random_number, x_samples, num_functions = 4):
        self.num_functions = num_functions
        self.choice = random_number
        self.x_samples = x_samples
        self.dim = x_samples.shape[1]

    def CPW_affine(self, seed = 42): # Affine CPW
        num_functions = self.num_functions
        x_samples = self.x_samples
        dim = self.dim
        np.random.seed(seed)  # For reproducibility
        coeff_list = []
        intercept_list = []
        for _ in range(num_functions):
            coeff = (np.random.rand(dim) - 0.5) * 10
            intercept = (np.random.rand() - 0.5) * 10
            coeff_list.append(coeff)
            intercept_list.append(intercept)
        values = np.zeros((num_functions, x_samples.shape[0]))
        gradient = np.zeros((num_functions, x_samples.shape[0], dim))
        for i in range(num_functions):
            values[i, :] = np.dot(x_samples, coeff_list[i]) + intercept_list[i]
            gradient[i, :]= np.repeat(coeff_list[i][np.newaxis, :], x_samples.shape[0], axis=0)
        sample_value = np.max(values, axis=0)
        max_indices = np.argmax(values, axis=0)
        sample_gradient = gradient[max_indices, np.arange(x_samples.shape[0]), :]
        # breakpoint()
        return sample_value, sample_gradient, max_indices
    
    def CPW_quadratic(self, seed = 42): # Quadratic CPW
        x_samples = self.x_samples
        dim = self.dim
        num_functions = self.num_functions
        values = np.zeros((num_functions, x_samples.shape[0]))
        gradient = np.zeros((num_functions, x_samples.shape[0], dim))
        np.random.seed(seed)
        for i in range(num_functions):
            Q = np.random.rand(dim, dim)
            Q = np.dot(Q, Q.T) * 1e-5
            w = (np.random.rand(dim) - 0.5) * 10
            values[i, :] = np.diag(x_samples @ Q @ x_samples.T + np.dot(x_samples, w))
            gradient[i, :] = 2 * x_samples @ Q + np.repeat(w[np.newaxis, :], x_samples.shape[0], axis=0)
        sample_value = np.max(values, axis=0)
        max_indices = np.argmax(values, axis=0)
        sample_gradient = gradient[max_indices, np.arange(x_samples.shape[0]), :]
        return sample_value, sample_gradient, max_indices
    
    def log_sum_exp(self, seed = 42): # Log-sum-exp
        np.random.seed(seed)
        dim = self.dim
        x_samples = self.x_samples
        coeff_matrix = np.random.rand(dim, dim) - 0.5
        intercept_matrix = np.tile((np.random.rand(dim) - 0.5) * 10, (len(x_samples), 1))
        sample_value = np.log(np.sum(np.exp(x_samples @ coeff_matrix + intercept_matrix), axis=1))
        sample_gradient = np.exp(x_samples) @ coeff_matrix / np.sum(np.exp(x_samples @ coeff_matrix + intercept_matrix), axis=1)[:, np.newaxis]
        return sample_value, sample_gradient
    
    def plot_CPW(self, sample_values, max_indices, name = None):
        num_functions = self.num_functions
        x_samples = self.x_samples
        grid_x, grid_y = np.mgrid[0:10:100j, 0:10:100j]
        grid_z = griddata(x_samples, sample_values, (grid_x, grid_y), method='cubic')
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(111, projection='3d')
        colors = cm.rainbow(np.linspace(0, 1, num_functions))
        for i, color in zip(range(num_functions), colors):
            mask = (max_indices == i)
            ax.scatter(x_samples[mask][:, 0], x_samples[mask][:, 1], sample_values[mask], color =color, marker='o', label=f'CPW{i+1}')
        ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.5)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        ax.set_title('CPW with Interpolated Surface')   
        ax.legend()
        plt.savefig(name)

    def plot_log_sum_exp(self, sample_values, name = None):
        x_samples = self.x_samples
        grid_x, grid_y = np.mgrid[0:10:100j, 0:10:100j]
        grid_z = griddata(x_samples, sample_values, (grid_x, grid_y), method='cubic')
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_samples[:, 0], x_samples[:, 1], sample_values, color ='b', marker='o', label='Log-sum-exp')
        ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.5)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        ax.set_title('Log-sum-exp with Interpolated Surface')   
        ax.legend()
        plt.savefig(name)

    def generate_function(self, index, seed = 42):
        if self.choice == 0:
            print("Affine CPW")
            sample_value, sample_gradient, max_indices = self.CPW_affine(seed)
            if self.dim == 2:
                self.plot_CPW(sample_value, max_indices, name = f'function_{index}_CPWA.png')
            return sample_value, sample_gradient
        elif self.choice == 2:
            print("Quadratic CPW")
            sample_value, sample_gradient, max_indices = self.CPW_quadratic(seed)
            if self.dim == 2:
                self.plot_CPW(sample_value, max_indices, name = f'function_{index}_CPWQ.png')
            return sample_value, sample_gradient
        elif self.choice == 1:
            print("Log-sum-exp")
            sample_value, sample_gradient = self.log_sum_exp(seed)
            if self.dim == 2:
                self.plot_log_sum_exp(sample_value, name = f'function_{index}_LSE.png')
            return sample_value, sample_gradient
        

class Input_Measure_Sampling:
### Input measure sampler ### 
### Input measures are defined as the pushforward of the barycenter by a combination of smooth and strongly convex functions ###

    def __init__(self):
        pass

    def generate_cvx_functions(self, dim, num_x = 100, num_measures = 5, seed = 42):
        ## Generate convex functions ###
        # - Inputs: 
        # dim - dimension of the vector space
        # num_x - number of samples for interpolation
        # num_measures - number of input measures (= number of convex functions to generate)
        # - Outputs:
        # function_info - a list of dictionaries containing the information of each convex function

        self.dim = dim
        self.num_measures = num_measures
        np.random.seed(seed)
        self.num_x = num_x
        x_samples = (np.random.rand(num_x, dim) - 0.5) * 100 + np.random.randint(0, 100, (num_x, dim))
        self.function_info = []

        for k in range(num_measures):
            function_profile = {}
            np.random.seed(seed + k)
            l_para = np.random.rand(2) * 2 # 0 < l_down < l_up < 2 to ensure strong convexity and smoothness, since \|x\|^2 is C_{2, 2}
            l_down, l_up = min(l_para), max(l_para)
            U = np.random.randint(0, 1) # all CPWA functions
            Convex = convex_function(U, x_samples, num_functions=5)
            x_values, x_gradients= Convex.generate_function(index = k + 1, seed = seed + k)
            # breakpoint()
            # reparametrize
            # add a linear term
            w = (np.random.rand(dim) - 0.5) * 10
            x_values = x_values + np.dot(x_samples, w)
            x_gradients = x_gradients + np.repeat(w[np.newaxis, :], x_samples.shape[0], axis=0)

            X_interp = x_samples + x_gradients / (l_up - l_down)
            G_interp = x_gradients + l_down * X_interp
            V_interp = x_values
            + np.diag(X_interp @ X_interp.T) * l_down * l_up / (2 * (l_up - l_down))
            + np.diag(G_interp @ G_interp.T) / (2 * (l_up - l_down))
            - np.diag(G_interp @ X_interp.T) * l_down / (l_up - l_down)
            tilde_BG = (G_interp - l_down * X_interp).T # dim: d x m    
            Bv = V_interp + np.diag(X_interp @ X_interp.T) * l_down * l_up / (2 * (l_up - l_down))
            + np.diag(G_interp @ G_interp.T) / (2 * (l_up - l_down))
            - np.diag(G_interp @ X_interp.T) * l_down / (l_up - l_down)

            # store the information as a dictionary
            function_profile['l_down'] = l_down
            function_profile['l_up'] = l_up
            function_profile['X_interp'] = X_interp
            function_profile['G_interp'] = G_interp
            function_profile['V_interp'] = V_interp
            function_profile['tilde_BG'] = tilde_BG
            function_profile['Bv'] = Bv

            self.function_info.append(function_profile)

            function_profile_store = function_profile.copy()
            function_profile_store['X_interp'] = X_interp.tolist()
            function_profile_store['G_interp'] = G_interp.tolist()
            function_profile_store['V_interp'] = V_interp.tolist()
            function_profile_store['tilde_BG'] = tilde_BG.tolist()
            function_profile_store['Bv'] = Bv.tolist()
            with open(f'test_records_KS_ADMM/function_info/function_info_{k}.json', 'w') as file:
                json.dump(function_profile_store, file)
            # with open(f'test_records_KS_solver/function_info/function_info_{k}.json', 'w') as file:
            #     json.dump(function_profile_store, file)

        # with open('function_info.pkl', 'wb') as file:
        #     pickle.dump(self.function_info, file)
        
        # breakpoint()

    def index_permutation(self, seed = 42):
        ### Generate a random permutation of the indices of the convex functions
        ### The functions would be randomly combined based on the premutation

        num_measures = self.num_measures
        origin = np.arange(2 * num_measures)
        permutation = np.zeros(2 * num_measures)
        for i in range(num_measures):
            permutation[2 * i] = origin[i]
            permutation[2 * i + 1] = origin[i + num_measures]
        self.permutation = permutation

    def interp_QP(self, function_index, x): 
        ### Solve the QP for shape-constrained iinterpolation based on Theorem~4.7
        # - Inputs:
        # function_index - index of the convex function to interpolate
        # x - input sample ( a single vector)
        # - Outputs:
        # eval_value - the value of the interpolated function at x
        # eval_gradient - the gradient of the interpolated function at x

        function_profile = self.function_info[function_index]
        l_down, l_up, tilde_BG, Bv = function_profile['l_down'], function_profile['l_up'], function_profile['tilde_BG'], function_profile['Bv']
        num_x = self.num_x
        
        ######## Solve the embedded maximization problem ########
        model = gp.Model("QP")
        # Define the variables
        BIw = {}
        BIw = model.addMVar(shape = (num_x,), lb = 0, ub = 1, name = "BIw") # BIw as \R^m-vector
        # Define the objective function
        obj_expr = gp.QuadExpr()
        tilde_BG_x_V = tilde_BG.T @ x + Bv
        innerprod = tilde_BG_x_V.T @ BIw
        norm_Gw = (tilde_BG @ BIw) @ (tilde_BG @ BIw)
        obj_expr += innerprod - (1 / (2 * (l_up - l_down))) * norm_Gw
        model.setObjective(obj_expr, GRB.MAXIMIZE)
        # Define the constraints
        model.addConstr(BIw.sum() == 1)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            optimal_weight = np.array(BIw.X)
            optimal_objective = model.ObjVal
        else:
            print("No optimal solution found")

        eval_value = optimal_objective + norm(x)**2 * l_down / 2
        eval_gradient = tilde_BG @ optimal_weight + l_down * x

        return eval_value, eval_gradient
    
    def KS_estimate(self, function_index, eval_sample, theta = 100, Tau = 10): 
        ### KS-smoothing based on Theorem~4.8
        # - Inputs:
        # function_index - index of the convex function to interpolate
        # eval_sample - input sample ( a single vector)
        # theta - smoothing parameter
        # Tau - number of Monte Carlo samples
        # - Outputs:
        # KS_eval_value - the value of the KS-smoothed function at eval_sample
        # KS_eval_gradient - the gradient of the KS-smoothed function at eval_sample

        dim = self.dim
        eta = np.random.multivariate_normal(np.zeros(dim), (1 / theta) * np.eye(dim), Tau)
        MC_samples = np.tile(eval_sample, (Tau, 1)) + eta
        MC_value_sum = 0
        MC_grad_list = np.zeros(dim)

        start_time = time.time()
        for t in range(Tau):
            eval_value, eval_gradient = self.interp_QP(function_index, MC_samples[t])
            MC_value_sum += eval_value
            MC_grad_list += eval_gradient
        print("Time taken for iteration %d: %f" % (t, time.time() - start_time))
        # breakpoint()

        KS_eval_value = MC_value_sum / Tau
        KS_eval_gradient = MC_grad_list / Tau

        return KS_eval_value, KS_eval_gradient/ Tau
    
    def SM_estimate(self, function_index, eval_sample, theta = 1000):
        function_profile = self.function_info[function_index]
        l_down, l_up, tilde_BG, Bv = function_profile['l_down'], function_profile['l_up'], function_profile['tilde_BG'], function_profile['Bv']
        num_x = self.num_x
        w = np.ones(num_x) / num_x

        class embedded_minimization:
            def __init__(self, x, tilde_G, Bv, lambda_lower, lambda_upper, theta):
                self.x = x
                self.tilde_G = tilde_G
                self.Bv = Bv
                self.lambda_lower = lambda_lower
                self.lambda_upper = lambda_upper
                self.theta = theta
                self.m = len(Bv)

            def objective_value(self, w):
                x, tilde_G, Bv, lambda_lower, lambda_upper, theta, m = self.x, self.tilde_G, self.Bv, self.lambda_lower, self.lambda_upper, self.theta, self.m
                value = (
                    -np.dot((tilde_G.T @ x + Bv), w)
                    + np.linalg.norm(tilde_G @ w) ** 2 / (2 * (lambda_upper - lambda_lower))
                    + (np.log(m) + np.dot(w, np.log(w))) / theta
                )
                return value
            
            def objective_gradient(self, w):
                x, tilde_G, Bv, lambda_lower, lambda_upper, theta, m = self.x, self.tilde_G, self.Bv, self.lambda_lower, self.lambda_upper, self.theta, self.m
                gradient = (np.log(w) / theta 
                            + np.ones(m) / theta 
                            + tilde_G.T @ tilde_G @ w / (lambda_upper - lambda_lower) 
                            - tilde_G.T @ x - Bv
                )
                return gradient
            
            def solve_KKT_system_woodbury(self, w):
                m = self.m
                tilde_G = self.tilde_G
                lambda_lower, lambda_upper = self.lambda_lower, self.lambda_upper
                theta = self.theta
                d = tilde_G.shape[0]
                gradient = self.objective_gradient(w)
                # hessian = self.objective_hessian(w)

                A_inv = theta * np.diag(w)
                mid_inverse = solve(np.eye(d) + tilde_G @ A_inv @ tilde_G.T / (lambda_upper - lambda_lower), np.eye(d))
                hessian_inv = A_inv - A_inv @ tilde_G.T @ mid_inverse @ tilde_G @ A_inv / (lambda_upper - lambda_lower)
                # print("hessian_inv_check = ", inv(hessian))
                # breakpoint()
                z1 = - hessian_inv @ gradient
                s = - np.ones(m) @ hessian_inv @ np.ones(m)
                z2 = - np.sum(z1) / s
                newton_step = z1 - hessian_inv @ np.ones(m) * z2
                newton_decrement_sq = - gradient @ newton_step
                ## checked
                return newton_step, newton_decrement_sq, gradient
            
            def backtracking_line_search(self, w, newton_step, gradient, alpha = 0.2, beta = 0.5):
                t = 1
                while np.any(w + t * newton_step <= 0) or self.objective_value(w + t * newton_step) > self.objective_value(w) + alpha * t * gradient @ newton_step:
                    t = beta * t
                return t, self.objective_value(w + t * newton_step)
        
        def newton_method(w, x, tilde_G, Bv, lambda_lower, lambda_upper, theta):
            embedded_min = embedded_minimization(x, tilde_G, Bv, lambda_lower, lambda_upper, theta)
            value_list = []
            step_count = 0
            while True:
                
                newton_step, newton_decrement, gradient = embedded_min.solve_KKT_system_woodbury(w)

                # if count % 100 == 0:
                #     print("w = ", w)
                #     print("newton_step = ", newton_step)
                #     print("gradient = ", gradient)
                #     print("newton_decrement = ", newton_decrement)
                #     breakpoint()
                
                if newton_decrement < 0.1:
                    print(f"step_count = {step_count}")
                    break
                else:
                    newton_step[np.abs(newton_step) < 1e-10] = 0
                    t, objective_value = embedded_min.backtracking_line_search(w, newton_step, gradient)
                    w = w + t * newton_step
                    value_list.append(objective_value)
                    step_count += 1
            # breakpoint()
            return w, value_list
        
        w_star, _ = newton_method(w, eval_sample, tilde_BG, Bv, l_down, l_up, theta)
        SM_eval_value = (l_down * norm(eval_sample) ** 2 / 2 
                         + np.dot((tilde_BG.T @ eval_sample + Bv), w_star) - norm(tilde_BG @ w_star) ** 2 / (2 * (l_up - l_down))
                         - (np.log(num_x) + np.dot(w_star, np.log(w_star))) / theta
        )
        SM_eval_gradient = tilde_BG @ w_star + l_down * eval_sample

        return SM_eval_value, SM_eval_gradient
    
    def compute_mapping(self, base_samples, smoothing = 'KS'):
        ### Compute the mapping of the base samples under the convex functions
        # - Inputs:
        # base_samples - the samples to be mapped
        # - Outputs:
        # dict_base_samples - a dictionary containing the evaluated values and gradients of the base samples under each convex function

        dim = self.dim
        function_info = self.function_info 
        dict_base_samples = {} 
        for function_index in range(len(function_info)):
            dict_base_samples[f"function_{function_index}"] = {}
            eval_values = np.zeros(len(base_samples))
            eval_gradients = np.zeros((len(base_samples), dim))
            for i in range(len(base_samples)):
                if smoothing == 'KS':
                    eval_value, eval_gradient = self.KS_estimate(function_index, base_samples[i])
                if smoothing == 'SM':
                    eval_value, eval_gradient = self.SM_estimate(function_index, base_samples[i])
                # breakpoint()
                print(f"Function {function_index}, Sample {i}: Value = {eval_value}, Gradient = {eval_gradient}")
                # breakpoint()
                eval_values[i] = eval_value
                eval_gradients[i] = eval_gradient
            dict_base_samples[f"function_{function_index}"]['eval_values'] = eval_values
            dict_base_samples[f"function_{function_index}"]['eval_gradients'] = eval_gradients
            print(f"Function {function_index} finished")
            # breakpoint()

        return dict_base_samples
    
    def measure_sampling(self, base_samples, smoothing = 'KS'):
        ### Generate samples from the input measures (based on the permutation of the convex functions)
        # - Inputs:
        # base_samples - the samples to be mapped
        # - Outputs:
        # dict_measure_samples - a dictionary containing the generated samples from the input measures
        permutation = self.permutation
        dict_base_samples = self.compute_mapping(base_samples, smoothing)
        num_measures = self.num_measures
        dict_measure_samples = {}
        for measure_index in range(num_measures):
            measure_samples = np.zeros((len(base_samples), self.dim))
            active_indices = permutation[2 * measure_index: 2 * measure_index + 2]

            if active_indices[0] % 2 == 0: # corresponds to f(x)
                part1 = dict_base_samples[f"function_{int(active_indices[0] / 2)}"]['eval_gradients']
            else: # corresponds to \|x\|^2 - f(x)
                part1 = 2 * base_samples - dict_base_samples[f"function_{int((active_indices[0] - 1) / 2)}"]['eval_gradients']
            if active_indices[1] % 2 == 0:
                part2 = dict_base_samples[f"function_{int(active_indices[1] / 2)}"]['eval_gradients']
            else:
                part2 = 2 * base_samples - dict_base_samples[f"function_{int((active_indices[1] - 1) / 2)}"]['eval_gradients']

            measure_samples = (part1 + part2) / 2
            dict_measure_samples[f"measure_{measure_index}"] = measure_samples

        return dict_measure_samples
    
class Gaussian_Measure:
    def __init__(self, dim, num_measures, distribution_type = None, parameters = None):
        self.dim = dim
        self.type = distribution_type
        self.parameters = parameters
        self.num_measures = num_measures

    def generate_random_parameters(self, num_measures, seed = 41):
        dim = self.dim
        np.random.seed(seed)
        parameters = []
        for _ in range(num_measures):
            mean = np.random.rand(dim) * 10
            A = np.random.rand(dim, dim)
            A = np.dot(A, A.T) * 10 + np.eye(dim) # Ensure covariance matrix is positive definite
            parameters.append((mean, A))
        return parameters

    def generate_gaussian_sample(self, size, k, seed = None):
        np.random.seed(seed)
        if self.type == "gaussian":
            mean, covariance_matrix = self.parameters[k]
            return np.random.multivariate_normal(mean, covariance_matrix, size)
        else:
            raise ValueError("Unsupported distribution type")
        
    def generate_mixture_gaussian_sample(self, size, seed = None):
        np.random.seed(seed)
        if self.type == "mixture_gaussian":
            gaussians = self.parameters
            n_mix = len(gaussians)
            U = np.random.rand(size)
            samples = []
            for u in U:
                group = math.floor(u * n_mix)
                mean, covariance_matrix = gaussians[group]
                sample = np.random.multivariate_normal(mean, covariance_matrix)
                samples.append(sample)
            return np.array(samples)
        else:
            raise ValueError("Unsupported distribution type")
        
    def generate_truncated_sample(self, size, R = 1, A = None, b = None, seed = None):
        dim = self.dim
        if self.type == "gaussian":
            accepted = []
            sample = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), 10 * size)
            # breakpoint()
            index = 0
            while len(accepted) < size:
                if norm(sample[index]) < R:
                    accepted.append(A @ sample[index] + b)
                index += 1
            return np.squeeze(np.array(accepted))

        elif self.type == "mixture_gaussian": # not location scatter
            accepted = []
            
            while len(accepted) < size:
                sample = self.generate_mixture_gaussian_sample(10 * size, seed)
                index = 0
                while len(accepted) < size and index < 10 * size:
                    if norm(sample[index]) < R:
                        accepted.append(sample[index])
                    index += 1
            return np.squeeze(np.array(accepted))
        
        else:
            raise ValueError("Unsupported distribution type")

    def measure_sampling(self, num_samples):
        num_measures = self.num_measures
        nu_para = self.generate_random_parameters(num_measures, seed = 41)
        print(nu_para)
        # breakpoint()
        dict_measure_samples = {}
        for i in range(num_measures):
            b, A = nu_para[i][0], nu_para[i][1]
            dict_measure_samples[f"measure_{i}"] = self.generate_truncated_sample(size = num_samples, R = 1, A = A, b = b) 
        return dict_measure_samples
    
# mixture = MixtureOfGaussians()
# dim = 2
# # Add Gaussian components to the mixture
# num_mixture = 4
# for k in range(num_mixture):
#     np.random.seed(10 * k)
#     mean = np.random.rand(dim) - 0.5 
#     A = np.random.rand(dim, dim) - 0.5
#     cov = np.dot(A, A.T) + np.eye(dim) # Ensure covariance matrix is positive definite
#     mixture.add_gaussian(mean, cov)
# # Set custom weights
# mixture.set_weights(np.random.rand(num_mixture))
# mixture.set_truncation(2) # Set truncation radius
# # Generate samples from the mixture
# num_mixture_samples = 1000
# mixture_samples = mixture.sample(num_mixture_samples)
# print("Generated samples from the mixture of multivariate Gaussian distributions:")
# print(mixture_samples)

# test = Input_Measure_Sampling()
# num_x, num_measures = 100, 5
# test.generate_cvx_functions(dim=2, num_x=100, num_measures=5)
# test.index_permutation()
# base_samples = mixture_samples
# dict_measure_samples = test.measure_sampling(base_samples)
# # breakpoint()
# fig, axes = plt.subplots(1, num_measures + 1, figsize=(15, 5)) 
# ax = axes[0]
# ax.scatter(base_samples[:, 0], base_samples[:, 1], color ='b', marker ='o', label='Base Sample', alpha=0.6)
# ax.set_title('Base Sample')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')

# for k in range(num_measures):
#     measure_samples = dict_measure_samples[f"measure_{k}"]
#     ax = axes[k + 1]
#     ax.scatter(measure_samples[:, 0], measure_samples[:, 1], color ='r', marker ='o', label=f'Measure {k}', alpha=0.6)
#     ax.set_title(f'Measure {k+1}')
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
# plt.tight_layout()
# plt.savefig('measure_samples.png')
# plt.show()



    


