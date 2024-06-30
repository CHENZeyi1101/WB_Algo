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
from scipy.spatial import KDTree

from ADMM import QCQP_ADMM

class OT_Map_Estimator:
    def __init__(self, dim):
        self.dim = dim
        self.estimator_info = {}
        self.SM_warmstart = {}

    def label(self, iter, measure_index):
        self.iter = iter
        self.measure_index = measure_index

    def solve_OptCoupling_matrix(self, BX, BY):
        x, y = BX, BY
        m, n = len(BX), len(BY)
        model = gp.Model("LP_OptCoupling")
        pi = {}
        for i in range(m):
            for j in range(n):
                pi[i, j] = model.addVar(lb=0.0, ub = 1.0, vtype=GRB.CONTINUOUS, name=f"pi_{i}_{j}")
        model.update()
        
        obj = gp.quicksum(pi[i, j] * np.linalg.norm(x[i] - y[j])**2 for i in range(m) for j in range(n))
        model.setObjective(obj, GRB.MINIMIZE)
        # Add constraints: sum(pi_ij) = 1/n for all j
        for j in range(n):
            model.addConstr(gp.quicksum(pi[i, j] for i in range(m)) == 1/n)
        # Add constraints: sum(pi_ij) = 1/m for all i
        for i in range(m):
            model.addConstr(gp.quicksum(pi[i, j] for j in range(n)) == 1/m)
        # Optimize the model
        model.optimize()
        optimal_solution = np.array([[pi[i, j].x for j in range(n)] for i in range(m)])
        optimal_objective = model.objVal
        
        return optimal_solution, optimal_objective
    
    def solve_opt_tuples(self, BX, BY, lambda_lower, lambda_upper, radi, ADMM = False):
        self.BX, self.BY = BX, BY
        self.lambda_lower, self.lambda_upper = lambda_lower, lambda_upper
        pi_star, _ = self.solve_OptCoupling_matrix(BX, BY)
        m, n = len(BX), len(BY)
        dim = self.dim
        
        if not ADMM: # Solve the optimization problem using Gurobi QCQP solver
            model = gp.Model("OptTuple_qcqp")
            model.setParam('NumericFocus', 3)
            tilde_g = {}
            tilde_varphi = {}
            for i in range(m):
                tilde_g[i] = model.addMVar(shape = (dim,), lb=-GRB.INFINITY, name="tilde_g_{}".format(i))
                tilde_varphi[i] = model.addVar(lb=-GRB.INFINITY, name="tilde_varphi_{}".format(i))
            model.update()

            obj_expr = gp.QuadExpr()
            for i in range(m):
                for j in range(n):
                    if pi_star[i][j] > 1e-8:
                        obj_expr += ((BY[j] - tilde_g[i] - lambda_lower * BX[i]) @ (BY[j] - tilde_g[i]- lambda_lower * BX[i])) * pi_star[i][j]
                    else:
                        pass

            model.setObjective(obj_expr, GRB.MINIMIZE)

            for i in range(m):
                aux = tilde_g[i] + lambda_lower * BX[i]
                model.addConstr(aux @ aux <= radi ** 2)
                for j in range(m):
                    if i != j:
                        constraint_expr = gp.QuadExpr()
                        inner_product = tilde_g[i]@(BX[j] - BX[i])
                        norm_squared_tilde_g = (tilde_g[i] - tilde_g[j])@(tilde_g[i] - tilde_g[j])
                        constraint_expr += tilde_varphi[i] - tilde_varphi[j] + inner_product + norm_squared_tilde_g / (2*(lambda_upper - lambda_lower))
                        model.addConstr(constraint_expr <= 0, "constraint_{}_{}".format(i, j))
                    else:
                        pass
            model.optimize()
        
            if model.status == GRB.OPTIMAL:
                print("Optimal solution found")
                tilde_g_star = np.array([[tilde_g[i][j].x for j in range(len(BY[0]))] for i in range(m)])
                tilde_varphi_star = np.array([tilde_varphi[i].x for i in range(m)])
            else:
                print("No optimal solution found")
                rho = 10 # penalty parameter
                ADMM_solver = QCQP_ADMM(BX, BY, rho, lambda_lower, lambda_upper, pi_star, radi)
                _, tilde_varphi_star, tilde_g_star = ADMM_solver.update_vars(presi_threshold=1e-4, dresi_threshold=1e-4)

        else: # Solve the optimization problem using ADMM
            rho = 0.5 # penalty parameter
            ADMM_solver = QCQP_ADMM(BX, BY, rho, lambda_lower, lambda_upper, pi_star, radi)
            _, tilde_varphi_star, tilde_g_star = ADMM_solver.update_vars(presi_threshold=1e-4, dresi_threshold=1e-4)

        tilde_G = tilde_g_star.T
        g_star = tilde_g_star + lambda_lower * BX
        varphi_star = tilde_varphi_star + np.diag(self.BX @ self.BX.T) * self.lambda_lower / 2

        Bv = varphi_star
        + np.diag(self.BX @ self.BX.T) * self.lambda_lower * self.lambda_upper / (2 * (self.lambda_upper - self.lambda_lower))
        + np.diag(g_star @ g_star.T) / (2 * (self.lambda_upper - self.lambda_lower))
        - np.diag(g_star @ self.BX.T) * self.lambda_lower / (self.lambda_upper - self.lambda_lower)

        self.estimator_info[f'Iteration_{self.iter}_Measure_{self.measure_index}'] = {
            'BX': BX, 
            'tilde_g_star': tilde_g_star,
            'tilde_varphi_star': tilde_varphi_star,
            'tilde_G': tilde_G,
            'Bv': Bv,
            'lambda_lower': lambda_lower,
            'lambda_upper': lambda_upper
        }

        store_estimator_info = {
            'BX': BX.tolist(), 
            'tilde_g_star': tilde_g_star.tolist(),
            'tilde_varphi_star': tilde_varphi_star.tolist(),
            'tilde_G': tilde_G.tolist(),
            'Bv': Bv.tolist(),
            'lambda_lower': lambda_lower,
            'lambda_upper': lambda_upper
        }

        # store the estimator_info as json file in folder "records"
        with open(f"test_records_KS_ADMM/estimator_info/estimator_info_{self.iter}_{self.measure_index}.json", "w") as f:
            json.dump(store_estimator_info, f)
        # with open(f"test_records_KS_solver/estimator_info/estimator_info_{self.iter}_{self.measure_index}.json", "w") as f:
        #     json.dump(store_estimator_info, f)

    def interp_QP(self, iter, measure_index, input_vector):
        local_esimator_info = self.estimator_info[f'Iteration_{iter}_Measure_{measure_index}']
        tilde_G, Bv, lambda_lower, lambda_upper = local_esimator_info['tilde_G'], local_esimator_info['Bv'], local_esimator_info['lambda_lower'], local_esimator_info['lambda_upper']
        x = input_vector
        m = len(Bv)

        ######## Solve the embedded maximization problem ########
        model = gp.Model("QP")
        # Define the variables
        BIw = {}
        BIw = model.addMVar(shape = (m,), lb = 0, ub = 1, name = "BIw") # BIw as \R^m-vector
        # Define the objective function
        obj_expr = gp.QuadExpr()
        tilde_BG_x_V = tilde_G.T @ x + Bv
        innerprod = tilde_BG_x_V.T @ BIw
        norm_Gw = (tilde_G @ BIw) @ (tilde_G @ BIw)
        obj_expr += innerprod - (1 / (2 * (lambda_lower - lambda_upper))) * norm_Gw
        model.setObjective(obj_expr, GRB.MAXIMIZE)
        # Define the constraints
        model.addConstr(BIw.sum() == 1)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            optimal_weight = np.array(BIw.X)
            optimal_objective = model.ObjVal
        else:
            print("No optimal solution found")

        eval_value = optimal_objective + norm(x)**2 * lambda_lower / 2
        eval_gradient = tilde_G @ optimal_weight + lambda_lower * x

        return eval_value, eval_gradient
    
    def KS_estimate(self, iter, measure_index, eval_sample, theta = 100, Tau = 100): 
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
            eval_value, eval_gradient = self.interp_QP(iter, measure_index, MC_samples[t])
            MC_value_sum += eval_value
            MC_grad_list += eval_gradient
        print("Time taken for iteration %d: %f" % (t, time.time() - start_time))
        # breakpoint()

        KS_eval_value = MC_value_sum / Tau
        KS_eval_gradient = MC_grad_list / Tau

        return KS_eval_value, KS_eval_gradient
    
    class KDTreeWithInfo:
        def __init__(self):
            self.points_with_info = []

        def add_point(self, point, info):
            self.points_with_info.append((point, info))
            self.rebuild_tree()

        def rebuild_tree(self):
            coordinates = np.array([entry[0] for entry in self.points_with_info])
            self.kd_tree = KDTree(coordinates)

        def query(self, point):
            _, idx = self.kd_tree.query(point)
            return self.points_with_info[idx][0], self.points_with_info[idx][1]  # Return coordinates and associated info
    
    def SM_estimate(self, iter, measure_index, eval_sample, theta = 1000):
        ### SM-smoothing based on Theorem~4.10
        # - Inputs:
        # function_index - index of the convex function to interpolate
        # eval_sample - input sample ( a single vector)
        # theta - smoothing parameter
        # - Outputs:
        # SM_eval_value - the value of the SM-smoothed function at eval_sample
        # SM_eval_gradient - the gradient of the SM-smoothed function at eval_sample
        
        local_esimator_info = self.estimator_info[f'Iteration_{iter}_Measure_{measure_index}']
        # print("local_esimator_info = ", local_esimator_info)
        # breakpoint()
        tilde_G, Bv, lambda_lower, lambda_upper = local_esimator_info['tilde_G'], local_esimator_info['Bv'], local_esimator_info['lambda_lower'], local_esimator_info['lambda_upper']
        x = eval_sample
        # x = eval_sample.ravel()
        m = len(Bv)
        # breakpoint()
        ######## Solve the embedded maximization problem ########
        # if (f'Iteration_{iter}_Measure_{measure_index}' in self.SM_warmstart) == False:
        #     self.SM_warmstart[f'Iteration_{iter}_Measure_{measure_index}'] = self.KDTreeWithInfo()
        #     w = np.ones(m) / m # concatenated vector of w and xi (initialization)
        # else:
        #     # breakpoint()
        #     w = self.SM_warmstart[f'Iteration_{iter}_Measure_{measure_index}'].query(x)[1]
        #     print("w = ", w)
        self.SM_warmstart[f'Iteration_{iter}_Measure_{measure_index}'] = self.KDTreeWithInfo()
        w = np.ones(m) / m

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
                # value = (
                #     -np.dot((tilde_G.T @ x + Bv), w)
                #     + np.linalg.norm(tilde_G @ w) ** 2 / (2 * (lambda_upper - lambda_lower))
                #     + (np.log(m) - np.dot(np.ones(m), np.log(w))) / theta
                # )
                return value
            
            def objective_gradient(self, w):
                x, tilde_G, Bv, lambda_lower, lambda_upper, theta, m = self.x, self.tilde_G, self.Bv, self.lambda_lower, self.lambda_upper, self.theta, self.m
                gradient = (np.log(w) / theta 
                            + np.ones(m) / theta 
                            + tilde_G.T @ tilde_G @ w / (lambda_upper - lambda_lower) 
                            - tilde_G.T @ x - Bv
                )
                # gradient = (- 1 / theta * w  
                #             + tilde_G.T @ tilde_G @ w / (lambda_upper - lambda_lower) 
                #             - tilde_G.T @ x - Bv
                # )
                return gradient
            
            def objective_hessian(self, w):
                x, tilde_G, Bv, lambda_lower, lambda_upper, theta, m = self.x, self.tilde_G, self.Bv, self.lambda_lower, self.lambda_upper, self.theta, self.m
                hessian = (np.diag(1 / w) / theta 
                           + tilde_G.T @ tilde_G / (lambda_upper - lambda_lower)
                )
                # hessian = (np.diag(1 / w ** 2) / theta 
                #            + tilde_G.T @ tilde_G / (lambda_upper - lambda_lower)
                # )
                return hessian
            
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
            
            def solve_KKT_system(self, w):
                m = self.m
                hessian = self.objective_hessian(w)
                top_left = hessian
                top_right = np.ones(m).reshape(-1, 1)
                bottom_left = np.ones(m).reshape(1, -1)
                bottom_right = np.array([[0]])
                # Assemble the full matrix
                top = np.hstack((top_left, top_right))
                bottom = np.hstack((bottom_left, bottom_right))
                KKT_matrix = np.vstack((top, bottom))

                KKT_vector = np.zeros(m + 1)
                gradient = self.objective_gradient(w)

                # # check gradient
                # gradient_check = np.zeros_like(w)
                # h = 1e-5
                # for i in range(len(w)):
                #     w_forward = np.copy(w)
                #     w_backward = np.copy(w)
                #     w_forward[i] += h
                #     w_backward[i] -= h
                #     gradient_check[i] = (self.objective_value(w_forward) - self.objective_value(w_backward)) / (2 * h)
                # breakpoint()

                KKT_vector[:-1] = - gradient
                
                # print("KKT_matrix = ", KKT_matrix)
                # print("KKT_vector = ", KKT_vector)
                # print("w = ", w)
                # breakpoint()

                newton_step = solve(KKT_matrix, KKT_vector)[:-1]
                newton_decrement_sq = newton_step.T @ hessian @ newton_step
                # directional_derivative = (self.objective_value(w + 1e-5 * newton_step) - self.objective_value(w)) / 1e-5
                # breakpoint()
                # breakpoint()
                return newton_step, newton_decrement_sq, gradient
            
            def backtracking_line_search(self, w, newton_step, gradient, alpha = 0.2, beta = 0.5):
                t = 1
                # breakpoint()
                # while np.any(w + t * newton_step <= 1e-8) or self.objective_value(w + t * newton_step) > self.objective_value(w) + alpha * t * gradient.T @ newton_step:
                # print("w = ", w)
                # print("newton_step: ", newton_step)
                while np.any(w + t * newton_step <= 1e-20) or self.objective_value(w + t * newton_step) > self.objective_value(w) + alpha * t * gradient @ newton_step:
                    # print(np.any(w + t * newton_step <= 0), self.objective_value(w + t * newton_step) > self.objective_value(w) + alpha * t * gradient @ newton_step)
                    t = beta * t
                    # print("t = ", t)
                    # breakpoint()
                return t, self.objective_value(w + t * newton_step)
        
        def newton_method(w, x, tilde_G, Bv, lambda_lower, lambda_upper, theta):
            embedded_min = embedded_minimization(x, tilde_G, Bv, lambda_lower, lambda_upper, theta)
            value_list = []
            step_count = 0
           
            while True:
                # newton_step, newton_decrement, gradient = embedded_min.solve_KKT_system(w)
                newton_step, newton_decrement, gradient = embedded_min.solve_KKT_system_woodbury(w)
                # if count % 100 == 0:
                #     print("w = ", w)
                #     print("newton_step = ", newton_step)
                #     print("gradient = ", gradient)
                #     print("newton_decrement = ", newton_decrement)
                    # breakpoint()
                if newton_decrement < 0.1:
                    print("w = ", w)   
                    print("newton_decrement = ", newton_decrement)
                    print(step_count)
                    # breakpoint()
                    break
                # if newton_decrement < 0.5 or np.any(w <= 5e-8):
                #     break
                else:
                    newton_step[np.abs(newton_step) < 1e-10] = 0
                    t, objective_value = embedded_min.backtracking_line_search(w, newton_step, gradient)
                    w = w + t * newton_step
                    value_list.append(objective_value)
                    step_count += 1
            # breakpoint()
            return w, value_list
        
        w_star, _ = newton_method(w, x, tilde_G, Bv, lambda_lower, lambda_upper, theta)
        # print("w = ", w)
        # print("w_star = ", w_star)
        # print("value_list = ", value_list)
        # breakpoint()
        SM_eval_value = (lambda_lower * norm(x) ** 2 / 2 
                         + np.dot((tilde_G.T @ x + Bv), w_star) - norm(tilde_G @ w_star) ** 2 / (2 * (lambda_upper - lambda_lower))
                         - (np.log(m) + np.dot(w_star, np.log(w_star))) / theta
        )
        SM_eval_gradient = tilde_G @ w_star + lambda_lower * x
        self.SM_warmstart[f'Iteration_{iter}_Measure_{measure_index}'].add_point(x, w_star)

        return SM_eval_value, SM_eval_gradient

        


        
            


    
