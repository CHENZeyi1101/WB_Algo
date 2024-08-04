import numpy as np
import gurobipy as gp
from gurobipy import GRB
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import psutil
import math
import time
import scipy as sp
from scipy.linalg import sqrtm, pinv, norm, inv, solve
import pdb

from .config_log import *

class QCQP_ADMM:
    def __init__(self, X, Y, rho, lambda_lower, lambda_upper, pi, radi, logger = None, log_file_path = None):
        self.X = X # Sample set X (domain)
        self.Y = Y # Sample set Y (range)
        self.rho = rho # Penalty parameter
        self.lambda_lower = lambda_lower
        self.lambda_upper = lambda_upper
        self.pi = pi # optimal coupling matrix
        # self.max_iter = max_iter # Maximum iteration number

        self.m = X.shape[0] # Number of samples
        self.n = Y.shape[0]
        self.d = X.shape[1] # Dimension of samples
        self.radi = radi # Radius of the ball

        ############ Initialize variables ############
        self.varphi_e_source = np.zeros((self.m, self.m)) # edge source value-variables
        self.varphi_e_target = np.zeros((self.m, self.m)) # edge target value-variables
        self.g_e_source = np.zeros((self.m, self.m, self.d)) # edge source gradient-variables
        self.g_e_target = np.zeros((self.m, self.m, self.d)) # edge target gradient-variables
        self.varphi = np.zeros((self.m)) # node value-variables
        self.g = np.zeros((self.m, self.d)) # node gradient-variables
        self.u_source = np.zeros((self.m, self.m)) # edge source dual value-variables (scaled)
        self.u_target = np.zeros((self.m, self.m)) # edge target dual value-variables (scaled)
        self.v_source = np.zeros((self.m, self.m, self.d)) # edge source dual gradient-variables (scaled)
        self.v_target = np.zeros((self.m, self.m, self.d)) # edge target dual gradient-variables (scaled)
        self.presi =  np.inf # primal residual
        self.dresi = np.inf # dual residual

        self.I = np.identity(2 * self.d + 2)
        block1 = np.zeros((self.d + 1, self.d + 1))
        np.fill_diagonal(block1[1:, 1:], 1)
        block2 = np.zeros((self.d + 1, self.d + 1))
        np.fill_diagonal(block2[1:, 1:], -1)
        self.Q = 1/(2 * (self.lambda_upper - self.lambda_lower)) * np.block([[block1, block2], [block2, block1]])


        self.logger = logger
        self.log_file_path = log_file_path
        self.eig_value, self.U = np.linalg.eig(self.Q)
        # self.D = np.diag(self.eig_value)

    def objective(self):
        # define the objective function of the shape-constrained QCQP
        obj = 0
        for i in range(self.m):
            for j in range(self.n):
                obj += self.pi[i, j] * np.linalg.norm(self.g[i] + self.lambda_lower * self. X[i] - self.Y[j]) ** 2
        return obj

    def local_QCQP(self, source_index, target_index):
        logger, log_file_path = self.logger, self.log_file_path
        # Solve the local QCQP using the solver
        d = self.d
        X = self.X
        varphi, g = self.varphi, self.g
        u_source = self.u_source
        u_target = self.u_target
        v_source = self.v_source
        v_target = self.v_target
        I, Q = self.I, self.Q
        i, j = source_index, target_index
        varphi_i, varphi_j = varphi[i], varphi[j]
        g_i, g_j = g[i], g[j]
        
        eta_i, eta_j = np.insert(g_i, 0, varphi_i), np.insert(g_j, 0, varphi_j)  # construct eta vector
        u_ij_source, u_ji_target = u_source[i, j], u_target[j, i]
        v_ij_source, v_ji_target = v_source[i, j, :], v_target[j, i, :]
        vartheta_ij_source, vartheta_ji_target = np.insert(v_ij_source, 0, u_ij_source), np.insert(v_ji_target, 0, u_ji_target)  # construct z vector for scaled duals
        # breakpoint()
        q0 = -2 * np.hstack((eta_i - vartheta_ij_source, eta_j - vartheta_ji_target))
        q1 = np.hstack((np.insert(X[j] - X[i], 0, 1), np.insert(np.zeros(d), 0, -1)))

        try:
            model = gp.Model("Edge_QCQP")
            model.setParam('NumericFocus', 2)
            if logger:
                model.setParam('LogFile', log_file_path)

            xi = model.addMVar(shape = (2 * d + 2,), lb=-GRB.INFINITY, name="xi")

            # Define the objective function
            obj_expr = gp.QuadExpr()
            obj_expr += xi @ I @ xi + q0 @ xi
            model.setObjective(obj_expr, GRB.MINIMIZE)

            # Add constraints
            model.addConstr(xi @ Q @ xi + q1 @ xi <= 0, "constraint_1")

            model.optimize()
            # breakpoint()
            if model.status == GRB.OPTIMAL:
                print("Optimal solution found")
                if logger:
                    logger.info(f"\n"
                                f"Optimal solution found for the local QCQP\n"
                                f"Source index: {i}, Target index: {j}\n"
                                f"Objective value: {model.objVal}\n"
                                # f"Optimal solution: {xi.x}\n"
                                )
            else: 
                print("No optimal solution found")
                # breakpoint()

            xi_star = np.array(xi.x)
        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            pdb.set_trace()
        return xi_star
    
    def phi_info(self, v, q0, q1):
        d = self.d
        Q = self.Q
        eig_value, U = self.eig_value, self.U
        # D = np.diag(eig_value)
        U_tsp_q0 = U.T @ q0
        U_tsp_q1 = U.T @ q1

        if v != 0:
            # K = np.diag(1 / (np.ones(2 * d + 2) / v + eig_value))
            # K2 = np.diag(1 / (np.ones(2 * d + 2) / v + eig_value)**2)
            # K3 = np.diag(1 / (np.ones(2 * d + 2) / v + eig_value)**3)
            # D2 = np.diag(eig_value**2)
            K_vec = 1 / (np.ones(2 * d + 2) / v + eig_value)
            K2_vec = 1 / (np.ones(2 * d + 2) / v + eig_value)**2
            K3_vec = 1 / (np.ones(2 * d + 2) / v + eig_value)**3
            D2_vec = eig_value**2

            q0_inv_q0 = (1/v) * np.dot(K_vec, U_tsp_q0 ** 2)
            q0_inv_q1 = (1/v) * np.dot(K_vec, U_tsp_q0 * U_tsp_q1)
            q1_inv_q1 = (1/v) * np.dot(K_vec, U_tsp_q1 ** 2)
            q0_inv_Q_inv_q0 = (1/v**2) * np.dot(K2_vec * eig_value, U_tsp_q0 ** 2)
            q0_inv_Q_inv_q1 = (1/v**2) * np.dot(K2_vec * eig_value, U_tsp_q0 * U_tsp_q1)
            q1_inv_Q_inv_q1 = (1/v**2) * np.dot(K2_vec * eig_value, U_tsp_q1 ** 2)
            q0_inv_Q_inv_Q_inv_q0 = (1/v**3) * np.dot(K3_vec * D2_vec, U_tsp_q0 ** 2)
            q0_inv_Q_inv_Q_inv_q1 = (1/v**3) * np.dot(K3_vec * D2_vec, U_tsp_q0 * U_tsp_q1)
            q1_inv_Q_inv_Q_inv_q1 = (1/v**3) * np.dot(K3_vec * D2_vec, U_tsp_q1 ** 2)
        
            # q0_inv_q0 = (1/v) * q0.T @ U @ K @ U.T @ q0
            # q0_inv_q1 = (1/v) * q0.T @ U @ K @ U.T @ q1
            # q1_inv_q1 = (1/v) * q1.T @ U @ K @ U.T @ q1
            # q0_inv_Q_inv_q0 = (1 / v**2) * q0.T @ U @ K2 @ D @ U.T @ q0
            # q0_inv_Q_inv_q1 = (1 / v**2) * q0.T @ U @ K2 @ D @ U.T @ q1
            # q1_inv_Q_inv_q1 = (1 / v**2) * q1.T @ U @ K2 @ D @ U.T @ q1
            # q0_inv_Q_inv_Q_inv_q0 = (1 / v**3) * q0.T @ U @ K3 @ D2 @ U.T @ q0
            # q0_inv_Q_inv_Q_inv_q1 = (1 / v**3) * q0.T @ U @ K3 @ D2 @ U.T @ q1
            # q1_inv_Q_inv_Q_inv_q1 = (1 / v**3) * q1.T @ U @ K3 @ D2 @ U.T @ q1

            # phi = -0.25 * (q0_inv_q0 + 2 * v * q0_inv_q1 + v**2 * q1_inv_q1)
            # phi_grad = -0.5 * (q0_inv_q1 + v * q1_inv_q1) + 0.25 * (q0_inv_Q_inv_q0 + 2 * v * q0_inv_Q_inv_q1 + v**2 * q1_inv_Q_inv_q1)
            # phi_hess = -0.5 * q1_inv_q1 + q0_inv_Q_inv_q1 + v * q1_inv_Q_inv_q1 - 0.5 * (q0_inv_Q_inv_Q_inv_q0 + 2 * v * q0_inv_Q_inv_Q_inv_q1 + v**2 * q1_inv_Q_inv_Q_inv_q1)
            phi = 0.25 * (q0_inv_q0 + 2 * v * q0_inv_q1 + v**2 * q1_inv_q1)
            phi_grad = 0.5 * (q0_inv_q1 + v * q1_inv_q1) - 0.25 * (q0_inv_Q_inv_q0 + 2 * v * q0_inv_Q_inv_q1 + v**2 * q1_inv_Q_inv_q1)
            phi_hess = 0.5 * q1_inv_q1 - q0_inv_Q_inv_q1 - v * q1_inv_Q_inv_q1 + 0.5 * (q0_inv_Q_inv_Q_inv_q0 + 2 * v * q0_inv_Q_inv_Q_inv_q1 + v**2 * q1_inv_Q_inv_Q_inv_q1)
        else:
            # phi = -0.25 * q0.T @ q0
            # phi_grad = -0.5 * q1.T @ q0 + 0.25 * q0.T @ Q @ q0
            # phi_hess = -0.5 * q1.T @ q1 + q1.T @ Q @ q0 - 0.5 * q0.T @ Q @ Q @ q0
            phi = 0.25 * q0.T @ q0
            phi_grad = 0.5 * q1.T @ q0 - 0.25 * q0.T @ Q @ q0
            phi_hess = 0.5 * q1.T @ q1 - q1.T @ Q @ q0 + 0.5 * q0.T @ Q @ Q @ q0
            # print(gradient_phi(v, q0, q1, np.eye(2 * d + 2), Q))

        return phi, phi_grad, phi_hess
    
    def projected_newton(self, v0, q0, q1, alpha = 0.4, beta = 0.5):
        logger = self.logger
        v = v0
        newton_decrement_sq = np.inf
        count = 0

        if logger:
            logger.info(f"\n"
                        f"Projected Newton started\n"
                        )

        while newton_decrement_sq > 1e-10:
            phi_value, phi_grad, phi_hess = self.phi_info(v, q0, q1)
            newton_step = - phi_grad / phi_hess
            newton_decrement_sq = - phi_grad * newton_step
            # print(f"Newton decrement squared ins step_{count}: ", newton_decrement_sq)

            # if logger:
            #     logger.info(f"\n"
            #                 f"Step {count} started\n"
            #                 f"Current value: {v}\n"
            #                 f"Newton decrement squared: {newton_decrement_sq}\n"
            #                 f"Newton step: {newton_step}\n"
            #                 )

            # backtracking line search
            t = 1
            stop = False
            while not stop:
                v_new = v + newton_step * t
                phi_value_new, _, _ = self.phi_info(v + t * newton_step, q0, q1)
                if phi_value_new <= phi_value + alpha * t * phi_grad * newton_step:
                    v = v_new
                    phi, _, _ = self.phi_info(v, q0, q1)
                    stop = True
                    # v = max(v_new, 0)
                    # phi = self.phi_info(v, q0, q1)[0]
                    # stop = True
                else:
                    t *= beta

            # if logger:
            #     logger.info(f"\n"
            #                 f"Step {count} finished\n"
            #                 f"Updated value: {v}\n"
            #                 f"Objective value: {phi}\n"
            #                 f"Step size: {t}\n"
            #                 )
            count += 1

        if logger:
            logger.info(f"\n"
                        f"Projected Newton finished\n"
                        # f"Optimal value: {v}\n"
                        # f"Objective value: {phi}\n"
                        f"Number of iterations: {count}\n"
                        )
        # print("Newton Count = ", count)
        # v = max(v, 0)
        # phi, _, _ = self.phi_info(v, q0, q1)
            
        return v, phi

    def local_QCQP_newton(self, source_index, target_index):
        logger = self.logger
        # Solve the local QCQP using the Newton's method
        d = self.d
        X = self.X
        varphi, g = self.varphi, self.g
        u_source, u_target = self.u_source, self.u_target
        v_source, v_target = self.v_source, self.v_target 
        Q = self.Q
        eig_value, U = np.linalg.eig(Q)
        
        i, j = source_index, target_index
        varphi_i, varphi_j = varphi[i], varphi[j]
        g_i, g_j = g[i], g[j]
        
        eta_i, eta_j = np.insert(g_i, 0, varphi_i), np.insert(g_j, 0, varphi_j)  # construct eta vector
        u_ij_source, u_ji_target = u_source[i, j], u_target[j, i]
        v_ij_source, v_ji_target = v_source[i, j, :], v_target[j, i, :]
        z_ij_source, z_ji_target = np.insert(v_ij_source, 0, u_ij_source), np.insert(v_ji_target, 0, u_ji_target)  # construct z vector for scaled duals
        
        q0 = -2 * np.hstack((eta_i - z_ij_source, eta_j - z_ji_target))
        q1 = np.hstack((np.insert(X[j] - X[i], 0, 1), np.insert(np.zeros(d), 0, -1)))
        eig_value, U = np.linalg.eig(Q)

        if logger:
            logger.info(f"\n"
                        f"@@@@@@@@@@@@@@@@@@@@@@@@\n"
                        f"Local QCQP Newton started\n"
                        f"Source index: {i}, Target index: {j}\n"
                        f"@@@@@@@@@@@@@@@@@@@@@@@@\n"
                        )
        
        # gradient of phi at v = 0
        phi_grad_0 = 0.5 * q1.T @ q0 - 0.25 * q0.T @ Q @ q0

        if phi_grad_0 > 0:
            v_star = 0
            xi_ij_star = - 0.5 * q0
        else:
            v0 = 10
            v_star, _ = self.projected_newton(v0, q0, q1)
            xi_ij_star = - 0.5 * (1/v_star) * U @ np.diag(1 / (1/v_star + eig_value)) @ U.T @ (q0 + v_star * q1)
            
        # retrieve optimal xi_star from v_star
        # if v_star != 0: 
        #     xi_ij_star = - 0.5 * (1/v_star) * U @ np.diag(1 / (1/v_star + eig_value)) @ U.T @ (q0 + v_star * q1)
        # else:
        #     xi_ij_star = - 0.5 * q0

        # if logger:
        #     logger.info(f"\n"
        #                 f"@@@@@@@@@@@@@@@@@@@@@@@@\n"
        #                 f"Local QCQP Newton finished\n"
        #                 f"Source index: {i}, Target index: {j}\n"
        #                 f"Optimal v_value: {v_star}\n"
        #                 f"Optimal xi_star: {xi_ij_star}\n"
        #                 f"@@@@@@@@@@@@@@@@@@@@@@@@\n"
        #                 )

        return xi_ij_star
    
    def node_update(self):
        logger = self.logger
        varphi_old = self.varphi.copy()
        g_old = self.g.copy()
        for i in range(self.m):
            # update varphi
            
            self.varphi[i] = (np.sum(self.varphi_e_source[i]) + np.sum(self.varphi_e_target[i]) + np.sum(self.u_source[i]) + np.sum(self.u_target[i])) / (2 * self.m - 2)
            # all elements in the diagonal are zero, thus there are essentially only 2(m - 1) terms.

            # update g
            vec_sum =  - (np.sum(self.g_e_source[i], axis = 0) + np.sum(self.g_e_target[i], axis = 0) + np.sum(self.v_source[i], axis = 0) + np.sum(self.v_target[i], axis = 0)) * self.rho / 2
            for j in range(self.n):
                vec_sum += self.pi[i, j] * (self.lambda_lower * self.X[i] - self.Y[j]) 
            z =  - vec_sum / (1/self.m + self.rho * (self.m - 1)) # minimize \|g + K\|^2
            diff = z + self.lambda_lower * self.X[i]
            if np.linalg.norm(diff) <= self.radi:
                self.g[i] = z
            else:
                self.g[i] = self.radi * diff / np.linalg.norm(diff) - self.lambda_lower * self.X[i]

        # if logger:
        #     logger.info(f"\n"
        #                 f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        #                 f"Node update finished\n"
        #                 f"Updated varphi: {self.varphi}\n"
        #                 f"Updated g: {self.g}\n"
        #                 f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        #                 )
        
        self.dresi = self.rho ** 2 * (np.sum((self.g - g_old) ** 2) + norm(self.varphi - varphi_old) ** 2)
            
    def dual_update(self):
        logger = self.logger
        # Ax + Bz - c = 0
        delta_u_source = self.varphi_e_source - np.repeat(self.varphi[:, np.newaxis], self.m, 1)
        delta_u_target = self.varphi_e_target - np.repeat(self.varphi[:, np.newaxis], self.m, 1)
        delta_v_source = self.g_e_source - np.repeat(self.g[:, np.newaxis, :], self.m, axis=1)
        delta_v_target = self.g_e_target - np.repeat(self.g[:, np.newaxis, :], self.m, axis=1)
        for i in range(self.m):
            delta_u_source[i, i] = 0
            delta_u_target[i, i] = 0
            delta_v_source[i, i, :] = np.zeros(self.d)
            delta_v_target[i, i, :] = np.zeros(self.d) 
            #################### !!!! replace for loop with assigning indices ####################
        
        self.u_source = self.u_source + delta_u_source
        self.u_target = self.u_target + delta_u_target
        self.v_source = self.v_source + delta_v_source
        self.v_target = self.v_target + delta_v_target

        self.presi = np.sum(delta_u_source**2)
        + np.sum(delta_u_target**2) 
        + np.sum(delta_v_source**2)
        + np.sum(delta_v_target**2)

        # if logger:
        #     logger.info(f"\n"
        #                 f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        #                 f"Dual update finished\n"
        #                 f"Updated u_source: {self.u_source}\n"
        #                 f"Updated u_target: {self.u_target}\n"
        #                 f"Updated v_source: {self.v_source}\n"
        #                 f"Updated v_target: {self.v_target}\n"
        #                 f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        #                 )
    
    def edge_update(self, newton = False, check = False):
        logger = self.logger
        for i in range(self.m):
            # start_time = time.time()
            for j in range(self.m):
                if i == j:
                    continue
                else:
                    if check: 
                        xi1 = self.local_QCQP(i, j)
                        xi2 = self.local_QCQP_newton(i, j)
                        diff = xi1 - xi2
                        if np.linalg.norm(diff) > 1e-3:
                            if logger:
                                logger.warning(f"\n"
                                            f"Error in the local QCQP solver\n"
                                            f"Difference: {diff}\n"
                                            )
                        else:
                            if logger:
                                logger.info(f"\n"
                                            f"Solver and Newton get the same result\n"
                                            )
                        
                    if newton:
                        xi_ij_star = self.local_QCQP_newton(i, j)
                    else:
                        xi_ij_star = self.local_QCQP(i, j)

                    self.varphi_e_source[i, j] = xi_ij_star[0]
                    self.varphi_e_target[j, i] = xi_ij_star[self.d + 1]
                    self.g_e_source[i, j] = xi_ij_star[1: self.d + 1]  
                    self.g_e_target[j, i] = xi_ij_star[self.d + 2:]

        # if logger:
        #     logger.info(f"\n"
        #                 f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        #                 f"Edge update finished\n"
        #                 f"Updated varphi_e_source: {self.varphi_e_source}\n"
        #                 f"Updated varphi_e_target: {self.varphi_e_target}\n"
        #                 f"Updated g_e_source: {self.g_e_source}\n"
        #                 f"Updated g_e_target: {self.g_e_target}\n"
        #                 f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        #                 )

    def update_vars(self, presi_threshold=1e-6, dresi_threshold=1e-6, newton = False):
        logger = self.logger
        k = 0
        while (self.presi > presi_threshold or np.max(self.dresi) > dresi_threshold):
        # for k in range(self.max_iter):
            print("############### Round {} started ############### ".format(k))
            start_time = time.time()
            
            self.edge_update(newton = True, check = False)
            print("Edge update finished")
            self.node_update()
            print("Node update finished")
            # breakpoint()
            self.dual_update()
            print("Dual update finished")
            print("Objective value at iteration {}: {}".format(k, self.objective()))
            print("Primal residual at iteration {}: {}".format(k, self.presi))
            print("Dual residual at iteration {}: {}".format(k, self.dresi))
            if logger:
                logger.info(f"\n"
                            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                            f"Iteration {k} finished\n"
                            f"Objective value: {self.objective()}\n"
                            f"Primal residual: {self.presi}\n"
                            f"Dual residual: {self.dresi}\n"
                            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                            )   
            end_time = time.time()
            print("Time elapsed: ", end_time - start_time)

            k += 1

        obj = self.objective()

        return obj, self.varphi, self.g
    

########## Check with the QCQP solver ##########
def solve_OptCoupling_matrix(x, y, log_file_path = None):
    m, n = len(x), len(y)
    model = gp.Model("LP_OptCoupling")
    if log_file_path:
        model.setParam('LogFile', log_file_path)

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

def solve_QCQP_OptTuple(BX, BY, lambda_lower, lambda_upper, radi, logger = None, log_file_path = None):
    pi_star, _ = solve_OptCoupling_matrix(BX, BY, log_file_path)
    model = gp.Model("OptTuple_qcqp")
    m = len(BX)
    n = len(BY)
    dim = len(BX[0])
        
    # Set NumericFocus parameter to 2 (Aggressive numerical emphasis)
    model.setParam('NumericFocus', 2)
    if log_file_path:
        model.setParam('LogFile', log_file_path)
    
    tilde_BIg = {}
    tilde_varphi = {}
    for i in range(m):
        tilde_BIg[i] = model.addMVar(shape = (dim,), lb=-GRB.INFINITY, name="tilde_BIg_{}".format(i))
        tilde_varphi[i] = model.addVar(lb=-GRB.INFINITY, name="tilde_varphi_{}".format(i))
    model.update()
    
    # Define the objective function
    obj_expr = gp.QuadExpr()
    for i in range(m):
        for j in range(n):
            if pi_star[i, j] > 1e-8:
                obj_expr += ((BY[j] - tilde_BIg[i] - lambda_lower * BX[i]) @ (BY[j] - tilde_BIg[i]- lambda_lower * BX[i])) * pi_star[i, j]
            else:
                pass

    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Add constraints
    for i in range(m):
        aux = tilde_BIg[i] + lambda_lower * BX[i]
        model.addConstr(aux @ aux <= radi**2, "radius_constraint_{}".format(i))
        # breakpoint()
        for j in range(m):
            if i != j:
                constraint_expr = gp.QuadExpr()
                inner_product = tilde_BIg[i]@(BX[j] - BX[i])
                norm_squared_tilde_BIg = (tilde_BIg[i] - tilde_BIg[j])@(tilde_BIg[i] - tilde_BIg[j])
                # breakpoint()
                constraint_expr += tilde_varphi[i] - tilde_varphi[j] + inner_product + norm_squared_tilde_BIg / (2*(lambda_upper - lambda_lower))
                model.addConstr(constraint_expr <= 0, "constraint_{}_{}".format(i, j))
            else:
                pass
                
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found")
        optimal_tilde_BIg = np.array([[tilde_BIg[i][j].x for j in range(len(BY[0]))] for i in range(m)])
        optimal_tilde_varphi = np.array([tilde_varphi[i].x for i in range(m)])
        obj = model.objVal
        if logger:
            logger.info(f"\n"
                        f"Optimal solution found for the QCQP\n"
                        f"Objective value: {obj}\n"
                        f"Optimal tilde_BIg: {optimal_tilde_BIg}\n"
                        f"Optimal tilde_varphi: {optimal_tilde_varphi}\n"
                        )
    else:
        print("No optimal solution found")

    return obj, optimal_tilde_BIg, optimal_tilde_varphi
        
# np.random.seed(42)

# # Generate random data with 100 samples and 100 features
# data = np.random.randn(2, 2)

# mean = np.zeros(2)
# # Compute the covariance matrix
# covariance = np.cov(data, rowvar=False) + np.eye(2)

# ADMM_dir = "ADMM_test_smalldim2"
# os.makedirs(ADMM_dir, exist_ok=True)
# solver_logger, solver_log_file_path = configure_logging('solver', ADMM_dir, 'solver_test.log')
# ADMM_logger, ADMM_log_file_path = configure_logging('ADMM', ADMM_dir, 'ADMM_test_notnewton.log')

# X_sample = 10 * np.random.multivariate_normal(mean, covariance, 200)
# X_radi = 2 * math.ceil(np.max(np.linalg.norm(X_sample, axis=1)))
# Y_sample = 3 * 10 * np.random.multivariate_normal(mean, covariance, 200) + 5
# Y_radi = 3 * X_radi
# # breakpoint()
# pi_star, _ = solve_OptCoupling_matrix(X_sample, Y_sample)
# # obj, tilde_BIg_star, tilde_varphi_star = solve_QCQP_OptTuple(X_sample, Y_sample, 0.1, 10, Y_radi, solver_logger, solver_log_file_path)
# # print("Optimal tilde_BIg_star: ", tilde_BIg_star)
# # print("Optimal tilde_varphi_star: ", tilde_varphi_star)
# # print("Optimal objective value: ", obj)
# # breakpoint()
# ADMM_solver = QCQP_ADMM(X_sample, Y_sample, 0.1, 0.1, 10, pi_star, Y_radi, logger = ADMM_logger, log_file_path = ADMM_log_file_path)
# obj_ADMM, varphi_ADMM, g_ADMM = ADMM_solver.update_vars(newton = False, presi_threshold=0, dresi_threshold=0)
# print("Optimal objective value: ", obj_ADMM)
# print("Optimal varphi_ADMM: ", varphi_ADMM)
# print("Optimal g_ADMM: ", g_ADMM)





