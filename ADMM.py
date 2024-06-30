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

def phi(v, q0, q1, P0, P1):
    qv = q0 + v * q1
    Pv = P0 + v * P1
    Pv_inv = solve(Pv, np.eye(Pv.shape[0]))
    return -0.25 * qv.T @ Pv_inv @ qv

def gradient_phi(v, q0, q1, P0, P1):
    qv = q0 + v * q1
    Pv = P0 + v * P1
    Pv_inv = solve(Pv, np.eye(Pv.shape[0]))
    
    grad = -0.5 * q1.T @ Pv_inv @ qv + 0.25 * qv.T @ Pv_inv @ P1 @ Pv_inv @ qv
    return grad

def hessian_phi(v, q0, q1, P0, P1):
    qv = q0 + v * q1
    Pv = P0 + v * P1
    Pv_inv = solve(Pv, np.eye(Pv.shape[0]))
    
    hessian = - 0.5 * q1.T @ Pv_inv @ q1 + q1.T @ Pv_inv @ P1 @ Pv_inv @ qv - 0.5 * qv.T @ Pv_inv @ P1 @ Pv_inv @ P1 @ Pv_inv @ qv
    return hessian

def projected_newton_method(q0, q1, P0, P1, v0, max_iter=100):
    v = v0
    v_values = [v0]
    phi_values = [phi(v0, q0, q1, P0, P1)]
    count = 1

    while count <= max_iter:
        grad = gradient_phi(v, q0, q1, P0, P1)
        H = hessian_phi(v, q0, q1, P0, P1)

        # breakpoint()
        
        # print(f'Iteration {count}: v = {v}, phi(v) = {phi(v, q0, q1, P0, P1)}, grad = {grad}')
        # if np.linalg.norm(grad) < 1e-6:
        #     break
        
        delta_v = grad / H
        # breakpoint()

        # Update step for maximization
        v_new = v - delta_v
        
        # Projection
        v_new = np.maximum(v_new, 0)
        # breakpoint()
        # Save values for visualization
        v_values.append(v_new)
        phi_values.append(phi(v_new, q0, q1, P0, P1))
        
        if np.linalg.norm(v_new - v) < 1e-5:
            break

        v = v_new
        count += 1

    return v_values, phi_values


def plot_PGD(q0, q1, P0, P1, v_values):
    v_values = np.array(v_values)
    phi_values = [phi(v, q0, q1, P0, P1) for v in v_values]
    
    plt.plot(v_values, phi_values, color='red')
    
    # Plot arrows
    for i in range(len(v_values) - 1):
        plt.quiver(v_values[i], phi_values[i],
                   v_values[i + 1] - v_values[i],
                   phi_values[i + 1] - phi_values[i],
                   angles='xy', scale_units='xy', scale=1, color='green')
    
    plt.scatter(v_values[0], phi_values[0], color='black', marker='x', s=100, label='Start (x0)')
    plt.scatter(v_values[-1], phi_values[-1], color='black', marker='x', s=100, label='End (x)')

    plt.legend()
    plt.xlabel('v')
    plt.ylabel('f(v)')
    plt.title('Projected Gradient Descent')
    plt.show()

class QCQP_ADMM:
    def __init__(self, X, Y, rho, lambda_lower, lambda_upper, pi, radi):
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
        self.Q = 1/(2 * (self.lambda_upper - self.lambda_lower)) * np.block([[np.zeros((self.d + 1, self.d + 1)), np.eye(self.d + 1)], [np.eye(self.d + 1), np.zeros((self.d + 1, self.d + 1))]])
        self.eigenvalue, self.U = np.linalg.eig(self.Q)
        self.D = np.diag(self.eigenvalue)

    def objective(self):
        summation = 0
        for i in range(self.m):
            for j in range(self.n):
                summation += self.pi[i, j] * np.linalg.norm(self.g[i] + self.lambda_lower * self. X[i] - self.Y[j]) ** 2
        return summation
        
        
    def node_update(self):
        varphi_old = self.varphi.copy()
        g_old = self.g.copy()
        for i in range(self.m):
            # update varphi
            # breakpoint()
            self.varphi[i] = (np.sum(self.varphi_e_source[i]) + np.sum(self.varphi_e_target[i]) + np.sum(self.u_source[i]) + np.sum(self.u_target[i])) / (2 * self.m - 2)

            # update g
            vec_sum =  - (np.sum(self.g_e_source[i], axis = 0) + np.sum(self.g_e_target[i], axis = 0) + np.sum(self.v_source[i], axis = 0) + np.sum(self.v_target[i], axis = 0)) * self.rho / 2
            # breakpoint()
            for j in range(self.n):
                vec_sum += self.pi[i, j] * (self.lambda_lower * self.X[i] - self.Y[j]) 
            z =  - vec_sum / (1/self.m + self.rho * (self.m - 1)) # minimize \|g + K\|^2
            diff = z + self.lambda_lower * self.X[i]
            if np.linalg.norm(diff) <= self.radi:
                self.g[i] = z
            else:
                self.g[i] = self.radi * diff / np.linalg.norm(diff) - self.lambda_lower * self.X[i]
        
        self.dresi = self.rho ** 2 * (np.sum((self.g - g_old) ** 2) + norm(self.varphi - varphi_old) ** 2)
            
    def dual_update(self):
        delta_u_source = self.varphi_e_source - np.repeat(self.varphi[:, np.newaxis], self.m, 1)
        delta_u_target = self.varphi_e_target - np.repeat(self.varphi[:, np.newaxis], self.m, 1)
        delta_v_source = self.g_e_source - np.repeat(self.g[:, np.newaxis, :], self.m, axis=1)
        delta_v_target = self.g_e_target - np.repeat(self.g[:, np.newaxis, :], self.m, axis=1)
        for i in range(self.m):
            delta_u_source[i, i] = 0
            delta_u_target[i, i] = 0
            delta_v_source[i, i, :] = np.zeros(self.d)
            delta_v_target[i, i, :] = np.zeros(self.d) #################### !!!! replace for loop with assiging indices ####################
        
        self.u_source = self.u_source + delta_u_source
        self.u_target = self.u_target + delta_u_target
        self.v_source = self.v_source + delta_v_source
        self.v_target = self.v_target + delta_v_target

        # self.presi = delta_u_source + delta_u_target + delta_v_source + delta_v_target

        self.presi = np.sum(delta_u_source**2)
        + np.sum(delta_u_target**2) 
        + np.sum(delta_v_source**2)
        + np.sum(delta_v_target**2)
        
        # for i in range(self.m):
        #     self.u_source[i, i] = 0
        #     self.u_target[i, i] = 0
        #     self.v_source[i, i, :] = np.zeros(self.d)
        #     self.v_target[i, i, :] = np.zeros(self.d)
        # breakpoint()

    def local_QCQP(self, source_index, target_index):
        d = self.d
        X = self.X
        varphi, g = self.varphi, self.g
        u_source = self.u_source
        u_target = self.u_target
        v_source = self.v_source
        v_target = self.v_target
        i, j = source_index, target_index
        varphi_i, varphi_j = varphi[i], varphi[j]
        g_i, g_j = g[i], g[j]
        # breakpoint()
        eta_i, eta_j = np.insert(g_i, 0, varphi_i), np.insert(g_j, 0, varphi_j)  # construct eta vector
        u_ij_source, u_ji_target = u_source[i, j], u_target[j, i]
        v_ij_source, v_ji_target = v_source[i, j, :], v_target[j, i, :]
        vartheta_ij_source, vartheta_ji_target = np.insert(v_ij_source, 0, u_ij_source), np.insert(v_ji_target, 0, u_ji_target)  # construct z vector for scaled duals
        # breakpoint()
        I = np.identity(2 * d + 2)
        block1 = np.zeros((d + 1, d + 1))
        np.fill_diagonal(block1[1:, 1:], 1)
        block2 = np.zeros((d + 1, d + 1))
        np.fill_diagonal(block2[1:, 1:], -1)
        Q = 1/(2 * (self.lambda_upper - self.lambda_lower)) * np.block([[block1, block2], [block2, block1]])
        q0 = -2 * np.hstack((eta_i - vartheta_ij_source, eta_j - vartheta_ji_target))
        q1 = np.hstack((np.insert(X[j] - X[i], 0, 1), np.insert(np.zeros(d), 0, -1)))

        try:
            model = gp.Model("Edge_QCQP")
            model.setParam('NumericFocus', 2)
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
            else: 
                print("q0_vector: ", q0)
                print("q1_vector: ", q1)
                print("xi: ", xi)
                # breakpoint()
            x_star = np.array(xi.x)
        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))
            pdb.set_trace()
        return x_star
    
    def phi_info(self, v, q0, q1):
        d = self.d
        Q = self.Q
        eig_value, U = self.eigenvalue, self.U
        D = self.D
        # I = np.eye(2 * d + 2)
        # qv = q0 + v * q1
        # Pv = I + v * Q
        # Pv_inv = solve(Pv, np.eye(Pv.shape[0]))
        # phi_true = -0.25 * qv.T @ Pv_inv @ qv
        # grad_true = -0.5 * q1.T @ Pv_inv @ qv + 0.25 * qv.T @ Pv_inv @ Q @ Pv_inv @ qv
        # hess_true = - 0.5 * q1.T @ Pv_inv @ q1 + q1.T @ Pv_inv @ Q @ Pv_inv @ qv - 0.5 * qv.T @ Pv_inv @ Q @ Pv_inv @ Q @ Pv_inv @ qv

        if v != 0:
            K = np.diag(1 / (np.ones(2 * d + 2) / v + eig_value))
            K2 = np.diag(1 / (np.ones(2 * d + 2) / v + eig_value)**2)
            K3 = np.diag(1 / (np.ones(2 * d + 2) / v + eig_value)**3)
            D2 = np.diag(eig_value**2)
        
            q0_inv_q0 = (1/v) * q0.T @ U @ K @ U.T @ q0
            q0_inv_q1 = (1/v) * q0.T @ U @ K @ U.T @ q1
            q1_inv_q1 = (1/v) * q1.T @ U @ K @ U.T @ q1
            q0_inv_Q_inv_q0 = (1 / v**2) * q0.T @ U @ K2 @ D @ U.T @ q0
            q0_inv_Q_inv_q1 = (1 / v**2) * q0.T @ U @ K2 @ D @ U.T @ q1
            q1_inv_Q_inv_q1 = (1 / v**2) * q1.T @ U @ K2 @ D @ U.T @ q1
            q0_inv_Q_inv_Q_inv_q0 = (1 / v**3) * q0.T @ U @ K3 @ D2 @ U.T @ q0
            q0_inv_Q_inv_Q_inv_q1 = (1 / v**3) * q0.T @ U @ K3 @ D2 @ U.T @ q1
            q1_inv_Q_inv_Q_inv_q1 = (1 / v**3) * q1.T @ U @ K3 @ D2 @ U.T @ q1

            phi = -0.25 * (q0_inv_q0 + 2 * v * q0_inv_q1 + v**2 * q1_inv_q1)
            phi_grad = -0.5 * (q0_inv_q1 + v * q1_inv_q1) + 0.25 * (q0_inv_Q_inv_q0 + 2 * v * q0_inv_Q_inv_q1 + v**2 * q1_inv_Q_inv_q1)
            phi_hess = -0.5 * q1_inv_q1 + q0_inv_Q_inv_q1 + v * q1_inv_Q_inv_q1 - 0.5 * (q0_inv_Q_inv_Q_inv_q0 + 2 * v * q0_inv_Q_inv_Q_inv_q1 + v**2 * q1_inv_Q_inv_Q_inv_q1)
        else:
            phi = -0.25 * q0.T @ q0
            phi_grad = -0.5 * q1.T @ q0 + 0.25 * q0.T @ Q @ q0
            phi_hess = -0.5 * q1.T @ q1 + q1.T @ Q @ q0 - 0.5 * q0.T @ Q @ Q @ q0
            # print(gradient_phi(v, q0, q1, np.eye(2 * d + 2), Q))

        return phi, phi_grad, phi_hess
    
    def projected_newton(self, v0, q0, q1):
        v = v0
        count = 1
        v_values = []
        phi_values = []

        while count <= 100:
            phi_value, phi_grad, phi_hess = self.phi_info(v, q0, q1)
            v_values.append(v)
            phi_values.append(phi_value)
            # breakpoint()
            
            # print(f'Iteration {count}: v = {v}, phi(v) = {phi_value}, grad = {phi_grad}, hess = {phi_hess}')
            
            delta_v = - phi_grad / phi_hess

            v_new = v + delta_v
            # Projection
            v_new = np.maximum(v_new, 0)
            
            if np.linalg.norm(v_new - v) < 1e-6:
                break

            v = v_new
            count += 1

        return v_values, phi_values

    def local_QCQP_newton(self, source_index, target_index):
        d = self.d
        X, _ = self.X, self.Y
        varphi, g = self.varphi, self.g
        u_source = self.u_source
        u_target = self.u_target
        v_source = self.v_source
        v_target = self.v_target
        i, j = source_index, target_index
        varphi_i, varphi_j = varphi[i], varphi[j]
        g_i, g_j = g[i], g[j]
        # breakpoint()
        eta_i, eta_j = np.insert(g_i, 0, varphi_i), np.insert(g_j, 0, varphi_j)  # construct eta vector
        u_ij_source, u_ji_target = u_source[i, j], u_target[j, i]
        v_ij_source, v_ji_target = v_source[i, j, :], v_target[j, i, :]
        z_ij_source, z_ji_target = np.insert(v_ij_source, 0, u_ij_source), np.insert(v_ji_target, 0, u_ji_target)  # construct z vector for scaled duals
        # breakpoint()
        I = np.identity(2 * d + 2)
        block1 = np.zeros((d + 1, d + 1))
        np.fill_diagonal(block1[1:, 1:], 1)
        block2 = np.zeros((d + 1, d + 1))
        np.fill_diagonal(block2[1:, 1:], -1)
        Q = 1/(2 * (self.lambda_upper - self.lambda_lower)) * np.block([[block1, block2], [block2, block1]])
        q0 = -2 * np.hstack((eta_i - z_ij_source, eta_j - z_ji_target))
        q1 = np.hstack((np.insert(X[j] - X[i], 0, 1), np.insert(np.zeros(d), 0, -1)))
        eig_value, U = np.linalg.eig(Q)
        D = np.diag(eig_value)

        v0 = 10
        v_values, _ = self.projected_newton(v0, q0, q1)
        # v_values2, _ = projected_newton_method(q0, q1, I, Q, v0)
        # breakpoint()
        v_star = v_values[-1]
        # plot_PGD(q0, q1, P0, P1, v_values)

        # xi_ij_star = - 0.5 * inv(I + v_star * Q) @ (q0 + v_star * q1)
        # breakpoint()
        if v_star != 0: 
            xi_ij_star = - 0.5 * (1/v_star) * U @ np.diag(1 / (1/v_star + eig_value)) @ U.T @ (q0 + v_star * q1)
        else:
            xi_ij_star = - 0.5 * q0

        return xi_ij_star
    
    def edge_update(self):
        for i in range(self.m):
            # start_time = time.time()
            for j in range(self.m):
                if i == j:
                    continue
                else:
                    # print("edge update: i = {}, j = {}".format(i, j))
                    # start_time = time.time()
                    xi_ij_star = self.local_QCQP_newton(i, j)
                    # end_time = time.time()
                    # print("Time elapsed: ", end_time - start_time)
                    # breakpoint()
                    self.varphi_e_source[i, j] = xi_ij_star[0]
                    self.g_e_source[i, j] = xi_ij_star[1: self.d + 1]
                    self.varphi_e_target[j, i] = xi_ij_star[self.d + 1]
                    self.g_e_target[j, i] = xi_ij_star[self.d + 2:]
            # end_time = time.time()
            # print("Time elapsed: ", end_time - start_time)
            # breakpoint()

    def update_vars(self, presi_threshold=1e-4, dresi_threshold=1e-4, max_iter = 300):
        k = 0
        while (self.presi > presi_threshold or np.max(self.dresi) > dresi_threshold) and k <= max_iter:
        # for k in range(self.max_iter):
            print("############### Round {} started ############### ".format(k))
            start_time = time.time()
            
            self.edge_update()
            print("Edge update finished")
            self.node_update()
            print("Node update finished")
            # breakpoint()
            self.dual_update()
            print("Dual update finished")
            # breakpoint()
            # print("Objective value at iteration {}: {}".format(k, self.objective()))
            # print("Primal residual at iteration {}: {}".format(k, self.presi))
            # print("Dual residual at iteration {}: {}".format(k, self.dresi))
            end_time = time.time()
            print("Time elapsed: ", end_time - start_time)
            # breakpoint()

            if k % 100 == 0 :
                print("Objective value at iteration {}: {}".format(k, self.objective()))
                print("Primal residual at iteration {}: {}".format(k, self.presi))
                print("Dual residual at iteration {}: {}".format(k, self.dresi))
                # breakpoint()

            k += 1

        obj = self.objective()
        # breakpoint()

        return obj, self.varphi, self.g
    
def solve_OptCoupling_matrix(x, y):
    m, n = len(x), len(y)
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

def solve_QCQP_OptTuple(BX, BY, lambda_lower, lambda_upper, radi):
    pi_star, _ = solve_OptCoupling_matrix(BX, BY)
    model = gp.Model("OptTuple_qcqp")
    m = len(BX)
    n = len(BY)
    dim = len(BX[0])
        
    # Set NumericFocus parameter to 2 (Aggressive numerical emphasis)
    model.setParam('NumericFocus', 2)
    
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
            if pi_star[i][j] > 1e-8:
                obj_expr += ((BY[j] - tilde_BIg[i] - lambda_lower * BX[i]) @ (BY[j] - tilde_BIg[i]- lambda_lower * BX[i])) * pi_star[i][j]
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
    else:
        print("No optimal solution found")

    return obj, optimal_tilde_BIg, optimal_tilde_varphi
        
# f(x) = x^t x + 2 x
# mean = [-1, 0, 2]  # mean of the bivariate Gaussian distribution
# covariance = [[1, 0.6, 0.2], [0.6, 1, 0.1], [0.2, 0.1, 1]]  # covariance matrix of the bivariate Gaussian distribution

# np.random.seed(42)

# # Generate random data with 100 samples and 100 features
# data = np.random.randn(3, 3)

# mean = np.zeros(3)
# # Compute the covariance matrix
# covariance = np.cov(data, rowvar=False)

# # mean = [0, 0]
# # covariance = [[1, 0.5], [0.5, 1]]

# np.random.seed(0)
# X_sample = 10 * np.random.multivariate_normal(mean, covariance, 10)
# X_radi = 2 * math.ceil(np.max(np.linalg.norm(X_sample, axis=1)))
# Y_sample = 3 * 10 * np.random.multivariate_normal(mean, covariance, 10) + 5
# Y_radi = 3 * X_radi
# # breakpoint()
# pi_star, _ = solve_OptCoupling_matrix(X_sample, Y_sample)
# obj, tilde_BIg_star, tilde_varphi_star = solve_QCQP_OptTuple(X_sample, Y_sample, 0.1, 10, Y_radi)
# print("Optimal objective value: ", obj)
# print("Optimal tilde_BIg_star: ", tilde_BIg_star)
# print("Optimal tilde_varphi_star: ", tilde_varphi_star)
# breakpoint()
# ADMM_solver = QCQP_ADMM(X_sample, Y_sample, 0.1, 0.1, 10, pi_star, Y_radi)
# obj_ADMM, varphi_ADMM, g_ADMM = ADMM_solver.update_vars()
# print("Optimal objective value: ", obj_ADMM)
# print("Optimal varphi_ADMM: ", varphi_ADMM)
# print("Optimal g_ADMM: ", g_ADMM)



