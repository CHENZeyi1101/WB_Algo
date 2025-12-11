import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.linalg import sqrtm, pinv, norm, inv, solve

from config_log import *
from ADMM import QCQP_ADMM
from input_generate_plugin import KDTreeWithInfo

class plugin_OT_map_estimate:
    # FUNCTIONALITY:
    # 1. Estimate the OT map by using shape constrained least square regression
    # 2. Smooth the OT map by using the KS or BA estimator

    # DESCRIPTION:
    #           solve_optimal_coupling: solve the optimal coupling between X and Y by solving a LP problem
    #           solve_opt_tuples: solve the QCQP in search of the optimal tuples (tilde_g_star, tilde_varphi_star) for interpolation
    #           KS_estimate: estimate the KS evaluation value and gradient at x
    #           BA_estimate: estimate the BA evaluation value and gradient at x
    #           generate_samples: generate samples from the estimated measure

    # INPUT:
    #           X: the source samples
    #           Y: the target samples
    
    def __init__(self, X, Y, log = True):
        self.X = X
        self.Y = Y
        self.log = log
        self.kdtree = KDTreeWithInfo()
    
    def solve_optimal_coupling(self, 
                               logger = None, 
                               log_file_path = None):
        # the LP problem to solve the optimal coupling between X and Y
        X, Y = self.X, self.Y
        m, n = X.shape[0], Y.shape[0]

        # PROBLEM SETTING: create a LP model
        model = gp.Model("LP_OptCoupling")
        if self.log:
            model.Params.LogFile = log_file_path

        # PROBLEM SETTING: define the decision variables
        pi = {}
        for i in range(m):
            for j in range(n):
                pi[i, j] = model.addVar(lb=0.0, ub = 1.0, vtype=GRB.CONTINUOUS, name=f"pi_{i}_{j}")
        model.update()
        
        # PROBLEM SETTING: define the objective function
        obj = gp.quicksum(pi[i, j] * np.linalg.norm(X[i] - Y[j])**2 for i in range(m) for j in range(n))
        model.setObjective(obj, GRB.MINIMIZE)

        # PROBLEM SETTING: define the constraints
        for j in range(n):
            model.addConstr(gp.quicksum(pi[i, j] for i in range(m)) == 1/n) # sum(pi_ij) = 1/n for all j
        for i in range(m):
            model.addConstr(gp.quicksum(pi[i, j] for j in range(n)) == 1/m) # sum(pi_ij) = 1/m for all i
        
        # PROBLEM SETTING: optimize the model
        model.optimize()

        opt_coupling = np.array([[pi[i, j].x for j in range(n)] for i in range(m)])
        opt_obj = model.objVal

        if self.log:
            logger.info(f"\n"
                        f"Optimal coupling matrix: {opt_coupling}\n"
                        f"Optimal objective value: {opt_obj}\n"
                        )
        
        return opt_coupling, opt_obj
    
    def solve_opt_tuples(self, 
                         trunc_radi, 
                         lambda_lower = 0.01, 
                         lambda_upper = 100, 
                         ADMM = False, 
                         logger = None, 
                         log_file_path = None):
        
        # REMARK:
        # Solve the QCQP in search of the optimal tuples (tilde_g_star, tilde_varphi_star) for interpolation.
        
        X, Y = self.X, self.Y
        m, n = X.shape[0], Y.shape[0]
        dim = X.shape[1]
        pi_star, _ = self.solve_optimal_coupling(logger, log_file_path)
        
        # Solve the QCQP by using Gurobi solver.
        if not ADMM:
            if self.log:
                logger.info("Solving the QCQP by using the Gurobi solver")

            # PROBLEM SETTING: create a QCQP model
            model = gp.Model("OptTuple_qcqp")
            # model.setParam('NumericFocus', 3)
            model.setParam("OptimalityTol", 1e-3)
            if self.log:
                model.Params.LogFile = log_file_path

            # PROBLEM SETTING: define the decision variables
            tilde_g = {}
            tilde_varphi = {}
            for i in range(m):
                tilde_g[i] = model.addMVar(shape = (dim,), name="tilde_g_{}".format(i))
                tilde_varphi[i] = model.addVar(name="tilde_varphi_{}".format(i))
            model.update()

            # PROBLEM SETTING: define the objective function
            obj_expr = gp.QuadExpr()
            for i in range(m):
                for j in range(n):
                    if pi_star[i][j] > 1e-8:
                        obj_expr += ((Y[j] - tilde_g[i] - lambda_lower * X[i]) @ (Y[j] - tilde_g[i]- lambda_lower * X[i])) * pi_star[i][j]
                    else:
                        pass
            model.setObjective(obj_expr, GRB.MINIMIZE)

            # PROBLEM SETTING: define the constraints
            for i in range(m):
                aux = tilde_g[i] + lambda_lower * X[i]
                model.addConstr(aux @ aux <= trunc_radi ** 2)
                for j in range(m):
                    if i != j:
                        constraint_expr = gp.QuadExpr()
                        inner_product = tilde_g[i]@(X[j] - X[i])
                        norm_squared_tilde_g = (tilde_g[i] - tilde_g[j])@(tilde_g[i] - tilde_g[j])
                        constraint_expr += tilde_varphi[i] - tilde_varphi[j] + inner_product + norm_squared_tilde_g / (2*(lambda_upper - lambda_lower))
                        model.addConstr(constraint_expr <= 0, "constraint_{}_{}".format(i, j))
                    else:
                        pass

            # PROBLEM SETTING: optimize the model
            model.optimize()

            if model.status == GRB.OPTIMAL:
                optimal = True
                print("Optimal solution found")
                tilde_g_star = np.array([[tilde_g[i][j].x for j in range(len(Y[0]))] for i in range(m)])
                tilde_varphi_star = np.array([tilde_varphi[i].x for i in range(m)])
                logger.info(f"\n"
                            "Optimal solution found by Gurobi:\n"
                            f"tilde_g_star = {tilde_g_star}\n"
                            f"tilde_varphi_star = {tilde_varphi_star}"
                            )
            elif model.status == GRB.Status.SUBOPTIMAL:
                optimal = True
                tilde_g_star = np.array([[tilde_g[i][j].x for j in range(len(Y[0]))] for i in range(m)])
                tilde_varphi_star = np.array([tilde_varphi[i].x for i in range(m)])
                logger.info(f"\n"
                            "Optimal solution found by Gurobi:\n"
                            f"tilde_g_star = {tilde_g_star}\n"
                            f"tilde_varphi_star = {tilde_varphi_star}"
                            )
                logger.debug("Sub-optimal solution found")
                logger.debug(f"WARNING: Optimality Tolerance: {model.getParamInfo('OptimalityTol')}")
                logger.debug(f"upper_bound = {model.ObjVal}")
            elif model.status == GRB.Status.INFEASIBLE:
                optimal = False
                logger.debug("Model is infeasible")
                logger.debug("WARNING: No optimal solution found")
            elif model.status == GRB.Status.UNBOUNDED:
                optimal = False
                logger.debug("Model is unbounded")
                logger.debug("WARNING: No optimal solution found")
            else:
                optimal = False
                logger.debug(f"Optimization was stopped with status {model.status}")
                logger.debug("WARNING: No optimal solution found")

        else:
            # Solve the QCQP by using ADMM
            rho = 0.5 # penalty parameter
            ADMM_solver = QCQP_ADMM(X, Y, rho, lambda_lower, lambda_upper, pi_star, trunc_radi)
            _, tilde_varphi_star, tilde_g_star = ADMM_solver.update_vars(presi_threshold=1e-6, dresi_threshold=1e-6)
            optimal = True
            logger.info(f"\n"
                        "Optimal solution found by ADMM:\n"
                        f"tilde_g_star = {tilde_g_star}\n"
                        f"tilde_varphi_star = {tilde_varphi_star}"
                        )

        if optimal:
            tilde_BG = tilde_g_star.T
            g_star = tilde_g_star + lambda_lower * X
            varphi_star = tilde_varphi_star + np.diag(X @ X.T) * lambda_lower / 2
            Bv = (varphi_star
                + np.diag(X @ X.T) * lambda_lower * lambda_upper / (2 * (lambda_upper - lambda_lower))
                + np.diag(g_star @ g_star.T) / (2 * (lambda_upper - lambda_lower))
                - np.diag(g_star @ X.T) * lambda_upper / (lambda_upper - lambda_lower)
            )

            logger.info(f"\n"
                        f"tilde_BG = {tilde_BG}\n"
                        f"Bv = {Bv}\n"
                        f"lambda_lower = {lambda_lower}\n"
                        f"lambda_upper = {lambda_upper}"
                        )
            
            # store the shape and the interpolation paras.
            self.tilde_BG = tilde_BG
            self.Bv = Bv
            self.lambda_lower = lambda_lower
            self.lambda_upper = lambda_upper

    def KS_estimate(self, x, theta = 1000, Tau = 10, logger = None, log_file_path = None):
        # cf., Theorem~4.8
        X = self.X
        l_lower = self.lambda_lower
        l_upper = self.lambda_upper
        num_samples, dim = X.shape[0], X.shape[1]
        tilde_BG, Bv = self.tilde_BG, self.Bv
        
        eta = np.random.multivariate_normal(np.zeros(dim), (1 / theta) * np.eye(dim), Tau)
        MC_samples = np.tile(x, (Tau, 1)) - eta
        MC_value_sum = 0
        MC_grad_list = np.zeros(dim)

        def interp_QP(x):
            model = gp.Model("QP")
            if self.log:
                model.Params.LogFile = log_file_path
            # Define the variables
            BIw = {}
            BIw = model.addMVar(shape = (num_samples,), lb = 0, ub = 1, name = "BIw") # BIw as \R^m-vector
            # Define the objective
            obj_expr = gp.QuadExpr()
            tilde_BG_x_V = tilde_BG.T @ x + Bv
            innerprod = tilde_BG_x_V.T @ BIw
            norm_Gw = (tilde_BG @ BIw) @ (tilde_BG @ BIw)
            obj_expr += innerprod - (1 / (2 * (l_upper - l_lower))) * norm_Gw
            model.setObjective(obj_expr, GRB.MAXIMIZE)
            # Define the constraints
            model.addConstr(BIw.sum() == 1)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                optimal_weight = np.array(BIw.X)
                optimal_objective = model.ObjVal
                eval_value = optimal_objective + norm(x)**2 * l_lower / 2
                eval_gradient = tilde_BG @ optimal_weight + l_lower * x

                return eval_value, eval_gradient, optimal_weight
            else:
                print("No optimal solution found")

        # Monte Carlo estimation
        for t in range(Tau):
            eval_value, eval_gradient, optimal_weight = interp_QP(MC_samples[t])
            if self.log:
                logger.info(f"\n"
                            f"MC_Sample_{t}: \n"
                            f"MC evaluation value = {eval_value}\n"
                            f"MC evaluation gradient = {eval_gradient}\n"
                            f"optimal_weight = {optimal_weight}"
                        )
            MC_value_sum += eval_value
            MC_grad_list += eval_gradient

        KS_eval_value = MC_value_sum / Tau
        KS_eval_gradient = MC_grad_list / Tau
        if self.log:
            logger.info(
                        f"\n"
                        f"KS evaluation value = {KS_eval_value}\n"
                        f"KS evaluation gradient = {KS_eval_gradient}\n"
                        )

        return KS_eval_value, KS_eval_gradient

    def BA_estimate(self, x, theta = 1000, logger = None, log_file_path = None):
        X = self.X
        l_lower = self.lambda_lower
        l_upper = self.lambda_upper
        num_samples, dim = X.shape[0], X.shape[1]
        tilde_BG, Bv = self.tilde_BG, self.Bv

        class inner_optimization:
            def __init__(self, x, tilde_BG, Bv, l_lower, l_upper, log = self.log):
                self.x = x
                self.tilde_BG = tilde_BG
                self.Bv = Bv
                self.l_lower = l_lower
                self.l_upper = l_upper
                self.num_samples = tilde_BG.shape[1]
                self.dim = tilde_BG.shape[0]
                self.log = log

            def obj_value(self, w):
                x, tilde_BG, Bv, l_lower, l_upper = self.x, self.tilde_BG, self.Bv, self.l_lower, self.l_upper
                # m = tilde_BG.shape[1]

                # define the objective function
                obj_value = (
                    - np.dot((tilde_BG.T @ x + Bv), w)
                    + np.linalg.norm(tilde_BG @ w) ** 2 / (2 * (l_upper - l_lower))
                    - (np.sum(np.log(w))) / theta
                )

                #### for SM_estimator ####
                # obj_value = (
                #     - np.dot((tilde_BG.T @ x + Bv), w)
                #     + np.linalg.norm(tilde_BG @ w) ** 2 / (2 * (l_upper - l_lower))
                #     + (np.log(m) + np.dot(w, np.log(w))) / theta
                # )

                return obj_value
            
            def obj_gradient(self, w):
                x, tilde_BG, Bv, l_lower, l_upper = self.x, self.tilde_BG, self.Bv, self.l_lower, self.l_upper
                # m = tilde_BG.shape[1]

                # define the gradient associated with the objective function
                obj_gradient = ((-1/w) / theta 
                            + tilde_BG.T @ tilde_BG @ w / (l_upper - l_lower) 
                            - tilde_BG.T @ x - Bv
                )

                #### for SM_estimator ####
                # obj_gradient = (np.log(w) / theta 
                #             + np.ones(m) / theta 
                #             + tilde_BG.T @ tilde_BG @ w / (l_upper - l_lower) 
                #             - tilde_BG.T @ x - Bv
                # )

                ####### check gradient #######
                # gradient_check = np.zeros_like(w)
                # h = 1e-8
                # for i in range(len(w)):
                #     w_forward = np.copy(w)
                #     w_backward = np.copy(w)
                #     w_forward[i] += h
                #     # w_backward[i] -= h
                #     gradient_check[i] = (self.obj_value(w_forward) - self.obj_value(w_backward)) / h
                # print("check: ", gradient_check - obj_gradient)

                return obj_gradient
            
            def solve_KKT_system(self, w, slow = False):

                # REMARK:
                # Fast computation of the Newton step by using the Woodbury matrix inversion lemma
                # Slow computation of the Hessian matrix by term-wise computation

                x, tilde_BG, Bv, l_lower, l_upper = self.x, self.tilde_BG, self.Bv, self.l_lower, self.l_upper
                d, m = tilde_BG.shape[0], tilde_BG.shape[1]
                gradient = self.obj_gradient(w)

                if not slow:

                    # REMARK:
                    # Build the associated KKT system as in Boyd's book, p. 525 (10.11), 
                    # where A therein is the all-one vector in our setting.

                    # The system is solved by using the block elimination technique as in Boyd's book, p. 674
                    # The matrix inverse of hessian is computed by using the matrix inversion lemma as in Boyd's book, p. 678 (C9)
                    # where A = np.diag(1 / w ** 2) / theta, B = tilde_G, C = tilde_G / (lambda_upper - lambda_lower).

                    #### for SM_estimator ####
                    # A_inv = theta * np.diag(w)

                    # Compute hessian inverse
                    A_inv = theta * np.diag(w ** 2)
                    mid_inverse = solve(np.eye(d) + tilde_BG @ A_inv @ tilde_BG.T / (l_upper - l_lower), np.eye(d))
                    # hessian_inv = A_inv - A_inv @ tilde_BG.T @ mid_inverse @ tilde_BG @ A_inv / (l_upper - l_lower)
                    
                    # # Block elimination
                    # z1 = hessian_inv @ (-gradient)
                    # s = - np.ones(m) @ hessian_inv @ np.ones(m)
                    # z2 = - np.sum(z1) / s
                    # newton_step = z1 - hessian_inv @ np.ones(m) * z2

                    z1 = A_inv @ (-gradient) - A_inv @ (tilde_BG.T @ (mid_inverse @ (tilde_BG @ (A_inv @ (-gradient))))) / (l_upper - l_lower)
                    s = - np.ones(m) @ (A_inv @ np.ones(m) 
                                        - (A_inv @ (tilde_BG.T @ (mid_inverse @ (tilde_BG @ (A_inv @ np.ones(m)))))) / (l_upper - l_lower))
                    z2 = - np.sum(z1) / s
                    newton_step = z1 - (A_inv @ np.ones(m) 
                                        - (A_inv @ (tilde_BG.T @ (mid_inverse @ (tilde_BG @ (A_inv @ np.ones(m)))))) / (l_upper - l_lower)) * z2

                    # Define the squared newton decrement (for the stopping criterion; see Boyd's book)
                    newton_decrement_sq = - gradient @ newton_step
                
                else:
                    hessian = hessian = (np.diag(1 / w ** 2) / theta 
                           + tilde_BG.T @ tilde_BG / (l_upper - l_lower)
                    )
                    top_left = hessian
                    top_right = np.ones(m).reshape(-1, 1)
                    bottom_left = np.ones(m).reshape(1, -1)
                    bottom_right = np.array([[0]])
                    # Assemble the full matrix
                    top = np.hstack((top_left, top_right))
                    bottom = np.hstack((bottom_left, bottom_right))
                    KKT_matrix = np.vstack((top, bottom))

                    KKT_vector = np.zeros(m + 1)
                    gradient = self.obj_gradient(w)
                    KKT_vector[:-1] = - gradient
                    newton_step = solve(KKT_matrix, KKT_vector)[:-1] # the last element is the Lagrange multiplier
                    newton_decrement_sq = newton_step.T @ hessian @ newton_step
                    # directional_derivative = (self.objective_value(w + 1e-5 * newton_step) - self.objective_value(w)) / 1e-5

                return newton_step, newton_decrement_sq, gradient
            
            def backtracking_line_search(self, w, newton_step, gradient):
                alpha = 0.4
                beta = 0.5
                t = 1

                # REMARK: the while loop criterion is the Armijo condition 
                # plus a numerical stability condition in case w is too close to 0
                while np.any(w + t * newton_step <= 1e-20) or self.obj_value(w + t * newton_step) > self.obj_value(w) + alpha * t * gradient @ newton_step:
                    t *= beta
                return t, self.obj_value(w + t * newton_step)
            
            def newton_method(self, w, check = False):
                value_list = []
                step_count = 0
                while True:
                    newton_step, newton_decrement_sq, gradient = self.solve_KKT_system(w)
                    if check:
                        newton_step_slow, newton_decrement_sq_slow, gradient_slow = self.solve_KKT_system(w, slow = True)
                        print("newton_step_diff = ", newton_step - newton_step_slow)
                        print("newton_decrement_diff = ", newton_decrement_sq - newton_decrement_sq_slow)
                        print("gradient_diff = ", gradient - gradient_slow)
                    
                    if newton_decrement_sq < 1e-3:
                        if self.log:
                            logger.info(f"\n"
                                        f"Optimal solution found in Step_{step_count}: {w}\n"
                                        f"Objective Value: {self.obj_value(w)}\n"
                                        f"Eventual sqaured newton decrement = {newton_decrement_sq}"
                                    )
                        print("step_count = ", step_count)
                        break
                    else:
                        step_count += 1
                        print("squared newton decrement = ", newton_decrement_sq)
                        print("step_count = ", step_count)
                        newton_step[np.abs(newton_step) < 1e-10] = 0
                        t, new_value = self.backtracking_line_search(w, newton_step, gradient)
                        if self.log:
                            logger.info(
                                        f"\n"
                                        f"Current Solution: {w}\n"
                                        f"Stepsize: {t}\n"
                                        f"Step Count: {step_count}\n"
                                        f"Updated Objective Value: {new_value}\n"
                                        f"Newton Decrement Squared: {newton_decrement_sq}\n"
                                        f"Newton Step: {newton_step}\n"
                                    )
                        w = w + t * newton_step
                        value_list.append(new_value)
                
                return w, value_list
            
        inner_optimizer = inner_optimization(x, tilde_BG, Bv, l_lower, l_upper)

        # w0 = np.ones(num_samples) / num_samples

        # REMARK:
        # check whether the KDTree is empty for warm start;
        # if so, use the uniform distribution as the initial weight,
        # otherwise, use the nearest neighbor as the initial weight.
        if self.kdtree.points_with_info == []:
            w0 = np.ones(num_samples) / num_samples
        else:
            _, w0 = self.kdtree.query(x)

        # REMARK:
        # solve the inner optimization problem with the Newton method;
        # supplement the KDTree with the new solution
        w_star, _ = inner_optimizer.newton_method(w0)
        self.kdtree.add_point(x, w_star)

        inner_obj_value = inner_optimizer.obj_value(w_star)
        BA_eval_value = - inner_obj_value + norm(x)**2 * l_lower / 2
        BA_eval_gradient = tilde_BG @ w_star + l_lower * x

        if self.log:
            logger.info(f"BA evaluation value = {BA_eval_value}\n"
                        f"BA evaluation gradient = {BA_eval_gradient}")
            
        return BA_eval_value, BA_eval_gradient
    
    def generate_samples(self, source_samples, smoothing = "BA", logger = None, log_file_path = None):
        # REMARK: the source samples denote the samples from the generated measure (G(\mu)) at hand during iterations.
        num_samples = len(source_samples)
        samples_generated = np.zeros((num_samples, self.dim))
        count = 0
        if smoothing == "KS":
            for s in source_samples:
                _ , KS_gradient = self.KS_estimate(s, logger = logger, log_file_path = log_file_path)
                samples_generated[count, :] = KS_gradient
                count += 1
        elif smoothing == "BA":
            for s in source_samples:
                _ , BA_gradient = self.BA_estimate(s, logger = logger, log_file_path = log_file_path)
                samples_generated[count, :] = BA_gradient
                count += 1

        # if self.log:
        #     logger.info(f"Samples generated: {samples_generated}")

        return samples_generated
        
        
    

            


