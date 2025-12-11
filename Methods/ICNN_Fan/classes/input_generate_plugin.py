import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import gurobipy as gp
from gurobipy import GRB
from scipy.linalg import sqrtm, pinv, norm, inv, solve
from scipy.spatial import KDTree

from true_WB import *

class convex_function:

    r'''
    Python class for
    1. Generate convex functions and produce convex-interpolable tuples, based on given (random) samples.
    2. The potential functions associated with the OT map between the input measures and the barycenter 
    would be combinations of these functions after shape-constraint transformation.

    DESCRIPTION:
    All convex functions are defined as the supremum of a set of convex functions within a certain type   .
    Function types:
          CPWA: Convex Piecewise Affine Function
          quadratic_sq: Quadratic Function with the quadratic term further squared
          quadratic_sqrt: Quadratic Function with the quadratic term square-rooted
    Attributes:
          x_samples: the sample points for generating the convex function
          num_functions: the number of convex functions to be generated within each function type
          log: whether to log the information
          logger: the logger object for logging the information
    Methods:
            generate_CPWA(): generate the CPWA functions
            generate_quadratic_sq(): generate the quadratic functions with the quadratic term squared
            generate_quadratic_sqrt(): generate the quadratic functions with the quadratic term square-rooted
            generate(): generate a convex function
            plot_func(): plot the convex functions
    Outputs for each function type:
          sample_value: the function values of the sample points
          sample_gradient: the function gradients of the sample points (i.e., the image of the mapping)
          max_indices: the indices of the convex functions that achieve the maximum values at the sample points
    '''

    def __init__(self, x_samples, num_functions = 8, log = False, logger = None):
        self.x_samples = x_samples
        self.dim = x_samples.shape[1]
        self.num_functions = num_functions
        self.log = log
        self.logger = logger

    def generate_CPWA(self, seed = None): 
        # f(x) = max_i {x^T coeff_i + intercept_i}
        # For each function element, generate random coefficients and intercepts.
        num_functions = self.num_functions
        x_samples = self.x_samples
        dim = self.dim
        logger = self.logger

        if self.log:
            logger.info(f" ####### Function type: CPWA #######")

        coeff_list = []
        intercept_list = []
        rng_CPWA = np.random.RandomState(seed)
        for _ in range(num_functions):
            coeff = (rng_CPWA.rand(dim) - 0.5) * 10
            intercept = (rng_CPWA.rand() - 0.5) * 10
            coeff_list.append(coeff)
            intercept_list.append(intercept)

        if self.log:
            logger.info(f"coeff_list: {coeff_list}")
            logger.info(f"intercept_list: {intercept_list}")

        values = np.zeros((num_functions, x_samples.shape[0]))
        gradient = np.zeros((num_functions, x_samples.shape[0], dim))
        for i in range(num_functions):
            values[i, :] = np.dot(x_samples, coeff_list[i]) + intercept_list[i]
            gradient[i, :]= np.repeat(coeff_list[i][np.newaxis, :], x_samples.shape[0], axis=0)
    
        sample_value = np.max(values, axis=0)
        max_indices = np.argmax(values, axis=0)
        sample_gradient = gradient[max_indices, np.arange(x_samples.shape[0]), :]

        if self.log:
            logger.info(
                        f"\n"
                        f"sample_value selected: {sample_value}\n"
                        f"sample_gradient selected: \n{sample_gradient}"
                    )
        
        return sample_value, sample_gradient, max_indices
    
    def generate_quadratic_sq(self, seed = None): 
        # f(x) = max_i {x^T Q_i x + b_i^T x + c_i}
        # For each function element, generate random quadratic matrices and linear terms.
        num_functions = self.num_functions
        x_samples = self.x_samples
        dim = self.dim
        logger = self.logger

        if self.log:
            logger.info(f" ####### Function type: square quadratic #######")

        values = np.zeros((num_functions, x_samples.shape[0]))
        gradient = np.zeros((num_functions, x_samples.shape[0], dim))
        rng_quadratic_sq = np.random.RandomState(seed)
        for f in range(num_functions):
            A = rng_quadratic_sq.rand(dim, dim)
            Q = np.dot(A, A.T) * 0.01
            b = (rng_quadratic_sq.randn(dim) - 0.5) * 10 + np.array([20, 20])
            c = rng_quadratic_sq.randn() * 10
            values[f, :] = (np.diag(x_samples @ Q @ x_samples.T)) ** 2 + x_samples @ b + c
            gradient[f, :] = np.diag(2 * (np.diag(x_samples @ Q @ x_samples.T))) @ (2 * x_samples @ Q) + np.repeat(b[np.newaxis, :], x_samples.shape[0], axis=0)
            if self.log:
                logger.info(f"Q of function_{f}: \n {Q}; \nb of function_{f}: {b}; c of function_{f}: {c}")
            
        sample_value = np.max(values, axis=0)
        max_indices = np.argmax(values, axis=0)
        sample_gradient = gradient[max_indices, np.arange(x_samples.shape[0]), :]

        if self.log:
            logger.info(
                        f"\n"
                        f"sample_value selected: {sample_value}\n"
                        f"sample_gradient selected: \n{sample_gradient}"
                    )

        return sample_value, sample_gradient, max_indices
    
    def generate_quadratic_sqrt(self, seed = None): 
        # f(x) = max_i {10 * (sqrt(x^T Q_i x + 10) + b_i^T x + c_i)}
        # For each function element, generate random quadratic matrices and linear terms.
        num_functions = self.num_functions
        x_samples = self.x_samples
        dim = self.dim
        logger = self.logger
        if self.log:
            logger.info(f" ####### Function type: sqrt quadratic #######")

        values = np.zeros((num_functions, x_samples.shape[0]))
        gradient = np.zeros((num_functions, x_samples.shape[0], dim))
        rng_quadratic_sqrt = np.random.RandomState(seed)
        for f in range(num_functions):
            A = rng_quadratic_sqrt.rand(dim, dim) + np.eye(dim)
            Q = (np.dot(A, A.T)) * 100
            b = (rng_quadratic_sqrt.randn(dim) - 0.5) * 10 + np.array([30, 30])
            c = rng_quadratic_sqrt.randn() * 10
            values[f, :] = 10 * (np.sqrt((np.diag(x_samples @ Q @ x_samples.T) + 10)) + x_samples @ b + c)
            gradient[f, :] = 10 * (0.5 * np.diag((np.diag(x_samples @ Q @ x_samples.T) + 10) ** (-0.5)) @ (2 * x_samples @ Q) + np.repeat(b[np.newaxis, :], x_samples.shape[0], axis=0))
            if self.log:
                logger.info(f"Q of function_{f}: \n{Q}; \nb of function_{f}: {b}; c of function_{f}: {c}")

        sample_value = np.max(values, axis=0)
        max_indices = np.argmax(values, axis=0)
        sample_gradient = gradient[max_indices, np.arange(x_samples.shape[0]), :]

        if self.log:
            logger.info(
                        f"\n"
                        f"sample_value selected: {sample_value}\n"
                        f"sample_gradient selected: \n{sample_gradient}"
                    )

        return sample_value, sample_gradient, max_indices
            
    def generate(self, seed = None):
        rng_randgen = np.random.RandomState(seed)
        func_type = rng_randgen.choice(['quadratic_sq', 'quadratic_sqrt'])
        if func_type == 'CPWA':
            print("CPWA")
            sample_value, sample_gradient, _ = self.generate_CPWA()
        elif func_type == 'quadratic_sq':
            print("quadratic_sq")
            sample_value, sample_gradient, _ = self.generate_quadratic_sq()
        elif func_type == 'quadratic_sqrt':
            print("quadratic_sqrt")
            sample_value, sample_gradient, _ = self.generate_quadratic_sqrt()
        return sample_value, sample_gradient
    
    def plot_func(self, sample_values, max_indices, name = None):
        if self.dim != 2:
            print("Visualization only supported for 2D distributions")
        else:
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
            if name:
                plt.savefig(name)
            else:
                plt.show()

class KDTreeWithInfo:

    r'''
    Python class for building a KDTree with additional information for the nearest neighbor search.
    Methods:
        rebuild_tree(): rebuild the KDTree with the updated points
        add_point(): add a point with additional information to the KDTree
        query(): query the nearest neighbor of a given point and return the point and the additional information
    '''

    def __init__(self):
        self.points_with_info = []

    def rebuild_tree(self):
        coordinates = np.array([entry[0] for entry in self.points_with_info])
        self.kd_tree = KDTree(coordinates)

    def add_point(self, point, info):
        self.points_with_info.append((point, info))
        self.rebuild_tree()

    def query(self, point):
        _, idx = self.kd_tree.query(point)
        return self.points_with_info[idx][0], self.points_with_info[idx][1]

class cvx_based_OTmap:

    r'''
    Python class for generating the convex functions as the ingredients of the potential functions associated with the OT maps 
    between the input measures and the barycenter.
    The generated functions should be shape-constrained (smooth and strongly convex) and infinitely differentiable
    to make the input measures admissible.

    Attributes:
        x_samples: the sample points for generating the convex functions (without shape constraints)
        x_values: the function values of the sample points
        x_gradients: the function gradients of the sample points
        log: boolean, default True
        kdtree: the KDTree with additional information for the nearest neighbor search

    Methods:
        shape_paras(): 
            generate the shape parameters for the interpolation; 
            note that the strong convexity & smoothness parameters should be between 0 and 2.
        interp_paras(): 
            generate the parameters for shape-constrained interpolations;
            specifically, tilde_BG and Bv are characterized via transformation from the convex-interpolable tuples.
        KS_estimate():
            estimate the value and gradient via the KS estimator at a given point.
        BA_estimate():
            estimate the value and gradient via the BA estimator at a given point.
        generate_samples():
            generate samples from the mapped points resulting from either the KS or BA estimator.

    '''

    def __init__(self, x_samples, x_values, x_gradients, log = True):
        self.x_samples = x_samples
        self.x_values = x_values
        self.x_gradients = x_gradients
        self.num_samples, self.dim = x_samples.shape[0], x_samples.shape[1]
        self.log = log
        self.kdtree = KDTreeWithInfo()

    def shape_paras(self, logger = None):
        # the shape parameters for the interpolation are generated randomly.
        # np.random.seed(seed)
        rng_shape = np.random.RandomState()
        l_para = rng_shape.rand(2) * 2 
        self.l_lower, self.l_upper = min(l_para), max(l_para)
        if self.log:
            logger.info(f"Shape parameters: l_lower = {self.l_lower}, l_upper = {self.l_upper}")
    
    def interp_paras(self, logger = None):
        x_samples, x_values, x_gradients = self.x_samples, self.x_values, self.x_gradients
        l_lower, l_upper = self.l_lower, self.l_upper
        X_interp = x_samples + x_gradients / (l_upper - l_lower)
        G_interp = x_gradients + l_lower * X_interp
        varphi_interp = x_values
        + np.diag(X_interp @ X_interp.T) * l_lower * l_upper / (2 * (l_upper - l_lower))
        + np.diag(G_interp @ G_interp.T) / (2 * (l_upper - l_lower))
        - np.diag(G_interp @ X_interp.T) * l_lower / (l_upper - l_lower)

        # tilde_BG, Bv: cf. the definitions in the paper regarding shape-constrained interpolations.
        tilde_BG = (G_interp - l_lower * X_interp).T
        Bv = varphi_interp 
        + np.diag(X_interp @ X_interp.T) * l_lower * l_upper / (2 * (l_upper - l_lower))
        + np.diag(G_interp @ G_interp.T) / (2 * (l_upper - l_lower))
        - np.diag(G_interp @ X_interp.T) * l_upper / (l_upper - l_lower)

        self.x_interp, self.G_interp, self.varphi_interp = X_interp, G_interp, varphi_interp
        self.tilde_BG, self.Bv = tilde_BG, Bv

        if self.log:
            logger.info(f"Parameters for interpolation: tilde_BG = \n {self.tilde_BG}, \n Bv = {self.Bv}")

    def KS_estimate(self, x, theta = 10, Tau = 5, logger = None, log_file_path = None):
        # cf., Theorem~4.8
        l_lower = self.l_lower
        l_upper = self.l_upper
        num_samples, dim = self.num_samples, self.dim

        tilde_BG, Bv = self.tilde_BG, self.Bv
        
        # Generate the Monte Carlo samples for the KS estimator (the number is preset as Tau)
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
    
    def BA_estimate(self, x, theta = 10, logger = None, log_file_path = None):
        l_lower = self.l_lower
        l_upper = self.l_upper
        num_samples, dim = self.num_samples, self.dim
        tilde_BG, Bv = self.tilde_BG, self.Bv

        class inner_optimization:
            r'''
            1. Solve the inner optimization problem for the BA estimator at a given point (solve for optimal weight).
            2. The inner optimization problem is solved via the Newton method (details cf. Boyd's "Convex Optimmization").
            '''
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
                
                # REMARK: We write the inner optimization problem as a minimization problem.
                # obj = - (tilde_BG.T @ x + Bv) @ w + (tilde_BG @ w) @ (tilde_BG @ w) / (2 * (l_upper - l_lower)) - sum(log(w)) / theta
                obj_value = (
                    - np.dot((tilde_BG.T @ x + Bv), w)
                    + np.linalg.norm(tilde_BG @ w) ** 2 / (2 * (l_upper - l_lower))
                    - (np.sum(np.log(w))) / theta
                )

                #### for SM_estimator ####
                # m = tilde_BG.shape[1]
                # obj_value = (
                #     - np.dot((tilde_BG.T @ x + Bv), w)
                #     + np.linalg.norm(tilde_BG @ w) ** 2 / (2 * (l_upper - l_lower))
                #     + (np.log(m) + np.dot(w, np.log(w))) / theta
                # )

                return obj_value
            
            def obj_gradient(self, w):
                x, tilde_BG, Bv, l_lower, l_upper = self.x, self.tilde_BG, self.Bv, self.l_lower, self.l_upper
                # m = tilde_BG.shape[1]
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
                ## One can always check the gradient formula using finite difference; 
                ## h should be set between 1e-6 and 1e-8 for numerical stability.
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
                # Fast computation of the Newton step by using the Woodbury matrix inversion lemma
                # Slow computation of the Hessian matrix by term-wise computation

                x, tilde_BG, Bv, l_lower, l_upper = self.x, self.tilde_BG, self.Bv, self.l_lower, self.l_upper
                d, m = tilde_BG.shape[0], tilde_BG.shape[1]
                gradient = self.obj_gradient(w)

                if not slow:
                    # The associated KKT system: (Boyd's book, p. 526)
                    # A is the all-one vector.

                    # cf. Boyd's book, p. 678 (C9)
                    # A = np.diag(1 / w ** 2) / theta, B = tilde_G, C = tilde_G / (lambda_upper - lambda_lower)
                    # Apply the matrix inversion lemma to compute the inverse of A + B @ C, which is the hessian.

                    # A_inv = theta * np.diag(w)
                    A_inv = theta * np.diag(w ** 2)
                    mid_inverse = solve(np.eye(d) + tilde_BG @ A_inv @ tilde_BG.T / (l_upper - l_lower), np.eye(d))
                    # hessian_inv = A_inv - A_inv @ tilde_BG.T @ mid_inverse @ tilde_BG @ A_inv / (l_upper - l_lower)
                    
                    # cf. Boyd's book, p. 674
                    # Block elimination for the KKT system
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

                    # cf. Boyd's book, p. 527 (10.14)
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

                # REMARK: There are two conditions in the stopping criterion for the backtracking line search.
                # 1. The weight should be non-negative. We set the threshold as 1e-20 for stability issues.
                # 2. The Armijo condition should be satisfied.

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
                                        # f"Optimal solution found in Step_{step_count}: {w}\n"
                                        f"Optimal solution found in Step_{step_count}\n"
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
                        # if self.log:
                        #     logger.info(
                        #                 f"\n"
                        #                 f"Current Solution: {w}\n"
                        #                 f"Stepsize: {t}\n"
                        #                 f"Step Count: {step_count}\n"
                        #                 f"Updated Objective Value: {new_value}\n"
                        #                 f"Newton Decrement Squared: {newton_decrement_sq}\n"
                        #                 f"Newton Step: {newton_step}\n"
                        #             )
                        w = w + t * newton_step
                        value_list.append(new_value)
                
                return w, value_list
            
        inner_optimizer = inner_optimization(x, tilde_BG, Bv, l_lower, l_upper)

        # w0 = np.ones(num_samples) / num_samples

        # REMARK: check whether the KDTree is empty for warm start;
        # if so, use the uniform distribution as the initial weight
        # otherwise, use the nearest neighbor as the initial weight

        if self.kdtree.points_with_info == []:
            w0 = np.ones(num_samples) / num_samples
        else:
            _, w0 = self.kdtree.query(x)

        # solve the inner optimization problem with the Newton method
        # supplement the KDTree with the new solution
        w_star, _ = inner_optimizer.newton_method(w0)
        self.kdtree.add_point(x, w_star)

        inner_obj_value = inner_optimizer.obj_value(w_star)

        # cf. the formulae in the paper for the evaluated value and gradient of the BA estimator
        BA_eval_value = - inner_obj_value + norm(x)**2 * l_lower / 2
        BA_eval_gradient = tilde_BG @ w_star + l_lower * x

        # if self.log:
        #     logger.info(f"BA evaluation value = {BA_eval_value}\n"
        #                 f"BA evaluation gradient = {BA_eval_gradient}")
            
        return BA_eval_value, BA_eval_gradient
    
    def generate_samples(self, source_samples, smoothing = "KS", logger = None, log_file_path = None):
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
    
class input_sampler:

    r'''
    Python class for 
    1. Arrange and combine raw shape-constrained smooth convex functions to characterize the potential functions
    2. Generate samples from the input measures.

    Attributes:
        raw_func_list: the list of raw convex functions (which are shape-constrained and infinitely differentiable);
    '''

    # FUNCTIONALITY:
    # 1. Arrange and combine raw shape-constrained smooth convex functions to characterize the potential functions
    # 2. Generate samples from the input measures.

    # DESCRIPTION:
    #       base_function_sample():
    #             generate the sample points for the convex functions, and store the samples in the function_sample_collection dictionary
    #       measure_sample():
    #             generate the sample points for the input measures, and store the samples in the sample_collection dictionary
    #       generate_samples():
    #             generate the samples from the input measures (from the sample_collection dictionary) when the measure index is specified
    #             the indices for sample collection are also specified as inputs.
    #       dist_visualize():
    #             visualize the input measures.

    # Inputs:
    #       raw_func_list: the list of raw convex functions (which are shape-constrained and infinitely differentiable);
    #                      each element is an object of the class cvx_based_OTmap corresponding to a convex function
    #       source_samples: the sample points for generating the input measures (from the true barycenter)
    #                      it is assumed that the size of source samples should be sufficiently large.

    def __init__(self, raw_func_list, source_samples, log = True, func_logger = None, measure_logger = None, func_log_file_path = None):
        self.raw_func_list = raw_func_list
        self.log = log
        self.source_samples = source_samples
        self.func_sample_collection = {}
        self.sample_collection = {}
        # self.source_collection = {}
        self.func_logger = func_logger
        self.measure_logger = measure_logger
        self.func_log_file_path = func_log_file_path

    def base_function_sample(self, smoothing = "BA"):
        self.smoothing = smoothing
        source_samples = self.source_samples
        raw_func_list = self.raw_func_list
        num_measures = len(raw_func_list)
        for i in range(num_measures):
            func = raw_func_list[i]
            if self.log:
                self.func_logger.info(f" ####### Function_{i} starts generating samples using the {smoothing} smoothing method #######")
            image = func.generate_samples(source_samples, smoothing = smoothing, logger = self.func_logger, log_file_path = self.func_log_file_path)
            self.func_sample_collection[f"function_{i}"] = image
            if self.log:
                self.func_logger.info(f" ####### Function_{i} ends generating samples #######")
    
    def measure_sample(self):
        # logger = self.measure_logger
        raw_func_list = self.raw_func_list
        num_measures = len(raw_func_list)
        source_samples = self.source_samples
        func_sample_collection = self.func_sample_collection

        #### combination type 1 ####
        # func = (1/2) * (\|x\|^2 + phi_index(x) - phi_{(index + 1) % num_functions}(x))
        # func_plus = raw_func_list[measure_index]
        # func_minus = raw_func_list[(measure_index + 1) % num_measures]

        # if self.log:
        #     logger.info(f" ####### Measure_{measure_index} starts generating samples #######")

        # image_plus = func_plus.generate_samples(source_samples, smoothing, logger, log_file_path)
        # image_minus = func_minus.generate_samples(source_samples, smoothing, logger, log_file_path)
        # samples_generated = source_samples + (image_plus - image_minus) / 2
        
        # if self.log:
        #     logger.info(f" ####### Measure_{measure_index} ends generating samples #######")
        #     logger.info(f"Source samples of Measure_{measure_index}: \n {source_samples}")
        #     logger.info(f"Image data of Function_{measure_index}: \n {image_plus}")
        #     logger.info(f"Image data of Function_{(measure_index + 1) % num_measures}: \n {image_minus}")
        #     logger.info(f"Samples generated by Measure_{measure_index}: \n {samples_generated}")

        # self.source_collection[measure_index] = source_samples
        # self.sample_collection[f'image+_{measure_index}'] = image_plus
        # self.sample_collection[f'image-_{measure_index}'] = image_minus
        # self.sample_collection[measure_index] = samples_generated

        for measure_index in range(num_measures):
        #### combination type 2 ####
            if num_measures % 2 == 1:
                if measure_index == num_measures - 1:
                    # func_1st = raw_func_list[measure_index]
                    # func_2nd = raw_func_list[0]
                    image_1st = func_sample_collection[f"function_{measure_index}"]
                    image_2nd = func_sample_collection[f"function_{0}"]
                    samples_generated = source_samples + (image_1st - image_2nd) / 2
                    
                elif measure_index < (num_measures - 1) / 2:
                    # func_1st = raw_func_list[measure_index]
                    # func_2nd = raw_func_list[measure_index + 2]
                    image_1st = func_sample_collection[f"function_{measure_index}"]
                    image_2nd = func_sample_collection[f"function_{measure_index + 2}"]
                    samples_generated = (image_1st + image_2nd) / 2
                    
                elif measure_index >= (num_measures - 1) / 2:
                    # func_1st = raw_func_list[2 * measure_index - num_measures + 2]
                    # func_2nd = raw_func_list[2 * measure_index - num_measures + 3]
                    image_1st = func_sample_collection[f"function_{2 * measure_index - num_measures + 2}"]
                    image_2nd = func_sample_collection[f"function_{2 * measure_index - num_measures + 3}"]
                    samples_generated = 2 * source_samples - (image_1st + image_2nd) / 2

            else:
                if measure_index < num_measures / 2:
                    # func_1st = raw_func_list[measure_index]
                    # func_2nd = raw_func_list[measure_index + 2]
                    image_1st = func_sample_collection[f"function_{measure_index}"]
                    image_2nd = func_sample_collection[f"function_{measure_index + 2}"]
                    samples_generated = (image_1st + image_2nd) / 2
                    
                elif measure_index >= num_measures / 2:
                    # func_1st = raw_func_list[2 * measure_index - num_measures]
                    # func_2nd = raw_func_list[2 * measure_index - num_measures + 1]
                    image_1st = func_sample_collection[f"function_{2 * measure_index - num_measures}"]
                    image_2nd = func_sample_collection[f"function_{2 * measure_index - num_measures + 1}"]
                    samples_generated = 2 * source_samples - (image_1st + image_2nd) / 2

            self.sample_collection[f"measure_{measure_index}"] = samples_generated

            # if self.log:
            #     logger.info(f"Samples generated by Measure_{measure_index}: \n {samples_generated}")

    def generate_samples(self, measure_index, idx_start = 0, idx_end = 100):
        samples = self.sample_collection[measure_index][idx_start: idx_end, :]
        return samples
    
    # def dist_visualize(self, oneplot = False, save = False, savefile = None):
    #     raw_func_list = self.raw_func_list
    #     num_measures = len(raw_func_list)
    #     smoothing = self.smoothing

    #     if not oneplot:
    #         fig, axes = plt.subplots(1, num_measures, figsize=(18, 6), sharey=True)
            
    #         colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'olive', 'cyan']
    #         markers = ['s', '^', 'D', 'v', 'h', '*', 'p', 'P']
            
    #         for i in range(num_measures):
    #             source = self.source_samples[0: 500, :]
    #             image = self.sample_collection[f"measure_{i}"][0: 500, :]
    #             axes[i].scatter(source[:, 0], source[:, 1], color='black', marker='x', s=10, label="Source")
    #             axes[i].scatter(image[:, 0], image[:, 1], color=colors[i % len(colors)], marker=markers[i % len(markers)], s=15, label=f"Measure_{i}")
    #             axes[i].set_title(f"Measure_{i}")
    #             axes[i].set_xlabel('X1')
    #             if i == 0:
    #                 axes[i].set_ylabel('X2')
    #             axes[i].legend()
            
    #         fig.suptitle("Scatter Plot of Different Functions")
    #         if save:
    #             plt.savefig(f"{smoothing}_" + savefile)
    #         else:
    #             plt.show()

    #     else:
    #         fig, ax = plt.subplots(figsize=(18, 6))
        
    #         colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'olive', 'cyan']
    #         markers = ['s', '^', 'D', 'v', 'h', '*', 'p', 'P']
            
    #         for i in range(num_measures):
    #             source = self.source_samples[0: 500, :]
    #             image = self.sample_collection[f"measure_{i}"][0: 500, :]
    #             ax.scatter(source[:, 0], source[:, 1], color='black', marker='x', s=10, label=f"Source_{i}" if i == 0 else "")
    #             ax.scatter(image[:, 0], image[:, 1], color=colors[i % len(colors)], marker=markers[i % len(markers)], s=15, label=f"Measure_{i}")
            
    #         ax.set_title("Scatter Plot of Different Functions")
    #         ax.set_xlabel('X1')
    #         ax.set_ylabel('X2')
    #         ax.legend()
            
    #         if save:
    #             plt.savefig(f"{smoothing}_" + savefile)
    #         else:
    #             plt.show()

    
    

    
    
                


# num_measures = 5
# dim = 2
# log = False
# logger = None
# x_samples = np.random.uniform(low = -10, high = 10, size=(5, dim))
# cvxfunc_generator = convex_function(x_samples, num_functions = num_measures, log = log, logger = None)

# x_values, x_gradients, _ = cvxfunc_generator.generate_quadratic_sqrt(seed = 100) # 100, 200
# cvx_otmap_generator = cvx_based_OTmap(x_samples, x_values, x_gradients, log = log)

# # initialize parameters of cvx_otmap_generator

# cvx_otmap_generator.shape_paras(seed = 10, logger = logger) #4, 5
# cvx_otmap_generator.interp_paras(logger = logger)
# _, _ = cvx_otmap_generator.BA_estimate(x_samples[0], logger = logger, log_file_path = None)



    
## Example usage:
# mog = MixtureOfGaussians(2)
# mog.random_components(5)
# mog.set_truncation(50)
# samples = mog.sample(1000)
# cpwa = CPWA_function(samples, num_functions = 7)
# sample_values, sample_gradients, max_indices = cpwa.generate_CPWA()
# cpwa.plot_CPW(sample_values, max_indices, name = None)

# otmap_generator = CPWA_based_OTmap(samples, sample_values, sample_gradients)
# otmap_generator.shape_paras(seed = 42)
# otmap_generator.interp_paras()

# test_samples = mog.sample(10, seed = 10)
# test_samples_image1 = otmap_generator.generate_samples(test_samples, smoothing = "KS")
# test_samples_image2 = otmap_generator.generate_samples(test_samples, smoothing = "BA")

# # plot the images together
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].scatter(test_samples_image1[:, 0], test_samples_image1[:, 1], color = 'red', marker = 'o')
# ax[0].set_title('KS Gradient Image')
# ax[0].set_xlabel('X1')
# ax[0].set_ylabel('X2')
# ax[1].scatter(test_samples_image2[:, 0], test_samples_image2[:, 1], color = 'blue', marker = 'x')
# ax[1].set_title('BA Gradient Image')
# ax[1].set_xlabel('X1')
# ax[1].set_ylabel('X2')

# plt.show()
    






