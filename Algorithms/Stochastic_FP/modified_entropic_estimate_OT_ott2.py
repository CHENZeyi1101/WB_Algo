from typing import Optional
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from sklearn.neighbors import KNeighborsRegressor

class FirstOrderConditionInitializer:
    """
    Initialize the Sinkhorn algorithm using the first-order optimality condition from a previous run
    """
    def __init__(self):
        super().__init__()
        self.X_prev = None
        self.Y_prev = None
        self.f_prev = None
        self.g_prev = None
    
    def set_prev(self, 
                 X_prev: jnp.ndarray, Y_prev: jnp.ndarray,
                 f_prev: jnp.ndarray, g_prev: jnp.ndarray):
        self.X_prev = X_prev
        self.Y_prev = Y_prev
        self.f_prev = f_prev
        self.g_prev = g_prev

    def init_f_and_g(self, 
                     X_new : jnp.ndarray, Y_new : jnp.ndarray,
                     epsilon : jnp.ndarray):
        num_pt_X = X_new.shape[0]
        num_pt_Y = Y_new.shape[0]

        if self.X_prev is None or self.Y_prev is None or self.f_prev is None or self.g_prev is None:
            return jnp.zeros(num_pt_X), jnp.zeros(num_pt_Y)

        f_interp = [None] * num_pt_X
        g_interp = [None] * num_pt_Y

        for i in range(num_pt_X):
            exponents = (self.g_prev - jnp.sum(jnp.square(X_new[i, :] - self.Y_prev), axis=1)) / epsilon
            exponents_max = jnp.max(exponents)
            f_interp[i] = -epsilon * (jnp.log(jnp.mean(jnp.exp(exponents - exponents_max))) + exponents_max)

        for j in range(num_pt_Y):
            exponents = (self.f_prev - jnp.sum(jnp.square(Y_new[j, :] - self.X_prev), axis=1)) / epsilon
            exponents_max = jnp.max(exponents)
            g_interp[j] = -epsilon * (jnp.log(jnp.mean(jnp.exp(exponents - exponents_max))) + exponents_max)
        
        return jnp.array(f_interp), jnp.array(g_interp)


class KNNInitializer:
    """
    Initialize the Sinkhorn algorithm via the k-nearest-neighbors using the potentials from a previous run
    """
    def __init__(self, 
                 n_neighbors : int):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.f_knr = None
        self.g_knr = None
    
    def set_prev(self, 
                 X_prev: jnp.ndarray, Y_prev: jnp.ndarray,
                 f_prev: jnp.ndarray, g_prev: jnp.ndarray):
        self.f_knr = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.f_knr.fit(X_prev, f_prev)

        self.g_knr = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.g_knr.fit(Y_prev, g_prev)
    
    def init_f_and_g(self, 
                     X_new : jnp.ndarray, Y_new : jnp.ndarray,
                     epsilon : jnp.ndarray):

        if self.f_knr is None or self.g_knr is None:
            return jnp.zeros(X_new.shape[0]), jnp.zeros(Y_new.shape[0])

        f_interp = self.f_knr.predict(X_new)
        g_interp = self.g_knr.predict(Y_new)
        
        return jnp.array(f_interp.tolist()), jnp.array(g_interp.tolist())

@jax.jit
def sinkhorn_solve(X : jnp.ndarray, Y : jnp.ndarray, 
                   reg : float, 
                   init_f : jnp.ndarray, init_g : jnp.ndarray):
    out = sinkhorn.iterations(linear_problem.LinearProblem(pointcloud.PointCloud(X, Y, epsilon = reg)), 
                        sinkhorn.Sinkhorn(lse_mode = True, min_iterations=100, max_iterations=2000),
                        (init_f, init_g))
    return out.f, out.g

@jax.jit
def sinkhorn_modified_entropic_OT_map(X_new : jnp.ndarray, 
                                      Y : jnp.ndarray, g : jnp.ndarray, 
                                      reg : float, radius : float):
    diff_mat = (g[jnp.newaxis, :] - ((X_new[:, jnp.newaxis, :] - Y[jnp.newaxis, :, :]) ** 2).sum(axis = -1)) / reg
    weight_mat = jax.nn.softmax(diff_mat, axis = 1)

    X_new_norm_halfsq_diff = 0.5 * (jnp.sum(jnp.square(X_new), axis = 1) - radius ** 2)
    modification_weights = jnp.where(X_new_norm_halfsq_diff > 0, jnp.exp(-1 / X_new_norm_halfsq_diff), 0.0)
    return weight_mat @ Y + modification_weights[:, jnp.newaxis] * X_new

class modified_entropic_OT_map_estimate_ott2:

    r'''
    Python class for constructing the regularized entropic OT map estimator
    Attributes: 
    X: numpy array, shape (n, d)
        Support of the empirical measure \widehat{\mu}; i.e., samples from the source distribution \mu \in \CP(\CX)
    Y: numpy array, shape (m, d)
        Support of the empirical measure \widehat{\nu}; i.e., samples from the input distribution \nu \in \CP(\CY)
    log: boolean, default None
        If True, the class will log the outputs
    
    Methods:
    get_dual_potential(epsilon = None)
        Compute the dual potential g of the entropic regularized OT problem
    construct_entropic_OT_map(x)
        Construct the entropic OT map at the point x, and compute the image of x under the entropic OT map
    regularize_entropic_OT_map(M, x)
        Regularize the entropic OT map at the point x to make the corresponding potential strongly convex
    '''
    
    def __init__(self, X, Y, log = None, initializer : FirstOrderConditionInitializer | KNNInitializer | None = None):
        self.X = jnp.array(X)
        self.Y = jnp.array(Y)
        self.g_potential = None
        self.epsilon = None
        self.initializer = initializer

    @staticmethod
    def create_initializer(warm_start : dict | None):
        if warm_start is not None:
            if warm_start["type"] == "first-order":
                return FirstOrderConditionInitializer()
            elif warm_start["type"] == 'kNN':
                return KNNInitializer(warm_start["k"])
            else:
                raise ValueError("unknown warm-start method")
        else:
            return None
    
    def get_dual_potential(self, epsilon = None):
        '''
        In ott, the default cost function is the squared Euclidean distance (without the 0.5 factor).
        In our paper, we follow the convention in OT literature by using the 0.5 factor in front of the squared Euclidean distance, therefore the corresponding potential functions differ by a factor of 2.

        In the current version of ott-jax (0.6.0), the dual potential functions evaluated at X and Y are given by SinkhornOutput.f and SinkhornOutput.g, respectively. They can be directly accessed after solving the OT problem.

        For detailed information, refer to the source code: 
        ./src/ott/solvers/linear/sinkhorn.py
        '''

        if self.initializer is not None:
            init_f, init_g = self.initializer.init_f_and_g(self.X, self.Y, epsilon)
        else:
            init_f, init_g = jnp.zeros(self.X.shape[0]), jnp.zeros(self.Y.shape[0])

        f, g = sinkhorn_solve(self.X, self.Y, epsilon, init_f, init_g)
        # Make sure to wait for completion if using JAX with device async
        f = jax.block_until_ready(f)  # if available
        g = jax.block_until_ready(g)  # if available

        self.g_potential = g
        self.epsilon = epsilon

        if self.initializer is not None:
            self.initializer.set_prev(X_prev = self.X, Y_prev = self.Y, 
                                      f_prev = f, g_prev = g)
    
    def get_initializer(self):
        return self.initializer
    
    def compute_modified_entropic_OT_map(self, X_new, radius):
        X_new = jnp.array(X_new)
        out_list = []
        chunk_size = 10**9 // self.Y.shape[0] 

        for X_new_sub in [X_new[i:min(i + chunk_size, X_new.shape[0])] for i in range(0, X_new.shape[0], chunk_size)]:
            out_list.append(self._compute_modified_entropic_OT_map_inner(X_new_sub, radius))
        return jnp.vstack(out_list)
    
    def _compute_modified_entropic_OT_map_inner(self, X_new, radius):
        return sinkhorn_modified_entropic_OT_map(X_new, self.Y, self.g_potential, self.epsilon, radius).block_until_ready()