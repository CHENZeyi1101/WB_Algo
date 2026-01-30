from typing import Optional
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import warnings
import pdb
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import time

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from ott.initializers.linear.initializers import SinkhornInitializer  # adjust import path if your version differs

class FirstOrderConditionInitializer(SinkhornInitializer):
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

    def init_gv(
        self,
        ot_prob: linear_problem.LinearProblem,
        lse_mode: bool,
        rng: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """Initialize Sinkhorn potential/scaling f_u.

        Args:
        ot_prob: Linear OT problem.
        lse_mode: Return potential if ``True``, scaling if ``False``.
        rng: Random number generator for stochastic initializers.

        Returns:
        potential/scaling, array of size ``[n,]``.
        """
        geom : pointcloud.PointCloud = ot_prob.geom
        Y_curr = geom.y
        epsilon = geom.epsilon
        num_pt = Y_curr.shape[0]

        if self.X_prev is None or self.Y_prev is None or self.f_prev is None or self.g_prev is None:
            return jnp.zeros(num_pt)

        g_interp = [None] * num_pt
        for i in range(num_pt):
            exponents = (self.f_prev - jnp.sum(jnp.square(Y_curr[i, :] - self.X_prev), axis=1)) / epsilon
            exponents_max = jnp.max(exponents)
            g_interp[i] = -epsilon * (jnp.log(jnp.mean(jnp.exp(exponents - exponents_max))) + exponents_max)
        
        return jnp.array(g_interp)
    

    def init_fu(
        self,
        ot_prob: linear_problem.LinearProblem,
        lse_mode: bool,
        rng: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """Initialize Sinkhorn potential/scaling g_v.

        Args:
        ot_prob: Linear OT problem.
        lse_mode: Return potential if ``True``, scaling if ``False``.
        rng: Random number generator for stochastic initializers.

        Returns:
        potential/scaling, array of size ``[m,]``.
        """
        geom : pointcloud.PointCloud = ot_prob.geom
        X_curr = geom.x
        epsilon = geom.epsilon
        num_pt = X_curr.shape[0]

        if self.X_prev is None or self.Y_prev is None or self.f_prev is None or self.g_prev is None:
            return jnp.zeros(num_pt)

        f_interp = [None] * num_pt
        for i in range(num_pt):
            exponents = (self.g_prev - jnp.sum(jnp.square(X_curr[i, :] - self.Y_prev), axis=1)) / epsilon
            exponents_max = jnp.max(exponents)
            f_interp[i] = -epsilon * (jnp.log(jnp.mean(jnp.exp(exponents - exponents_max))) + exponents_max)

        return jnp.array(f_interp)

class KNNInitializer(SinkhornInitializer):
    """
    Initialize the Sinkhorn algorithm via the k-nearest-neighbors using the potentials from a previous run
    """
    def __init__(self, 
                 n_neighbors : int):
        super().__init__()
        self.n_neighbors = 1
        self.f_knr = None
        self.g_knr = None
    
    def set_prev(self, 
                 X_prev: jnp.ndarray, Y_prev: jnp.ndarray,
                 f_prev: jnp.ndarray, g_prev: jnp.ndarray):
        self.f_knr = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.f_knr.fit(X_prev, f_prev)

        self.g_knr = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.g_knr.fit(Y_prev, g_prev)

    def init_gv(
        self,
        ot_prob: linear_problem.LinearProblem,
        lse_mode: bool,
        rng: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """Initialize Sinkhorn potential/scaling f_u.

        Args:
        ot_prob: Linear OT problem.
        lse_mode: Return potential if ``True``, scaling if ``False``.
        rng: Random number generator for stochastic initializers.

        Returns:
        potential/scaling, array of size ``[n,]``.
        """
        geom : pointcloud.PointCloud = ot_prob.geom
        Y_curr = geom.y

        if self.f_knr is None or self.g_knr is None:
            return jnp.zeros(Y_curr.shape[0])
        
        g_interp = self.g_knr.predict(Y_curr)
        
        return jnp.array(g_interp.tolist())
    

    def init_fu(
        self,
        ot_prob: linear_problem.LinearProblem,
        lse_mode: bool,
        rng: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """Initialize Sinkhorn potential/scaling g_v.

        Args:
        ot_prob: Linear OT problem.
        lse_mode: Return potential if ``True``, scaling if ``False``.
        rng: Random number generator for stochastic initializers.

        Returns:
        potential/scaling, array of size ``[m,]``.
        """
        geom : pointcloud.PointCloud = ot_prob.geom
        X_curr = geom.x

        if self.f_knr is None or self.g_knr is None:
            return jnp.zeros(X_curr.shape[0])

        f_interp = self.f_knr.predict(X_curr)
        
        return jnp.array(f_interp.tolist())


class modified_entropic_OT_map_estimate_ott:

    r'''
    Python class for constructing the regularized entropic OT map estimator
    Attributes: 
    X: numpy array, shape (n, d)
        Support of the empirical measure \widehat{\mu}; i.e., samples from the source distribution \mu \in \CP(\CX)
    Y: numpy array, shape (m, d)
        Support of the empirical measure \widehat{\nu}; i.e., samples from the input distribution \nu \in \CP(\CY)
    log: boolean, default True
        If True, the class will log the outputs
    
    Methods:
    get_dual_potential(epsilon = None)
        Compute the dual potential g of the entropic regularized OT problem
    construct_entropic_OT_map(x)
        Construct the entropic OT map at the point x, and compute the image of x under the entropic OT map
    regularize_entropic_OT_map(M, x)
        Regularize the entropic OT map at the point x to make the corresponding potential strongly convex
    '''
    
    def __init__(self, X, Y, log = True, initializer : SinkhornInitializer | None = None):
        self.X = X
        self.Y = Y
        self.log = log
        self.g_potential = None
        self.epsilon = None
        self.dual_potentials = None
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

        X, Y = self.X, self.Y
        geom = pointcloud.PointCloud(X, Y, epsilon = epsilon) # set the epsilon parameter for the entropic regularization
        prob = linear_problem.LinearProblem(geom) # uniform weights

        # t0 = time.perf_counter()
        solver = sinkhorn.Sinkhorn(initializer = self.initializer)
        out = solver(prob) # <class 'ott.solvers.linear.sinkhorn.SinkhornOutput'>
        # Make sure to wait for completion if using JAX with device async
        out = jax.block_until_ready(out)  # if available
        # t1 = time.perf_counter()

        # elapsed = t1 - t0
        # print(f"OT map constructed in {elapsed:.4f} seconds")

        self.dual_potentials = out.potentials
        self.g_potential = out.g
        self.epsilon = epsilon

        if self.initializer is not None:
            self.initializer.set_prev(X_prev = jnp.array(X), Y_prev = jnp.array(Y), 
                                      f_prev = out.f, g_prev = out.g)
    
    def get_initializer(self):
        return self.initializer
    
    def construct_entropic_OT_map(self, x):
        Y = self.Y
        n = Y.shape[0]
        epsilon = self.epsilon
        g_potential = self.g_potential

        x_tile = np.tile(x, (n, 1))
        exponent_vec = (g_potential - np.sum(np.square(x_tile - Y), axis = 1)) / epsilon
        exponent_vec_max = np.max(exponent_vec)
        exponent_vec -= exponent_vec_max # normalize the exponent_vec for numerical stability in np.exp()

        # Convert warnings to exceptions within this block
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                numerator = Y.T @ np.exp(exponent_vec)
                denominator = np.sum(np.exp(exponent_vec))
                entropic_image = numerator / denominator
            except Warning as w:
                print(f"Warning converted to exception: {w}")
                pdb.set_trace()  # Trigger breakpoint for debugging
            except Exception as e:
                print(f"Error encountered: {e}")
                pdb.set_trace()  # Trigger breakpoint for debugging

        return entropic_image
    
    def regularize_entropic_OT_map(self, M, x):
        # Regularize the entropic OT map at point x to make the corresponding potential strongly convex
        # To avoid amendation of the original entropic OT map on the support of \widehat{\mu}
        # We set M = 0.5 * R^2, where R is the radius of the support of \widehat{\mu}

        entropic_image = self.construct_entropic_OT_map(x)
        half_xsq = 0.5 * x.T @ x
        if half_xsq <= M:
            return entropic_image
        else:
            regularized_entropic_image = entropic_image + np.exp(-1/(half_xsq - M)) * x
            return regularized_entropic_image
        


    
