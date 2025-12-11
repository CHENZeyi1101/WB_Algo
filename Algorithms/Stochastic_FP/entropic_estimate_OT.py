import jax
import jax.numpy as jnp
import warnings
import pdb
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import time

import numpy as np
from scipy.linalg import sqrtm, norm

from ott.initializers.linear.initializers import SinkhornInitializer  # adjust import path if your version differs
from ott.problems.linear.linear_problem import LinearProblem

class WarmStartInitializer(SinkhornInitializer):
    def __init__(self, f_prev: jnp.ndarray, g_prev: jnp.ndarray):
        super().__init__()
        self.f_prev = f_prev
        self.g_prev = g_prev

    def init_dual_a(
        self,
        ot_prob: LinearProblem,
        lse_mode: bool,
        rng: jax.Array = None,
    ) -> jnp.ndarray:
        # Return the previously stored f_prev as initial dual potentials for source
        # Optionally you may check shapes:
        assert self.f_prev.shape == ot_prob.a.shape, \
            f"Warm start f_prev shape {self.f_prev.shape} does not match ot_prob.a shape {ot_prob.a.shape}"
        return self.f_prev

    def init_dual_b(
        self,
        ot_prob: LinearProblem,
        lse_mode: bool,
        rng: jax.Array = None,
    ) -> jnp.ndarray:
        # Return the previously stored g_prev as initial dual potentials for target
        assert self.g_prev.shape == ot_prob.b.shape, \
            f"Warm start g_prev shape {self.g_prev.shape} does not match ot_prob.b shape {ot_prob.b.shape}"
        return self.g_prev

class entropic_OT_map_estimate:

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
    
    def __init__(self, X, Y, log = True):
        self.X = X
        self.Y = Y
        self.log = log
        self.g_potential = None
        self.epsilon = None
        self.dual_potentials = None
    
    def get_dual_potential(self, epsilon = None, initializer = "default"):
        X, Y = self.X, self.Y
        geom = pointcloud.PointCloud(X, Y, epsilon = epsilon) # set the epsilon parameter for the entropic regularization
        prob = linear_problem.LinearProblem(geom) # uniform weights

        t0 = time.perf_counter()
        solver = sinkhorn.Sinkhorn(initializer=initializer)
        out = solver(prob) # <class 'ott.solvers.linear.sinkhorn.SinkhornOutput'>
        # Make sure to wait for completion if using JAX with device async
        out = jax.block_until_ready(out)  # if available
        t1 = time.perf_counter()

        elapsed = t1 - t0
        print(f"Solver constructed and run with warm-start initializer in {elapsed:.4f} seconds")

        self.dual_potentials = out.to_dual_potentials() # <class 'ott.problems.linear.potentials.EntropicPotentials'>
        # EntropicPotential object
        # c.f. https://ott-jax.readthedocs.io/en/latest/tutorials/geometry/000_point_cloud.html#transport-map-using-sinkhorn-potentials

        g_potential_machine = self.dual_potentials.g # <class 'jax.tree_util.Partial'>
        self.g_potential = g_potential_machine(Y)
        # potential function g evaluated at Y
        # c.f. https://ott-jax.readthedocs.io/en/latest/_modules/ott/problems/linear/potentials.html#EntropicPotentials
        self.epsilon = self.dual_potentials.f.keywords['epsilon']

        # print(f"epsilon: {self.epsilon}")

    def customize_initializers(self):
        f_init = self.dual_potentials.f(self.X)
        g_init = self.dual_potentials.g(self.Y)
        custom_initializer = WarmStartInitializer(f_init, g_init)
        return custom_initializer
    
    def construct_entropic_OT_map(self, x):
        Y = self.Y
        n = Y.shape[0]
        epsilon = self.epsilon
        g_potential = self.g_potential

        x_tile = np.tile(x, (n, 1))
        # print(max(norm(x_tile - Y, axis = 1)))
        exponent_vec = (g_potential - norm(x_tile - Y, axis = 1)**2) / epsilon
        exponent_vec_max = np.max(exponent_vec)
        exponent_vec -= exponent_vec_max
        # normalize the exponent_vec for numerical stability in np.exp()

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
        


    
