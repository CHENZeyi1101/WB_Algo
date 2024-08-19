import jax
import jax.numpy as jnp

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

import numpy as np

class entropic_OT_map_estimate:
    
    def __init__(self, X, Y, log = True):
        self.X = X
        self.Y = Y
        self.log = log
    
    def get_dual_potential(self):
        X, Y = self.X, self.Y
        geom = pointcloud.PointCloud(X, Y)
        prob = linear_problem.LinearProblem(geom) # uniform weights

        solver = sinkhorn.Sinkhorn()
        out = solver(prob) # EntropicPotential object
        dual_potentials = out.to_dual_potentials()

        self.g_potential = dual_potentials.f.keywords['potential']
        # the potential g corresponds to the output of EntropicPotntial.f
        # c.f. https://ott-jax.readthedocs.io/en/latest/_modules/ott/problems/linear/potentials.html#EntropicPotentials
        # Y = dual_potentials.f.keywords['y']
        self.epsilon = dual_potentials.f.keywords['epsilon']

        print(f"epsilon: {self.epsilon}")

    def construct_entropic_OT_map(self, x):
        Y = self.Y
        n = Y.shape[0]
        epsilon = self.epsilon
        g_potential = self.g_potential

        x_tile = np.tile(x, (n, 1))
        exponent_vec = (g_potential - (x_tile - Y)**2) / epsilon
        numerator = Y.T @ np.exp(exponent_vec)
        denominator = np.sum(np.exp(exponent_vec))
        entropic_image = numerator / denominator

        return entropic_image
    
    def regularize_entropic_OT_map(self, M, x):
        # M is the parameter in the definition of \varsigma_M
        # To avoid amendation of the original entropic OT map on the support of \widehat{\mu}
        # we set M > 0.5 * R^2, where R is the radius of the support of \widehat{\mu}

        entropic_image = self.construct_entropic_OT_map(x)
        half_xsq = 0.5 * x.T @ x
        if half_xsq <= M:
            return entropic_image
        else:
            regularized_entropic_image = entropic_image + np.exp(1/(half_xsq - M)) * x
            return regularized_entropic_image
        


    
