from typing import Optional
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import numpy as np
import time

from sklearn.neighbors import KNeighborsRegressor

from ott.initializers.linear.initializers import SinkhornInitializer  # adjust import path if your version differs

class CustomInitializer(SinkhornInitializer):
    def __init__(self, 
                 X_prev: jnp.ndarray, Y_prev: jnp.ndarray,
                 f_prev: jnp.ndarray, g_prev: jnp.ndarray):
        super().__init__()
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

        g_interp = np.zeros(num_pt)

        for i in range(num_pt):
            exponents = (self.f_prev - jnp.sum(jnp.square(Y_curr[i, :] - self.X_prev), axis=1)) / epsilon
            exponents_max = jnp.max(exponents)
            g_interp[i] = -epsilon * (jnp.log(jnp.mean(jnp.exp(exponents - exponents_max))) + exponents_max)
        
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
        epsilon = geom.epsilon
        num_pt = X_curr.shape[0]

        f_interp = np.zeros(num_pt)

        for i in range(num_pt):
            exponents = (self.g_prev - jnp.sum(jnp.square(X_curr[i, :] - self.Y_prev), axis=1)) / epsilon
            exponents_max = jnp.max(exponents)
            f_interp[i] = -epsilon * (jnp.log(jnp.mean(jnp.exp(exponents - exponents_max))) + exponents_max)

        return jnp.array(f_interp.tolist())

class KNNInitializer(SinkhornInitializer):
    def __init__(self, 
                 n_neighbors : int,
                 X_prev: jnp.ndarray, Y_prev: jnp.ndarray,
                 f_prev: jnp.ndarray, g_prev: jnp.ndarray):
        super().__init__()
        self.f_knr = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.f_knr.fit(X_prev, f_prev)

        self.g_knr = KNeighborsRegressor(n_neighbors=n_neighbors)
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
        f_interp = self.f_knr.predict(X_curr)
        
        return jnp.array(f_interp.tolist())

def main():
    num_of_rep = 5
    N = 10000
    reg = 1


    # Sinkhorn OT via OTT with log-sum-exp and warm-start
    print("With warm-start (first-order condition):")
    rs = np.random.RandomState(seed = 2500)
    time_list = np.zeros(num_of_rep)
    initializer = None
    for iter in range(num_of_rep):
        X = rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[2, 1], [1, 2]]) * 100, size = N)
        Y = rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1, 0], [0, 1]]) * 100, size = N)
        t0 = time.perf_counter()
        geom = pointcloud.PointCloud(X, Y, epsilon = reg) # set the epsilon parameter for the entropic regularization
        ott_problem = linear_problem.LinearProblem(geom) # uniform weights

        solver = sinkhorn.Sinkhorn(lse_mode = True, max_iterations=10**6, initializer=initializer)
        out = solver(ott_problem)
        initializer = CustomInitializer(X, Y, out.f, out.g)
        t1 = time.perf_counter()
        print(f"Iteration {iter}: {t1 - t0} seconds")
        time_list[iter] = t1 - t0

    print("\n")

    # Sinkhorn OT via OTT with log-sum-exp and warm-start
    print("With warm-start (1-NN):")
    rs = np.random.RandomState(seed = 2500)
    time_list = np.zeros(num_of_rep)
    initializer = None
    for iter in range(num_of_rep):
        X = rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[2, 1], [1, 2]]) * 100, size = N)
        Y = rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1, 0], [0, 1]]) * 100, size = N)
        t0 = time.perf_counter()
        geom = pointcloud.PointCloud(X, Y, epsilon = reg) # set the epsilon parameter for the entropic regularization
        ott_problem = linear_problem.LinearProblem(geom) # uniform weights

        solver = sinkhorn.Sinkhorn(lse_mode = True, max_iterations=10**6, initializer=initializer)
        out = solver(ott_problem)
        initializer = KNNInitializer(1, X, Y, out.f, out.g)
        t1 = time.perf_counter()
        print(f"Iteration {iter}: {t1 - t0} seconds")
        time_list[iter] = t1 - t0

    print("\n")

    # Sinkhorn OT via OTT with log-sum-exp
    print("Without warm-start:")
    time_list = np.zeros(num_of_rep)
    rs = np.random.RandomState(seed = 2500)
    for iter in range(num_of_rep):
        X = rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[2, 1], [1, 2]]) * 100, size = N)
        Y = rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1, 0], [0, 1]]) * 100, size = N)
        t0 = time.perf_counter()
        geom = pointcloud.PointCloud(X, Y, epsilon = reg) # set the epsilon parameter for the entropic regularization
        ott_problem = linear_problem.LinearProblem(geom) # uniform weights
        solver = sinkhorn.Sinkhorn(lse_mode = True, max_iterations=10**6)
        out = solver(ott_problem)
        t1 = time.perf_counter()
        print(f"Iteration {iter}: {t1 - t0} seconds")
        time_list[iter] = t1 - t0

if __name__ == "__main__":
    main()
