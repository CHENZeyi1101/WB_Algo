from typing import Optional
import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import numpy as np
import time

from sklearn.neighbors import KNeighborsRegressor

from ott.initializers.linear.initializers import SinkhornInitializer  # adjust import path if your version differs

class FirstOrderConditionInitializer(SinkhornInitializer):
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

def sinkhorn_solve(X : jnp.ndarray, Y : jnp.ndarray, reg : float, init_f : jnp.ndarray, init_g : jnp.ndarray):
    out = sinkhorn.iterations(linear_problem.LinearProblem(pointcloud.PointCloud(X, Y, epsilon = reg)), 
                        sinkhorn.Sinkhorn(lse_mode = True, min_iterations=100, max_iterations=1000),
                        (init_f, init_g))
    return {"f": out.f, "g": out.g}

def main():
    num_of_rep = 5
    N = 5000
    seed = 2900
    reg = 100

    # # Sinkhorn OT via OTT with log-sum-exp, without jit
    # print("Without warm-start, without jit:")
    # time_list = np.zeros(num_of_rep)
    # rs = np.random.RandomState(seed = seed)
    # for iter in range(num_of_rep):
    #     X = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[2, 1], [1, 2]]) * 100, size = N))
    #     Y = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1, 0], [0, 1]]) * 100, size = N))
    #     init_f = jnp.zeros(N)
    #     init_g = jnp.zeros(N)
    #     t0 = time.perf_counter()
    #     out = sinkhorn_solve(X, Y, reg, init_f, init_g)
    #     out = jax.block_until_ready(out)
    #     t1 = time.perf_counter()
    #     f = out["f"]
    #     g = out["g"]
    #     print(f"    Iteration {iter}: {t1 - t0} seconds, val = {jnp.mean(f) + jnp.mean(g)}")
    #     time_list[iter] = t1 - t0

    print("\n")

    # Sinkhorn OT via OTT with log-sum-exp, with jit
    print("Without warm-start, with jit:")
    time_list = np.zeros(num_of_rep)
    rs = np.random.RandomState(seed = seed)
    sinkhorn_solve_jit = jax.jit(sinkhorn_solve)
    for iter in range(num_of_rep):
        X = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[2, 1], [1, 2]]) * 100, size = N))
        Y = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1, 0], [0, 1]]) * 100, size = N))
        init_f = jnp.zeros(N)
        init_g = jnp.zeros(N)
        t0 = time.perf_counter()
        out = sinkhorn_solve_jit(X, Y, reg, init_f, init_g)
        out = jax.block_until_ready(out)
        t1 = time.perf_counter()
        f = out["f"]
        g = out["g"]
        print(f"    Iteration {iter}: {t1 - t0} seconds, val = {jnp.mean(f) + jnp.mean(g)}")
        print(sinkhorn_solve_jit._cache_size())
        time_list[iter] = t1 - t0

    print("\n")


    # # Sinkhorn OT via OTT with log-sum-exp and warm-start via first-order condition, without jit
    # print("With warm-start (first-order condition), without jit:")
    # rs = np.random.RandomState(seed = seed)
    # time_list = np.zeros(num_of_rep)
    # initializer = None
    # for iter in range(num_of_rep):
    #     X = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[2, 1], [1, 2]]) * 100, size = N))
    #     Y = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1, 0], [0, 1]]) * 100, size = N))
    #     if iter == 0:
    #         init_f = jnp.zeros(N)
    #         init_g = jnp.zeros(N)
    #     else:
    #         initializer = FirstOrderConditionInitializer(X_prev = X, Y_prev = Y, f_prev = out.f, g_prev = out.g)
    #         prob = linear_problem.LinearProblem(pointcloud.PointCloud(X, Y, epsilon = reg))
    #         init_f, init_g = initializer.init_fu(prob, True), initializer.init_gv(prob, True)
    #     t0 = time.perf_counter()
    #     out = sinkhorn_solve(X, Y, reg, init_f, init_g)
    #     out = jax.block_until_ready(out)
    #     t1 = time.perf_counter()
    #     print(f"    Iteration {iter}: {t1 - t0} seconds, val = {jnp.mean(out.f) + jnp.mean(out.g)}")
    #     time_list[iter] = t1 - t0

    # print("\n")

    # # Sinkhorn OT via OTT with log-sum-exp and warm-start via first-order condition, with jit
    # print("With warm-start (first-order condition), with jit:")
    # rs = np.random.RandomState(seed = seed)
    # time_list = np.zeros(num_of_rep)
    # initializer = None
    # for iter in range(num_of_rep):
    #     X = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[2, 1], [1, 2]]) * 100, size = N))
    #     Y = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1, 0], [0, 1]]) * 100, size = N))
    #     if iter == 0:
    #         init_f = jnp.zeros(N)
    #         init_g = jnp.zeros(N)
    #     else:
    #         initializer = FirstOrderConditionInitializer(X_prev = X, Y_prev = Y, f_prev = out.f, g_prev = out.g)
    #         prob = linear_problem.LinearProblem(pointcloud.PointCloud(X, Y, epsilon = reg))
    #         init_f, init_g = initializer.init_fu(prob, True), initializer.init_gv(prob, True)
    #     t0 = time.perf_counter()
    #     out = sinkhorn_solve_jit(X, Y, reg, init_f, init_g)
    #     out = jax.block_until_ready(out)
    #     t1 = time.perf_counter()
    #     print(f"    Iteration {iter}: {t1 - t0} seconds, val = {jnp.mean(out.f) + jnp.mean(out.g)}")
    #     time_list[iter] = t1 - t0

    # print("\n")

    # # Sinkhorn OT via OTT with log-sum-exp and warm-start via kNN, without jit
    # print("With warm-start (kNN), without jit:")
    # rs = np.random.RandomState(seed = seed)
    # time_list = np.zeros(num_of_rep)
    # initializer = None
    # for iter in range(num_of_rep):
    #     X = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[2, 1], [1, 2]]) * 100, size = N))
    #     Y = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1, 0], [0, 1]]) * 100, size = N))
    #     if iter == 0:
    #         init_f = jnp.zeros(N)
    #         init_g = jnp.zeros(N)
    #     else:
    #         initializer = KNNInitializer(n_neighbors = 1, X_prev = X, Y_prev = Y, f_prev = out.f, g_prev = out.g)
    #         prob = linear_problem.LinearProblem(pointcloud.PointCloud(X, Y, epsilon = reg))
    #         init_f, init_g = initializer.init_fu(prob, True), initializer.init_gv(prob, True)
    #     t0 = time.perf_counter()
    #     out = sinkhorn_solve(X, Y, reg, init_f, init_g)
    #     out = jax.block_until_ready(out)
    #     t1 = time.perf_counter()
    #     print(f"    Iteration {iter}: {t1 - t0} seconds, val = {jnp.mean(out.f) + jnp.mean(out.g)}")
    #     time_list[iter] = t1 - t0

    # print("\n")

    # # Sinkhorn OT via OTT with log-sum-exp and warm-start via kNN, with jit
    # print("With warm-start (kNN), with jit:")
    # rs = np.random.RandomState(seed = seed)
    # time_list = np.zeros(num_of_rep)
    # initializer = None
    # for iter in range(num_of_rep):
    #     X = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[2, 1], [1, 2]]) * 100, size = N))
    #     Y = jnp.array(rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1, 0], [0, 1]]) * 100, size = N))
    #     if iter == 0:
    #         init_f = jnp.zeros(N)
    #         init_g = jnp.zeros(N)
    #     else:
    #         initializer = KNNInitializer(n_neighbors = 1, X_prev = X, Y_prev = Y, f_prev = out.f, g_prev = out.g)
    #         prob = linear_problem.LinearProblem(pointcloud.PointCloud(X, Y, epsilon = reg))
    #         init_f, init_g = initializer.init_fu(prob, True), initializer.init_gv(prob, True)
    #     t0 = time.perf_counter()
    #     out = sinkhorn_solve_jit(X, Y, reg, init_f, init_g)
    #     out = jax.block_until_ready(out)
    #     t1 = time.perf_counter()
    #     print(f"    Iteration {iter}: {t1 - t0} seconds, val = {jnp.mean(out.f) + jnp.mean(out.g)}")
    #     time_list[iter] = t1 - t0

    # print("\n")

if __name__ == "__main__":
    main()
