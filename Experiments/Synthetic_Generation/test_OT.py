import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import ot
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import numpy as np
import time
import torch

from geomloss import SamplesLoss

def main():
    num_of_rep = 1
    N = 20
    rs = np.random.RandomState(seed = 2000)
    X = rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[2, 1], [1, 2]]) * 100, size = N)
    Y = rs.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1, 0], [0, 1]]) * 100, size = N)
    dist = ot.dist(X, Y)
    a = np.ones(N) / N
    b = np.ones(N) / N
    reg = 10
    verbose = False

    # Unregularized OT via POT
    t0 = time.perf_counter()
    for _ in range(num_of_rep):
        loss_pot_unreg, log_pot_unreg = ot.emd2(a, b, dist, log = True)
    t1 = time.perf_counter()
    print(f"POT unregularized OT: time ({num_of_rep} reps) = {t1 - t0} seconds")

    # # Log-domain Sinkhorn OT via POT
    # t0 = time.perf_counter()
    # for _ in range(num_of_rep):
    #     loss_ori, log_pot_sinkhorn_log = ot.sinkhorn2(a, b, dist, reg = reg, method = "sinkhorn_log", log = True, verbose = verbose)
    # t1 = time.perf_counter()
    # print(f"POT log-domain Sinkhorn OT: time ({num_of_rep} reps) = {t1 - t0} seconds")

    # f_ref = reg * np.log(log_pot_sinkhorn_log['u'])
    # g_ref = reg * np.log(log_pot_sinkhorn_log['v'])
    # g_ref += np.mean(f_ref)
    # f_ref -= np.mean(f_ref)
    # loss_ref = np.sum(f_ref * a) + np.sum(g_ref * b)

    # Sinkhorn OT via POT
    t0 = time.perf_counter()
    for _ in range(num_of_rep):
        _, log_pot_sinkhorn = ot.sinkhorn2(a, b, dist, reg = reg, method = "sinkhorn", log = True, verbose = verbose)
    t1 = time.perf_counter()
    f_sinkhorn = reg * np.log(log_pot_sinkhorn['u'])
    g_sinkhorn = reg * np.log(log_pot_sinkhorn['v'])
    g_sinkhorn += np.mean(f_sinkhorn)
    f_sinkhorn -= np.mean(f_sinkhorn)
    loss_pot_sinkhorn = np.sum(f_sinkhorn * a) + np.sum(g_sinkhorn * b)

    print(f"POT Sinkhorn OT: time ({num_of_rep} reps) = {t1 - t0} seconds")
    # print(f"POT Sinkhorn OT: loss diff. = {np.abs(loss_pot_sinkhorn - loss_ref)}")
    # print(f"POT Sinkhorn OT: f diff. = {np.linalg.norm(f_sinkhorn - f_ref)}")
    # print(f"POT Sinkhorn OT: g diff. = {np.linalg.norm(g_sinkhorn - g_ref)}")

    # Sinkhorn OT via OTT with log-sum-exp
    t0 = time.perf_counter()
    for _ in range(num_of_rep):
        geom = pointcloud.PointCloud(X, Y, epsilon = reg) # set the epsilon parameter for the entropic regularization
        ott_problem = linear_problem.LinearProblem(geom, a, b) # uniform weights
        solver = sinkhorn.Sinkhorn(lse_mode = True, max_iterations=1000)
        out = solver(ott_problem)
    t1 = time.perf_counter()
    f_ott_lse = out.f
    g_ott_lse = out.g
    g_ott_lse += jnp.mean(f_ott_lse)
    f_ott_lse -= jnp.mean(f_ott_lse)
    loss_ott_lse = jnp.sum(f_ott_lse * a) + jnp.sum(g_ott_lse * b)
    loss_ott_lse_ot = out.reg_ot_cost
    loss_ott_lse_primal = out.primal_cost
    loss_ott_lse_dual = out.dual_cost

    print(f"OTT Sinkhorn OT with LSE: time ({num_of_rep} reps) = {t1 - t0} seconds")
    print(f"OTT Sinkhorn OT with LSE: loss diff. = {np.abs(loss_ott_lse - loss_pot_sinkhorn)}")
    print(f"OTT Sinkhorn OT with LSE: f diff. = {np.linalg.norm(f_ott_lse - f_sinkhorn)}")
    print(f"OTT Sinkhorn OT with LSE: g diff. = {np.linalg.norm(g_ott_lse - g_sinkhorn)}")

    # Sinkhorn OT via geomloss
    t0 = time.perf_counter()
    Loss = SamplesLoss(loss = "sinkhorn", p = 2, blur = (reg / 2) ** 0.5, scaling = 1 - 10 ** -3, debias = False, potentials = True, verbose = True)
    out = Loss(torch.tensor(a), torch.tensor(X), torch.tensor(b), torch.tensor(Y))
    f_geomloss = np.squeeze(out[0].numpy()) * 2
    g_geomloss = np.squeeze(out[1].numpy()) * 2
    g_geomloss += np.mean(f_geomloss)
    f_geomloss -= np.mean(f_geomloss)
    t1 = time.perf_counter()
    loss_geomloss = np.sum(f_geomloss * a) + np.sum(g_geomloss * b)

    print(f"geomloss Sinkhorn OT: time ({num_of_rep} reps) = {t1 - t0} seconds")
    print(f"geomloss Sinkhorn OT: loss diff. = {np.abs(loss_geomloss - loss_pot_sinkhorn)}")
    print(f"geomloss Sinkhorn OT: f diff. = {np.linalg.norm(f_geomloss - f_ott_lse)}")
    print(f"geomloss Sinkhorn OT: g diff. = {np.linalg.norm(g_geomloss - g_ott_lse)}")


    print(loss_pot_sinkhorn)
    print(loss_ott_lse)
    print(loss_geomloss)

    print(f_ott_lse)
    print(f_geomloss)
    print(g_ott_lse)
    print(g_geomloss)

    # print('\n\n\n')
    # print(f"sample size = {N}, regularization = {reg}")
    # print(f"    Unregularized OT = {loss_pot_unreg}")
    # print(f"    OTT Sinkhorn (with regularization) = {loss_ott_lse_ot}")
    # print(f"    OTT OT primal (withtou regularization) = {loss_ott_lse_primal}")
    # print(f"    OTT OT dual (without regularization) = {loss_ott_lse_dual}")


if __name__ == "__main__":
    main()
