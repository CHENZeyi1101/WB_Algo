import numpy as np
from w2 import BFM

from scipy.fftpack import dctn, idctn
import matplotlib.pyplot as plt
import seaborn as sns

import ot
import ot.plot
import os
from tqdm import tqdm

from time import time


# ---------------------------------------------------------
# Functions for evaluating V-value and W2 distance to barycenter
# ---------------------------------------------------------
def W2_pot(X, Y): 
    r'''
    Compute the squared Wasserstein-2 distance between two empirical measures (using the POT library)
    '''
    M = ot.dist(X, Y)
    a, b = np.ones((X.shape[0],)) / X.shape[0], np.ones((Y.shape[0],)) / Y.shape[0]
    W2_sq = ot.emd2(a, b, M, numItermax=1e6)
    return W2_sq

def V_value_compute(self, bary_samples, input_samples_collection: dict):
        '''
        bary_samples denotes the samples from the true/approximated barycenter measure
        input_samples_collection is a dictionary with k keys, each key corresponds to the samples from the k-th input measure.
        '''
        V_value = 0
        for measure_index in tqdm(range(self.num_measures), desc = "V-value computation"):
            input_samples = np.array(input_samples_collection[measure_index])
            V_value += W2_pot(input_samples, bary_samples)
        V_value /= self.num_measures
        return V_value
    
def W2_to_bary_compute(self, bary_samples, generated_samples):
    '''
    Compute the (empirical) Wasserstein distance between the generated samples from the G-mapping
    and the barycenter samples at each iteration;
    '''
    W2_sq = W2_pot(generated_samples, bary_samples)
    return W2_sq

# ---------------------------------------------------------
# Functions for Fréchet mean computation (taken from Kim et al. (2025))
# ---------------------------------------------------------


# Initialize Fourier kernel
def initialize_kernel(n1, n2):
    xx, yy = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
    kernel = 2*n1*n1*(1-np.cos(xx)) + 2*n2*n2*(1-np.cos(yy))
    kernel[0,0] = 1     # to avoid dividing by zero
    return kernel

# 2d DCT
def dct2(a):
    return dctn(a, norm='ortho')

# 2d IDCT
def idct2(a):
    return idctn(a, norm='ortho')

# Update phi as
#       ϕ ← ϕ + σ Δ⁻¹(ρ − ν)
# and return the error
#       ∫(−Δ)⁻¹(ρ−ν) (ρ−ν)
# Modifies phi and rho
def update_potential(phi, rho, nu, kernel, sigma):
    n1, n2 = nu.shape

    rho -= nu
    workspace = dct2(rho) / kernel
    workspace[0,0] = 0
    workspace = idct2(workspace)

    phi += sigma * workspace
    h1 = np.sum(workspace * rho) / (n1*n2)

    return h1

def grad_norm(rho):
    n2, n1 = np.shape(rho)
    kernel = initialize_kernel(n1,n2)
    workspace = dct2(rho) / kernel
    workspace[0,0] = 0
    workspace = idct2(workspace)

    return np.sum(workspace * rho) / (n1*n2)

def compute_w2(phi, psi, mu, nu):
  n1, n2 = mu.shape
  x, y = np.meshgrid(np.linspace(0,np.pi,n1,False), np.linspace(0,np.pi,n2,False))
  return np.sum(0.5 * (x*x+y*y) * (mu + nu) - nu*phi - mu*psi)/(n1*n2)

def compute_ot(phi, psi, bf,mu, nu, sigma, inner ):
    n2, n1 = np.shape(phi)
    kernel = initialize_kernel(n1, n2)
    rho = np.copy(mu)

    x, y = np.meshgrid(np.linspace(0.5/n1,1-0.5/n1,n1),
                    np.linspace(0.5/n2,1-0.5/n2,n2))
    id = 1/2 * (x**2 + y**2)

    old_w2 = compute_w2(phi, psi, mu, nu)
    for k in range(inner):
        rho = np.zeros((n2,n1))
        bf.pushforward(rho, phi, nu)
        gradSq = update_potential(psi, rho, mu, kernel, sigma)

        bf.ctransform(phi, psi)
        bf.ctransform(psi, phi)

        bf.ctransform(psi, phi)
        bf.ctransform(phi, psi)

        new_w2 = compute_w2(phi, psi, mu, nu)

    return new_w2

# ---------------------------------------------------------
# Sample - grid transforms and Fréchet mean computation
# ---------------------------------------------------------

def infer_domain_from_samples(samples_dict, margin=0.05):
    all_samples = np.vstack(list(samples_dict.values()))
    xmin, ymin = all_samples.min(axis=0)
    xmax, ymax = all_samples.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    return (xmin - margin*dx, xmax + margin*dx), (ymin - margin*dy, ymax + margin*dy)

def samples_to_density(samples, n1, n2, xlim, ylim, eps=1e-12):
    H, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 1],
        bins=[n1, n2],
        range=[xlim, ylim],
        density=False
    )
    rd = H.T.astype(float)
    rd += eps
    rd /= rd.sum()
    return rd

def sample_from_density(rd, n, xlim, ylim, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n2, n1 = rd.shape
    p = rd.ravel() / rd.sum()

    idx = rng.choice(n1*n2, size=n, p=p)
    iy, ix = np.divmod(idx, n1)

    dx = (xlim[1] - xlim[0]) / n1
    dy = (ylim[1] - ylim[0]) / n2

    xs = xlim[0] + (ix + rng.random(n)) * dx
    ys = ylim[0] + (iy + rng.random(n)) * dy
    return np.column_stack([xs, ys])


def frechet_mean(
    input_measure_sampler,
    bary_sampler,
    num_samples,
    n_iter,
    name,
    n1=1024,
    n2=1024,
    inner=1,
    plot_option=True,
    save_option=True,
    return_option=False,
    generate_samples_fn=None,
):
    """
    Fréchet mean:
    - input measures: samples → densities
    - barycenter: grid density
    - OT updates: identical to original code
    """

    # --------------------------------------------------
    # Step 1: sample input measures
    # --------------------------------------------------
    input_samples_collection = input_measure_sampler.sample(num_samples)
    K = input_measure_sampler.num_measures

    # --------------------------------------------------
    # Step 2: infer domain
    # --------------------------------------------------
    xlim, ylim = infer_domain_from_samples(input_samples_collection)

    # --------------------------------------------------
    # Step 3: approximate input densities
    # --------------------------------------------------
    dists = [
        samples_to_density(
            input_samples_collection[k],
            n1=n1, n2=n2,
            xlim=xlim, ylim=ylim
        )
        for k in range(K)
    ]

    # --------------------------------------------------
    # Step 4: grid + initialization (UNCHANGED LOGIC)
    # --------------------------------------------------
    x, y = np.meshgrid(
        np.linspace(xlim[0] + 0.5*(xlim[1]-xlim[0])/n1,
                    xlim[1] - 0.5*(xlim[1]-xlim[0])/n1, n1),
        np.linspace(ylim[0] + 0.5*(ylim[1]-ylim[0])/n2,
                    ylim[1] - 0.5*(ylim[1]-ylim[0])/n2, n2),
    )

    id = 0.5 * (x**2 + y**2)
    id -= id.mean()

    rd = dists[0].copy()
    sigma = 5e-2 * np.ones(K)
    w2_list = np.zeros(K)

    phi = np.array([id] * K)
    psi = np.array([id] * K)

    bf = BFM(n1, n2, rd)

    # --------------------------------------------------
    # Tracking
    # --------------------------------------------------
    V_value_path = []
    W2_to_bary_path = []

    tic = time()

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    for i in range(n_iter):
        prev_psi = psi

        for j in range(K):
            new_w2 = compute_ot(
                phi[j], psi[j], bf,
                rd, dists[j], sigma[j],
                inner=inner
            )
            if new_w2 < w2_list[j]:
                sigma[j] *= 0.99
            w2_list[j] = new_w2

        lr = np.exp(-(i+1)/n_iter)
        rho = np.ones_like(rd)
        bf.pushforward(
            rho,
            id + lr*(np.mean(prev_psi, axis=0) - id),
            rd
        )
        rd = rho / rho.sum()

        # --------------------------------------------------
        # Sample-level evaluation
        # --------------------------------------------------
        bary_samples = sample_from_density(
            rd, num_samples, xlim, ylim
        )

        V_value = 0.0
        for k in tqdm(range(K), desc="V-value", leave=False):
            V_value += W2_pot(
                input_samples_collection[k],
                bary_samples
            )
        V_value /= K
        V_value_path.append(V_value)

        if generate_samples_fn is None:
            W2_sq = np.nan
        else:
            gen_samples = generate_samples_fn(
                bary_samples=bary_samples,
                iter_idx=i
            )
            W2_sq = W2_pot(gen_samples, bary_samples)

        W2_to_bary_path.append(W2_sq)

        if (i+1) % 50 == 0:
            print(
                f"Iter {i+1}: "
                f"V = {V_value:.4e}, "
                f"W2_to_bary = {W2_sq:.4e}"
            )

    toc = time()

    if plot_option:
        plotting(dists, rd, name, save_option=save_option)

    if return_option:
        return {
            "bary_density": rd,
            "V_value_path": np.array(V_value_path),
            "W2_to_bary_path": np.array(W2_to_bary_path),
            "runtime": toc - tic,
        }

