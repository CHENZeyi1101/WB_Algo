import sys
import os
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .posterior_sampling import *

def get_kde_data(samples, bins=1000, xlim=None, ylim=None):
    x = samples[:, 0]
    y = samples[:, 1]
    kde = gaussian_kde([x, y])

    # If no limits provided, fall back to your old logic
    if xlim is None:
        t_x = (x.max() - x.min()) / 2
        x_min, x_max = x.min() - t_x, x.max() + t_x
    else:
        x_min, x_max = xlim

    if ylim is None:
        t_y = (y.max() - y.min()) / 2
        y_min, y_max = y.min() - t_y, y.max() + t_y
    else:
        y_min, y_max = ylim

    x_grid = np.linspace(x_min, x_max, bins)
    y_grid = np.linspace(y_min, y_max, bins)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    kde_values = kde(np.vstack([x_mesh.ravel(), y_mesh.ravel()])).reshape(x_mesh.shape)
    return x_mesh, y_mesh, kde_values


def plot_2d_measures_kde(
    samples,
    truncated_radius=None,         # NEW: to match PDF box
    scatter=False,
    plot_dirc=None,
    plot_name=None,
    title=None
):
    dim = samples.shape[1]
    if dim > 2:
        pca = PCA(n_components=2)
        samples = pca.fit_transform(samples)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- set limits to match PDF plot if radius is given ---
    if truncated_radius is not None:
        xlim = (-truncated_radius, truncated_radius)
        ylim = (-truncated_radius, truncated_radius)
    else:
        xlim = ylim = None

    # Get KDE grid using those exact limits
    x_mesh, y_mesh, kde_values = get_kde_data(samples, bins=1000, xlim=xlim, ylim=ylim)

    h = ax.contourf(x_mesh, y_mesh, kde_values, levels=200, cmap="hot")

    if scatter:
        ax.scatter(samples[:, 0], samples[:, 1], s=5, color="green", alpha=0.5)

    if title:
        ax.set_title(title, fontsize=20)

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")

    # Colorbar: label “KDE”, no numbers
    cbar = fig.colorbar(h, ax=ax)
    cbar.ax.set_yticks([])
    cbar.ax.set_yticklabels([])

    if truncated_radius is not None:
        ax.set_xlim(-truncated_radius, truncated_radius)
        ax.set_ylim(-truncated_radius, truncated_radius)

    if plot_dirc:
        os.makedirs(plot_dirc, exist_ok=True)
        plt.savefig(f"{plot_dirc}/{plot_name}.png", dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    n_splits = 5  # ensure matches training splits
    DATA_DIR = os.path.dirname(__file__)
    print("Current working directory:", DATA_DIR)
    MODEL_DIR = os.path.join(DATA_DIR, "models_meta")
    PLOT_DIR = os.path.join(DATA_DIR, "plots")
    multiplication_factor = 10

    # sample from full model
    print("Sampling from full model...")
    full_samples = sample_from_meta(os.path.join(MODEL_DIR, "model_total.meta.pkl"), num_chains=1, num_samples=2000, save_samples=False) * multiplication_factor
    print("Full model samples shape:", full_samples.shape)

    # plot full samples
    plot_2d_measures_kde(full_samples.T, 
                         truncated_radius=None, 
                         scatter=False, 
                         plot_dirc=PLOT_DIR, 
                         plot_name="full_model_samples_kde", 
                         title="Full Model Samples KDE")

    # sample from split models
    for i in range(n_splits):
        print(f"Sampling from split model {i}...")
        split_samples = sample_from_meta(os.path.join(MODEL_DIR, f"model_split_{i}.meta.pkl"), num_chains=1, num_samples=2000, save_samples=False) * multiplication_factor

        # plot split samples
        plot_2d_measures_kde(split_samples.reshape(-1, split_samples.shape[-1]), 
                             truncated_radius=None, 
                             scatter=False, 
                             plot_dirc=PLOT_DIR, 
                             plot_name=f"split_model_{i}_samples_kde", 
                             title=f"Split Model {i} Samples KDE")
