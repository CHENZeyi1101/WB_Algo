import sys
import os
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from .samplers_dim2 import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_kde_data(samples, bins=1000, xlim=None, ylim=None):
    x = samples[:, 0]
    y = samples[:, 1]
    kde = gaussian_kde([x, y])

    # If no limits provided, fall back to your old logic
    if xlim is None:
        t_x = (x.max() - x.min()) / 3
        x_min, x_max = x.min() - t_x, x.max() + t_x
    else:
        x_min, x_max = xlim

    if ylim is None:
        t_y = (y.max() - y.min()) / 3
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



def plot_2d_gm_pdf(gm_sampler, truncated_radius, grid_size=1000, plot_contour=False, plot_dirc=None, plot_name=None, title = None):
    """
    Plots the PDF of a Gaussian Mixture Model (GMM) over a 2D grid.
    
    """
    os.makedirs(plot_dirc, exist_ok=True)
    # Create a grid of points over the specified range
    xlim=(-truncated_radius, truncated_radius)
    ylim=(-truncated_radius, truncated_radius)
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    # Evaluate the GMM PDF at each point on the grid
    points = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T
    pdf_values = gm_sampler.pdf(points).reshape(grid_size, grid_size)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    if plot_contour:
        # Plot contour lines
        contour = ax.contourf(x_mesh, y_mesh, pdf_values, levels=50, cmap='hot')
        cbar = fig.colorbar(contour, ax=ax)
        cbar.ax.set_yticklabels([])      # Remove tick labels
        cbar.ax.set_yticks([])           # Remove tick marks
    else:
        # Plot heatmap
        heatmap = ax.imshow(pdf_values, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), 
                            origin='lower', cmap='hot', aspect='auto')
        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.ax.set_yticklabels([])      # Remove tick labels
        cbar.ax.set_yticks([])           # Remove tick marks

    # Set axis labels and title
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(title, fontsize =20)

    if plot_dirc:
        # save as "soource_measure_pdf.png"
        plt.savefig(f"{plot_dirc}/{plot_name}.png", dpi=200, bbox_inches='tight')
        # set the name to be "GMM_pdf.png"
        plt.close()
    else:
        plt.show()


def combine_images_row(image_paths, save_path=None, figsize=(18, 6)):
    """
    Combines multiple images into a single row.

    Parameters:
        image_paths: list of file paths to images
        save_path: optional path to save combined image
        figsize: size of the output figure
    """
    n = len(image_paths)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    # If only one image, axes is not a list
    if n == 1:
        axes = [axes]

    for ax, img_path in zip(axes, image_paths):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")  # remove axes for clean look

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



if __name__ == "__main__":
    dim = 2
    num_components = 5
    num_samples = 1000
    num_measures = 5
    truncated_radius = 150
    seed = 1009

    load_dir = "./Synthetic_Generation/dim2_data/samplers_info"
    plot_dir = "./Synthetic_Generation/dim2_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Load the samplers
    source_sampler = MixtureOfGaussians(dim)
    auxiliary_measure_sampler_set = characterize_auxiliary_sampler_set(dim, num_components)
    entropic_sampler = characterize_entropic_sampler(dim = dim, 
                                                     num_measures = num_measures, 
                                                     auxiliary_measure_sampler_set = auxiliary_measure_sampler_set, 
                                                     source_sampler = source_sampler,
                                                     truncated_radius = truncated_radius,
                                                     manual = True)
    
    source_sampler = load_sampler(load_dir, source_sampler, sampler_type="source")
    entropic_sampler = load_sampler(load_dir, entropic_sampler, sampler_type="entropic")
    auxiliary_measure_sampler_set = entropic_sampler.auxiliary_measure_sampler_set

    ### Generate and visualize samples from the source measure
    # Plot the PDF of the source measure since it is a GM
    plot_2d_gm_pdf(source_sampler, truncated_radius, grid_size=1000, plot_contour=False, plot_dirc=f"{plot_dir}/source_measure", plot_name="source_measure_pdf", title=r"PDF of $\bar{\mu}$")
    # Plot the KDE heatmap of the source measure samples
    source_samples = source_sampler.sample(num_samples, seed=seed, multiplication_factor=1)
    plot_2d_measures_kde(source_samples, truncated_radius, scatter=False, plot_dirc=f"{plot_dir}/source_measure", plot_name = "source_measure_kde", title=r"KDE of $\bar{\mu}$ samples")

    ### Generate and visualize samples from the auxiliary measures
    for idx, auxiliary_sampler in enumerate(auxiliary_measure_sampler_set):
        # Plot the PDF of the auxiliary measure since it is a GM
        plot_2d_gm_pdf(auxiliary_sampler, truncated_radius, grid_size=1000, plot_contour=False, plot_dirc=f"{plot_dir}/auxiliary_measures", plot_name=f"auxiliary_measure_{idx+1}_pdf", title=fr"PDF of $\varkappa_{{{idx+1}}}$")
        
    ### Generate and visualize samples from the input measures
    # Sample input measures
    input_measure_samples = entropic_sampler.sample(num_samples)
    for measure_index in range(len(input_measure_samples)):
        measure_samples = np.array(input_measure_samples[measure_index])
        # Plot the KDE for each input measure
        plot_2d_measures_kde(measure_samples, truncated_radius = None, scatter=False, plot_dirc=f"{plot_dir}/input_measures", plot_name=f"input_measure_{measure_index}_kde", title=fr"KDE of $\nu_{{{measure_index + 1}}}$ samples")


    ### Put together all plots into a single row
    image_paths_1 = [
        f"{plot_dir}/source_measure/source_measure_pdf.png",
        f"{plot_dir}/auxiliary_measures/auxiliary_measure_1_pdf.png",
        f"{plot_dir}/auxiliary_measures/auxiliary_measure_2_pdf.png",
        f"{plot_dir}/auxiliary_measures/auxiliary_measure_3_pdf.png",
        f"{plot_dir}/auxiliary_measures/auxiliary_measure_4_pdf.png",
        f"{plot_dir}/auxiliary_measures/auxiliary_measure_5_pdf.png"
    ]

    image_paths_2 = [
        f"{plot_dir}/source_measure/source_measure_pdf.png",
        f"{plot_dir}/input_measures/input_measure_0_kde.png",
        f"{plot_dir}/input_measures/input_measure_1_kde.png",
        f"{plot_dir}/input_measures/input_measure_2_kde.png",
        f"{plot_dir}/input_measures/input_measure_3_kde.png",
        f"{plot_dir}/input_measures/input_measure_4_kde.png"
    ]

    combine_images_row(image_paths_1, save_path=f"{plot_dir}/source_auxiliary_pdf_combined.png", figsize=(24, 6))
    combine_images_row(image_paths_2, save_path=f"{plot_dir}/source_input_kde_combined", figsize=(24, 6))



    