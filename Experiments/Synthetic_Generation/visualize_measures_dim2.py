import sys
import os
import numpy as np
from tqdm import tqdm
import math
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from Experiments.Synthetic_Generation.input_generate_entropic import entropic_input_sampler
from Experiments.CSV_read import *



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
    bins = 1000,
    plot_radius=None,         # NEW: to match PDF box
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
    if plot_radius is not None:
        xlim = (-plot_radius, plot_radius)
        ylim = (-plot_radius, plot_radius)
    else:
        xlim = ylim = None

    # Get KDE grid using those exact limits
    x_mesh, y_mesh, kde_values = get_kde_data(samples, bins=bins, xlim=xlim, ylim=ylim)

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

    if plot_radius is not None:
        ax.set_xlim(-plot_radius, plot_radius)
        ax.set_ylim(-plot_radius, plot_radius)

    if plot_dirc:
        os.makedirs(plot_dirc, exist_ok=True)
        plt.savefig(f"{plot_dirc}/{plot_name}.png", dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def plot_2d_gm_pdf(gm_sampler, plot_radius, grid_size=1000, plot_contour=False, plot_dirc=None, plot_name=None, title = None):
    """
    Plots the PDF of a Gaussian Mixture Model (GMM) over a 2D grid.
    
    """
    os.makedirs(plot_dirc, exist_ok=True)
    # Create a grid of points over the specified range
    xlim=(-plot_radius, plot_radius)
    ylim=(-plot_radius, plot_radius)
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

def combine_images_2rows(image_paths, save_path=None, figsize=(18, 12)):
    """
    Combines multiple images into two rows.

    Parameters:
        image_paths: list of file paths to images
        save_path: optional path to save combined image
        figsize: size of the output figure
    """
    n = len(image_paths)
    cols = math.ceil(n / 2)
    fig, axes = plt.subplots(2, cols, figsize=figsize)

    # If only one image, axes is not a list
    if n == 1:
        axes = [axes]

    for ax, img_path in zip(np.reshape(axes, (cols * 2,))[:n], image_paths):
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
    from pathlib import Path
    import json, os
    
    Cfg_PATH = Path(__file__).parent / "cfg.json"
    with open(Cfg_PATH, "r") as f:
        cfg_dict = json.load(f)

    params = cfg_dict["params_synthetic_generation_dim2"]
    dim = params["dim"]
    instance_identifier = params["instance_identifier"]
    instance_dir = f"{cfg_dict['data_dir']}/Synthetic_Generation/dim{dim}_data/Instance{instance_identifier}"

    entropic_sampler = entropic_input_sampler.load_from_file(load_dir = f"{instance_dir}/samplers_info")
    source_sampler = entropic_sampler.source_sampler
    auxiliary_measure_sampler_set = entropic_sampler.auxiliary_measure_sampler_set

    plot_radius = 150
    num_samples_to_plot = 100000

    plot_dir = f"../../WB_data/Synthetic_Generation/dim{dim}_plots/Instance{instance_identifier}"
    os.makedirs(plot_dir, exist_ok=True)

    ### Generate and visualize samples from the source measure
    # Plot the PDF of the source measure since it is a GM
    plot_2d_gm_pdf(source_sampler, plot_radius, grid_size=1000, plot_contour=False, plot_dirc=f"{plot_dir}/source_measure", plot_name="source_measure_pdf", title=r"PDF of $\bar{\mu}$")
    print("Source measure PDF plotted.")
    # Plot the KDE heatmap of the source measure samples
    source_samples = source_sampler.sample(num_samples_to_plot)
    plot_2d_measures_kde(source_samples, plot_radius, scatter=False, plot_dirc=f"{plot_dir}/source_measure", plot_name = "source_measure_kde", title=r"KDE of $\bar{\mu}$ samples")
    print("Source measure samples KDE plotted.")

    for idx in range(entropic_sampler.num_measures):
        auxiliary_measure_sampler = auxiliary_measure_sampler_set[idx]
        plot_2d_gm_pdf(auxiliary_measure_sampler, plot_radius, grid_size=1000, plot_contour=False, plot_dirc=f"{plot_dir}/auxiliary_measures", plot_name=f"auxiliary_measure_{idx+1}_pdf", title=fr"PDF of $\varkappa_{{{idx+1}}}$")
        print(f"Auxiliary measure {idx+1} PDF plotted.")
        
    ### Generate and visualize samples from the input measures
    # Sample input measures
    
    input_measure_samples = [np.zeros((num_samples_to_plot, dim)) for _ in range(entropic_sampler.num_measures)]
    component_map_pushforwards = [np.zeros((num_samples_to_plot, dim)) for _ in range(entropic_sampler.num_measures * 2)]

    for i in tqdm(range(num_samples_to_plot), desc="Computing pushforwards"):
        OT_map_pushforward, component_map_pushforward = entropic_sampler.generate_input_measure_sample(source_samples[i])

        for k in range(entropic_sampler.num_measures):
            input_measure_samples[k][i,:] = OT_map_pushforward[k]

        for tilde_k in range(entropic_sampler.num_measures * 2):
            component_map_pushforwards[tilde_k][i,:] = component_map_pushforward[tilde_k]

    for measure_index in range(entropic_sampler.num_measures):
        measure_samples = input_measure_samples[measure_index]
        # Plot the KDE for each input measure
        plot_2d_measures_kde(measure_samples, bins = 400, plot_radius = plot_radius, scatter=False, plot_dirc=f"{plot_dir}/input_measures", plot_name=f"input_measure_{measure_index}_kde", title=fr"KDE of $\nu_{{{measure_index + 1}}}$ samples")
        plot_2d_measures_kde(measure_samples, bins = 400, plot_radius = plot_radius, scatter=False, plot_dirc=f"{plot_dir}/input_measures", plot_name=f"input_measure_{measure_index}_kde", title=fr"KDE of $\nu_{{{measure_index + 1}}}$ samples")
        print(f"Input measure {measure_index} KDE plotted.")
    
    for tilde_k in range(entropic_sampler.num_measures * 2):
        pushforward_samples = component_map_pushforwards[tilde_k]

        plot_name_suffix = "plus" if (tilde_k % 2 == 0) else "minus"
        plot_2d_measures_kde(pushforward_samples, bins = 400, plot_radius = plot_radius, scatter=False, plot_dirc=f"{plot_dir}/component_pushforwards", plot_name=f"component_{tilde_k // 2}_{plot_name_suffix}_kde", title=fr"KDE of pushforward {tilde_k}")
        print(f"Pushforward {tilde_k} KDE plotted.")


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


    image_paths_3 = [
        f"{plot_dir}/component_pushforwards/component_0_plus_kde.png",
        f"{plot_dir}/component_pushforwards/component_1_plus_kde.png",
        f"{plot_dir}/component_pushforwards/component_2_plus_kde.png",
        f"{plot_dir}/component_pushforwards/component_3_plus_kde.png",
        f"{plot_dir}/component_pushforwards/component_4_plus_kde.png",
        f"{plot_dir}/component_pushforwards/component_0_minus_kde.png",
        f"{plot_dir}/component_pushforwards/component_1_minus_kde.png",
        f"{plot_dir}/component_pushforwards/component_2_minus_kde.png",
        f"{plot_dir}/component_pushforwards/component_3_minus_kde.png",
        f"{plot_dir}/component_pushforwards/component_4_minus_kde.png"
    ]

    combine_images_row(image_paths_1, save_path=f"{plot_dir}/source_auxiliary_pdf_combined.png", figsize=(24, 6))
    print("Combined source and auxiliary measure PDFs.")
    combine_images_row(image_paths_2, save_path=f"{plot_dir}/source_input_kde_combined.png", figsize=(24, 6))
    print("Combined source and input measure KDEs.")

    combine_images_2rows(image_paths_3, save_path=f"{plot_dir}/component_pushforwards_kde_combined.png", figsize=(24, 12))
    print("Combined component pushforwards KDEs.")