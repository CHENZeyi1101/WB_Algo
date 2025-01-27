from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

# Function to generate kernel density estimation (KDE) data for contour plotting
def get_kde_data(samples, bins=1000):
    x = samples[:, 0]
    y = samples[:, 1]
    # Perform KDE
    kde = gaussian_kde([x, y])
    # Generate a grid
    t = (samples[:, 0].max() - samples[:, 0].min()) / 3
    x_min, x_max = x.min() - t, x.max() + t
    y_min, y_max = y.min() - t, y.max() + t
    x_grid = np.linspace(x_min, x_max, bins)
    y_grid = np.linspace(y_min, y_max, bins)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    # Evaluate KDE on the grid
    kde_values = kde(np.vstack([x_mesh.ravel(), y_mesh.ravel()])).reshape(x_mesh.shape)
    return x_mesh, y_mesh, kde_values

def plot_2d_input_measure_kde_row(samples, measure_index, scatter=False, ax=None):
    # dimension of the samples
    dim = samples.shape[1]
    if dim > 2:
        # Perform PCA to reduce dimensions to 2D
        pca = PCA(n_components=2)
        samples = pca.fit_transform(samples)

    # If ax is not provided, create a new figure and axis (for standalone plot)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        # Use a black background
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
    else:
        # Set black background for the provided axis
        ax.set_facecolor('black')

    # Get KDE data
    x_mesh, y_mesh, kde_values = get_kde_data(samples)

    # Plot KDE as a contour plot
    h = ax.contourf(x_mesh, y_mesh, kde_values, levels=200, cmap='hot')

    # Overlay scatter plot if requested
    if scatter:
        ax.scatter(samples[:, 0], samples[:, 1], s=5, color='green', alpha=0.5)

    # Set axis limits to include all samples and contours
    t = (samples[:, 0].max() - samples[:, 0].min()) / 2
    ax.set_xlim(samples[:, 0].min() - t, samples[:, 0].max() + t)
    ax.set_ylim(samples[:, 1].min() - t, samples[:, 1].max() + t)

    # Set title and labels
    ax.set_title(f'Measure {measure_index}', color='black')
    ax.set_xlabel('X1', color='black')
    ax.set_ylabel('X2', color='black')

    # Adjust axis colors for visibility on black background
    ax.tick_params(colors='black')

    # Add a colorbar to the subplot
    cbar = plt.colorbar(h, ax=ax, orientation='vertical', pad=0.05)
    cbar.ax.yaxis.set_tick_params(color='black')
    cbar.outline.set_edgecolor('black')
    cbar.ax.tick_params(colors='black')


def plot_2d_input_measure_kde(samples, measure_index, scatter = False, plot_dirc = None):
    # dimension of the samples
    dim = samples.shape[1]
    if dim > 2:
        # Perform PCA to reduce dimensions to 2D
        pca = PCA(n_components=2)
        samples = pca.fit_transform(samples)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    # Use a black background
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    # Get KDE data
    x_mesh, y_mesh, kde_values = get_kde_data(samples)
    # Plot KDE as a contour plot
    h = ax.contourf(x_mesh, y_mesh, kde_values, levels=200, cmap='hot')
    # Overlay scatter plot if requested
    if scatter:
        ax.scatter(samples[:, 0], samples[:, 1], s=5, color='green', alpha=0.5)

    # Set axis limits to include all samples and contours
    t = (samples[:, 0].max() - samples[:, 0].min()) / 2
    ax.set_xlim(samples[:, 0].min() - t, samples[:, 0].max() + t)
    ax.set_ylim(samples[:, 1].min() - t, samples[:, 1].max() + t)

    # Set title and labels
    ax.set_title('Samples', color='white')
    ax.set_xlabel('X1', color='white')
    ax.set_ylabel('X2', color='white')

    # Adjust axis colors for visibility on black background
    ax.tick_params(colors='white')
    # Add a colorbar
    cbar = fig.colorbar(h, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    
    if plot_dirc:
        os.makedirs(plot_dirc, exist_ok=True)
        # Save the figure
        plt.savefig(f"{plot_dirc}/measure_{measure_index}_samples.png", facecolor='black')
        # Close the plot to free memory
        plt.close()
    else:
        plt.show()

def plot_2d_source_measures_kde(samples, scatter = False, plot_dirc = None):
    # dimension of the samples
    dim = samples.shape[1]
    if dim > 2:
        # Perform PCA to reduce dimensions to 2D
        pca = PCA(n_components=2)
        samples = pca.fit_transform(samples)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    # Use a black background
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    # Set axis limits to include all samples and contours

    t = (samples[:, 0].max() - samples[:, 0].min()) / 2
    ax.set_xlim(samples[:, 0].min() - t, samples[:, 0].max() + t)
    ax.set_ylim(samples[:, 1].min() - t, samples[:, 1].max() + t)
    # Get KDE data
    x_mesh, y_mesh, kde_values = get_kde_data(samples)
    # Plot KDE as a contour plot
    h = ax.contourf(x_mesh, y_mesh, kde_values, levels=200, cmap='hot')
    # Overlay scatter plot if requested
    if scatter:
        ax.scatter(samples[:, 0], samples[:, 1], s=5, color='green', alpha=0.5)

    # Set title and labels
    ax.set_title('Samples', color='white')
    ax.set_xlabel('X1', color='white')
    ax.set_ylabel('X2', color='white')

    # Adjust axis colors for visibility on black background
    ax.tick_params(colors='white')
    # Add a colorbar
    cbar = fig.colorbar(h, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    
    if plot_dirc:
        os.makedirs(plot_dirc, exist_ok=True)
        # Save the figure
        plt.savefig(f"{plot_dirc}/source_measure.png", facecolor='black')
        # Close the plot to free memory
        plt.close()
    else:
        plt.show()

def plot_2d_compare_with_source_kde(source_samples, accepted, iter, scatter = False, plot_dirc = None):
    dim = source_samples.shape[1]
    if dim > 2:
        # Perform PCA to reduce dimensions to 2D
        pca = PCA(n_components=2)
        source_samples = pca.fit_transform(source_samples)
        accepted = pca.transform(accepted)

    # Create two subplots: one for G_samples and one for source_samples
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Use a black background
    fig.patch.set_facecolor('black')
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')

    # Plot source_samples first to determine the axis limits
    x_mesh, y_mesh, kde_values = get_kde_data(source_samples)
    h2 = ax2.contourf(x_mesh, y_mesh, kde_values, levels = 200, cmap='hot')  # Contour plot for KDE
    if scatter:
        ax2.scatter(source_samples[:, 0], source_samples[:, 1], s=5, color='green', alpha=0.5)  # Overlay scatter plot
    # Set axis limits to include all samples and contours
    t = (source_samples[:, 0].max() - source_samples[:, 0].min()) / 2
    ax2.set_xlim(source_samples[:, 0].min() - t, source_samples[:, 0].max() + t)
    ax2.set_ylim(source_samples[:, 1].min() - t, source_samples[:, 1].max() + t)
    
    ax2.set_title('Source_samples', color='white')
    ax2.set_xlabel('X1', color='white')
    ax2.set_ylabel('X2', color='white')

    # Adjust axis colors for the second plot
    ax2.tick_params(colors='white')

    # Add a colorbar for source_samples
    cbar2 = fig.colorbar(h2, ax=ax2)
    cbar2.ax.yaxis.set_tick_params(color='white')
    cbar2.outline.set_edgecolor('white')

    # Plot G_samples as a 2D contour plot in the first subplot, using the same axis limits
    x_mesh, y_mesh, kde_values = get_kde_data(accepted)
     # Contour plot for KDE
    h1 = ax1.contourf(x_mesh, y_mesh, kde_values, levels = 200, cmap='hot') 
    if scatter:
        ax1.scatter(accepted[:, 0], accepted[:, 1], s=5, color='green', alpha=0.5)  # Overlay scatter plot
    # Set axis limits to include all samples and contours
    t = (source_samples[:, 0].max() - source_samples[:, 0].min()) / 2
    ax1.set_xlim(source_samples[:, 0].min() - t, source_samples[:, 0].max() + t)
    ax1.set_ylim(source_samples[:, 1].min() - t, source_samples[:, 1].max() + t)
    ax1.set_title('G_samples', color='white')  # White text for contrast on black background
    ax1.set_xlabel('X1', color='white')
    ax1.set_ylabel('X2', color='white')

    # Adjust axis colors to be visible on the black background
    ax1.tick_params(colors='white')

    # Add a colorbar for G_samples
    cbar1 = fig.colorbar(h1, ax=ax1)
    cbar1.ax.yaxis.set_tick_params(color='white')  # Set color of colorbar ticks
    cbar1.outline.set_edgecolor('white')  # Set color of colorbar outline

    if plot_dirc:
        os.makedirs(plot_dirc, exist_ok=True)
        # Save the figure
        plt.savefig(f"{plot_dirc}/iteration_{iter}_samples.png", facecolor='black')
        # Close the plot to free memory
        plt.close()
    else:
        plt.show()


def plot_2d_gmm_pdf(gmm_sampler, truncated_radius, grid_size=1000, plot_contour=False, save_path=None):
    """
    Plots the PDF of a Gaussian Mixture Model (GMM) over a 2D grid.
    
    Args:
        gmm: GaussianMixture object.
        grid_size: Number of grid points along each axis.
        xlim: Tuple (xmin, xmax) defining the x-axis range.
        ylim: Tuple (ymin, ymax) defining the y-axis range.
        plot_contour: If True, plot contour lines. Otherwise, plot a heatmap.
    """
    # Create a grid of points over the specified range
    xlim=(-truncated_radius, truncated_radius)
    ylim=(-truncated_radius, truncated_radius)
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    # Evaluate the GMM PDF at each point on the grid
    points = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T
    pdf_values = gmm_sampler.pdf(points).reshape(grid_size, grid_size)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    if plot_contour:
        # Plot contour lines
        contour = ax.contourf(x_mesh, y_mesh, pdf_values, levels=50, cmap='hot')
        fig.colorbar(contour, ax=ax, label="PDF Value")
    else:
        # Plot heatmap
        heatmap = ax.imshow(pdf_values, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), 
                            origin='lower', cmap='hot', aspect='auto')
        fig.colorbar(heatmap, ax=ax, label="PDF Value")

    # Set axis labels and title
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('GMM PDF')

    if save_path:
        plt.savefig(save_path)
        # set the name to be "GMM_pdf.png"
        plt.close()
    else:
        plt.show()




