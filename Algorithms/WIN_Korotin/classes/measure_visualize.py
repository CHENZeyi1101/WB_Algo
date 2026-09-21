import numpy as np
import matplotlib.pyplot as plt

def dist_visualize(input_samples_dict, source_samples, oneplot=False, save=False, savefile=None, bins = 500):

    num_measures = len(input_samples_dict)
   
    if not oneplot:
        # if not oneplot, the plot the heatmap of source samples and each measure separately
        fig, axes = plt.subplots(1, num_measures + 1, figsize=(18, 6), sharey=False, facecolor='black')

        # Plot heatmap for source samples
        image = source_samples
        h_image = axes[0].hist2d(image[:, 0], image[:, 1], bins=bins, cmap='hot')
        axes[0].set_facecolor('black')
        axes[0].set_title("Source", color='white')
        axes[0].set_xlabel('X1', color='white')
        axes[0].set_ylabel('X2', color='white')
        axes[0].tick_params(axis='x', colors='white')
        axes[0].tick_params(axis='y', colors='white')

        # Plot heatmap for each measure in the dictionary
        for i in range(num_measures):
            image = input_samples_dict[f"measure_{i}"]

            # Plot heatmap for each measure
            h_image = axes[i + 1].hist2d(image[:, 0], image[:, 1], bins=bins, cmap='hot')

            # Set axis background to black
            axes[i + 1].set_facecolor('black')
            axes[i + 1].set_title(f"Measure_{i}", color='white')
            axes[i + 1].set_xlabel('X1', color='white')
            axes[i + 1].set_ylabel('X2', color='white')
            axes[i + 1].tick_params(axis='x', colors='white')
            axes[i + 1].tick_params(axis='y', colors='white')
            
        fig.suptitle("2D Histogram (Heatmap) of Different Functions", color='white')
        fig.colorbar(h_image[3], ax=axes, orientation='horizontal', label='Density', pad=0.1)  # Color bar for the density

        if save:
            plt.savefig(savefile, facecolor=fig.get_facecolor(), edgecolor='none')
        else:
            plt.show()

    else:
        # if oneplot, plot the scatter plot of source samples and each measure
        fig, ax = plt.subplots(figsize=(18, 6))
    
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'olive', 'cyan']
        markers = ['s', '^', 'D', 'v', 'h', '*', 'p', 'P']
        
        for i in range(num_measures):
            source = source_samples[0: 10000, :]
            image = input_samples_dict[f"measure_{i}"][0: 10000, :]
            ax.scatter(source[:, 0], source[:, 1], color='black', marker='x', s=10, label=f"Source_{i}" if i == 0 else "")
            ax.scatter(image[:, 0], image[:, 1], color=colors[i % len(colors)], marker=markers[i % len(markers)], s=15, label=f"Measure_{i}")
        
        ax.set_title("Scatter Plot of Different Functions")
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.legend()
        
        if save:
            plt.savefig(savefile)
        else:
            plt.show()