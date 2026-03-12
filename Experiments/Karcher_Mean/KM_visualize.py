import matplotlib.pyplot as plt
import math
import os, json
import numpy as np
from tqdm import tqdm

def visualize_iterations(all_bary_samples: list, n_cols=5, eps=None, plot_disks:dict=None,
                         xlim=None, ylim=None,
                         plot_dir=None, plot_name=None):

    n_iters = len(all_bary_samples)
    n_rows = math.ceil(n_iters / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i in tqdm(range(n_iters), desc="Visualizing iterations"):
        ax = axes[i]
        if plot_disks is not None:
            assert eps is not None, "eps must be provided when plot_disks is given"
            for label, (centers, color, fill) in plot_disks.items():
                for center in centers:
                    ax.add_artist(plt.Circle(center, eps, color=color, fill=fill, linestyle='--'))

        ax.scatter(all_bary_samples[i][:, 0], all_bary_samples[i][:, 1],
                   s=5, color='black', alpha=0.5, marker='x', label='Approximated samples' if i == 0 else "")


        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_title(f'Iteration {i + 1}')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.grid(True)
        ax.set_aspect('equal')

    # Hide any unused axes in the last row
    for j in range(n_iters, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=12, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Approximated samples across iterations', fontsize=16, y=1.01)
    plt.tight_layout()

    if plot_dir is not None and plot_name is not None:
        plt.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    data_dir = "../../WB_data"
    instance_dir = f"{data_dir}/Karcher_Mean"
    # instance_identifier = "TrialA"
    # instance_identifier = "TrialB"
    # instance_identifier = "TrialC"
    instance_identifier = "TrialD"
    outputs_dir = f"{instance_dir}/KM_Example_outputs/Instance_{instance_identifier}"
    assert os.path.exists(outputs_dir), f"Outputs directory {outputs_dir} does not exist."

    with open(f"{outputs_dir}/instance_info.json", 'r') as json_file:
        cfg_dict = json.load(json_file)
    a = cfg_dict["a"]
    M = cfg_dict["M"]
    eps = cfg_dict["eps"]

    # fetch approximated barycenter samples across iterations
    G_samples_dir = os.path.join(outputs_dir, "G_samples")
    n_iters = len(os.listdir(G_samples_dir))
    
    all_bary_samples = []
    for i in range(n_iters):
        with open(f'{G_samples_dir}/G_samples_iter{i}.json', 'r') as f:
            samples_dict = json.load(f)
        samples = samples_dict[f"iteration_{i}"][0] # there are MC_size runs, we take the first one for visualization
        all_bary_samples.append(np.array(samples))

    
    mu1_centers = [(-a,  M), ( a, -M)]
    mu2_centers = [(-a, -M), ( a,  M)]
    karcher_A_centers = [(-a, 0), (a, 0)] # this is the true barycenter
    karcher_B_centers = [(0, M), (0, -M)] # this is the other Karcher mean; used for initialization in the entropic iterative scheme


    # fetch circles info across iterations
    disks = {
        'mu1':      (mu1_centers,      'blue', True),
        'mu2':      (mu2_centers,      'orange', True),
        'karcherA': (karcher_A_centers, 'green', False),
        'karcherB': (karcher_B_centers, 'red', False),
    }

    xlim = (-1.2 * a, 1.2 * a)
    ylim = (-2 * M, 2 * M)

    visualize_iterations(all_bary_samples, eps=eps, plot_disks= disks, xlim=xlim, ylim=ylim, plot_dir = outputs_dir, plot_name = "Approximated_barycenter_samples_across_iterations")