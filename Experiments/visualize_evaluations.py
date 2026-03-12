import matplotlib.pyplot as plt
import math
import os, json
import numpy as np

def plot_v_values(v_values_path: str, true_v_OT_path: str=None, true_v_path: str=None, plot_dir=None, plot_name=None):
    # --- Load true V value from OT estimate ---
    if true_v_OT_path is not None:
        with open(true_v_OT_path, 'r') as f:
            true_OT_data = json.load(f)
        true_OT_mean = true_OT_data['mean']
        true_OT_std  = true_OT_data['std']
        true_OT_values    = true_OT_data['values']
        true_OT_n    = len(true_OT_values)
        true_OT_se   = true_OT_std / np.sqrt(true_OT_n)

    # --- Load true V value ---
    if true_v_path is not None:
        with open(true_v_path, 'r') as f:
            true_data = json.load(f)
        true_mean = true_data['mean']
        true_std  = true_data['std']
        true_n    = true_data['sample_size']
        true_se   = true_std / np.sqrt(true_n)

    # --- Load V values ---
    with open(v_values_path, 'r') as f:
        data = json.load(f)

    sorted_keys = sorted(data.keys(), key=lambda k: int(k.split('_')[1]))
    all_values  = [data[k]['values'] for k in sorted_keys]
    x_labels    = [f"Iter {int(k.split('_')[1])}" for k in sorted_keys]
    x_pos       = np.arange(1, len(sorted_keys) + 1)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.boxplot(all_values, positions=x_pos, widths=0.5,
               patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='steelblue'),
               medianprops=dict(color='steelblue', linewidth=2),
               whiskerprops=dict(color='steelblue'),
               capprops=dict(color='steelblue'),
               flierprops=dict(marker='x', color='steelblue', alpha=0.5))

    for i, vals in enumerate(all_values):
        ax.scatter(x_pos[i], max(vals), marker='^', color='red',    zorder=5, s=50, label='Max' if i == 0 else "")
        ax.scatter(x_pos[i], min(vals), marker='v', color='blue',  zorder=5, s=50, label='Min' if i == 0 else "")

    if true_v_OT_path is not None:
        ax.axhline(true_OT_mean, color='orange', linewidth=1.5, linestyle='--', label='True V-value (OT estimate)')
        # ax.axhspan(true_OT_mean - true_OT_se, true_OT_mean + true_OT_se, alpha=0.2, color='orange', label='True V ± SE')
    if true_v_path is not None:
        ax.axhline(true_mean, color='green', linewidth=1.5, linestyle='--', label='True V-value (Monte-Carlo estimate)')
        # ax.axhspan(true_mean - true_se, true_mean + true_se, alpha=0.2, color='green', label='True V ± SE')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('V-value')
    ax.set_title('Empirical approximations of the V-value across iterations')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if plot_dir is not None and plot_name is not None:
        plt.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
    else:
        plt.show()

def plot_w2_values(w2_values_path: str, plot_dir=None, plot_name=None):
    # --- Load W2 values ---
    with open(w2_values_path, 'r') as f:
        data = json.load(f)

    sorted_keys = sorted(data.keys(), key=lambda k: int(k.split('_')[1]))
    all_values  = [data[k]['values'] for k in sorted_keys]
    x_labels    = [f"Iter {int(k.split('_')[1])}" for k in sorted_keys]
    x_pos       = np.arange(1, len(sorted_keys) + 1)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.boxplot(all_values, positions=x_pos, widths=0.5,
               patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='steelblue'),
               medianprops=dict(color='steelblue', linewidth=2),
               whiskerprops=dict(color='steelblue'),
               capprops=dict(color='steelblue'),
               flierprops=dict(marker='x', color='steelblue', alpha=0.5))
    
    for i, vals in enumerate(all_values):
        ax.scatter(x_pos[i], max(vals), marker='^', color='red',   zorder=5, s=50, label='Max' if i == 0 else "")
        ax.scatter(x_pos[i], min(vals), marker='v', color='blue', zorder=5, s=50, label='Min' if i == 0 else "")

    ax.axhline(0, color='orange', linewidth=1.5, linestyle='--', label='Optimal (0)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('W2-to-Bary')
    ax.set_title('Empirical approximations of the 2-Wasserstein distance to the true barycenter across iterations')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if plot_dir is not None and plot_name is not None:
        plt.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
    else:
        plt.show()


def export_latex_table(results: dict, output_path: str, metric_names: list,
                       caption: str = "Results", label: str = "tab:results",
                       scientific: bool = False):
    """
    Parameters:
    -----------
    results      : dict with algorithm names as keys, each value is a list of dicts
                   (one per metric), each dict having 'mean' and 'values' keys.
    metric_names : list of metric names
    output_path  : path to output .txt file
    caption      : LaTeX table caption
    label        : LaTeX table label
    scientific   : if True, format numbers in scientific notation as a \times 10^{x}
    """

    def fmt(x):
        if scientific:
            s = f"{x:.6e}"                        # e.g. '1.422222e-03'
            mantissa, exp = s.split('e')
            exp = int(exp)                         # remove leading zeros and sign
            return f"${mantissa} \\times 10^{{{exp}}}$"
        else:
            return f"{x:.4f}"

    n_metrics = len(metric_names)
    col_spec = "l" + "cc" * n_metrics

    lines = []
    lines.append("% Requires \\usepackage{booktabs, multirow, adjustbox} in your LaTeX preamble")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"    \centering")
    lines.append(r"    \begin{adjustbox}{max width=\textwidth}")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \hline")

    metric_headers = " & ".join(
        [f"\\multicolumn{{2}}{{c}}{{\\textbf{{{name}}}}}" for name in metric_names]
    )
    lines.append(f"        \\multirow{{2}}{{*}}{{\\textbf{{Algorithm}}}} & {metric_headers} \\\\")

    cmidrules = " ".join([f"\\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(n_metrics)])
    lines.append(f"        {cmidrules}")

    sub_headers = " & ".join([r"\textbf{Mean} & \textbf{(Max, Min)}"] * n_metrics)
    lines.append(f"        & {sub_headers} \\\\")
    lines.append(r"        \hline")

    for algo_name, experiments in results.items():
        safe_name = algo_name.replace("_", r"\_")
        row = safe_name
        for exp_data in experiments:
            mean    = exp_data['mean']
            max_val = max(exp_data['values'])
            min_val = min(exp_data['values'])
            row += f" & {fmt(mean)} & ({fmt(max_val)}, {fmt(min_val)})"
        row += r" \\"
        lines.append(f"        {row}")

    lines.append(r"        \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \end{adjustbox}")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append(r"\end{table}")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"LaTeX table saved to {output_path}")