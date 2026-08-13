"""
Plots and tables for the paper's figures.

  plot_accuracy_matrix         the A[i, j] heatmap behind Fig. 7
  plot_average_accuracy_curves Fig. 3 -- ACC after each task, several methods
  plot_pruning_curves          Fig. 4 -- accuracy under increasing pruning
  plot_shapley_heatmap         layer-wise mean phi per task
  plot_mask_overlap            Jaccard similarity between the S_t
  plot_capacity_growth         |B_t| / N over tasks
  results_table                LaTeX table in the paper's ACC / BWT / PS layout

seaborn is used when available and matplotlib alone otherwise.
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:                                     # pragma: no cover
    _HAS_SNS = False


def _finish(fig, save_path: Optional[str]):
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _heatmap(ax, data, annot_fmt, cmap, vmin, vmax, cbar_label, mask=None):
    if _HAS_SNS:
        sns.heatmap(data, mask=mask, annot=True, fmt=annot_fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax, ax=ax, square=True,
                    cbar_kws={'label': cbar_label})
        return
    shown = np.ma.masked_where(mask, data) if mask is not None else data
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax, label=cbar_label)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask is None or not mask[i, j]:
                ax.text(j, i, format(data[i, j], annot_fmt), ha='center',
                        va='center', fontsize=7)


# --------------------------------------------------------------------------- #
def plot_accuracy_matrix(accuracy_matrix: np.ndarray, save_path: Optional[str] = None,
                         title: str = 'Accuracy matrix', figsize=(9, 7)):
    """A[i, j] as a heatmap; entries never evaluated are left blank."""
    data = np.asarray(accuracy_matrix, dtype=float) * 100
    mask = np.isnan(data)
    fig, ax = plt.subplots(figsize=figsize)
    _heatmap(ax, np.nan_to_num(data), '.1f', 'RdYlGn', 0, 100, 'Accuracy (%)', mask)
    n = data.shape[0]
    ax.set_xlabel('Evaluated on task')
    ax.set_ylabel('After training on task')
    ax.set_title(title)
    ax.set_xticks(np.arange(n) + 0.5 if _HAS_SNS else np.arange(n))
    ax.set_yticks(np.arange(n) + 0.5 if _HAS_SNS else np.arange(n))
    ax.set_xticklabels([f'T{i + 1}' for i in range(n)])
    ax.set_yticklabels([f'T{i + 1}' for i in range(n)], rotation=0)
    _finish(fig, save_path)


def plot_average_accuracy_curves(curves: Dict[str, Sequence[float]],
                                 memory_based: Sequence[str] = (),
                                 save_path: Optional[str] = None,
                                 title: str = 'ACC after each task', figsize=(8, 5)):
    """Fig. 3.  Solid lines are buffer-free, dashed lines memory-based.

    Each point is the average accuracy over every task seen so far, i.e. the
    row mean of A up to the current task.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(curves), 1)))
    for (name, values), color in zip(curves.items(), colors):
        values = np.asarray(values, dtype=float) * 100
        style = '--' if name in memory_based else '-'
        ax.plot(range(1, len(values) + 1), values, style, marker='o', color=color,
                label=name, linewidth=2, markersize=5)
    ax.set_xlabel('Number of tasks learned')
    ax.set_ylabel('ACC (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    _finish(fig, save_path)


def average_accuracy_curve(accuracy_matrix: np.ndarray) -> np.ndarray:
    """Row means of A up to the diagonal -- the series plotted in Fig. 3."""
    a = np.asarray(accuracy_matrix, dtype=float)
    return np.array([np.nanmean(a[i, :i + 1]) for i in range(a.shape[0])])


def plot_pruning_curves(results: Dict[str, Dict], save_path: Optional[str] = None,
                        title: str = 'Parameter usage efficiency',
                        cliff: float = 0.5, figsize=(8, 5)):
    """Fig. 4.

    ``results`` maps a method name to the dict written by ``pruning.py``
    (``rates``, ``accuracies``, ``critical_rate``).  The dashed vertical line
    marks each method's critical pruning percentage.  A sharper, earlier cliff
    means the parameters are carrying information; a flat curve is redundancy.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))
    for (name, res), color in zip(results.items(), colors):
        rates = np.asarray(res['rates']) * 100
        accs = np.asarray(res['accuracies']) * 100
        ax.plot(rates, accs, '-o', color=color, label=name, linewidth=2, markersize=4)
        crit = res.get('critical_rate')
        if crit:
            ax.axvline(crit * 100, color=color, linestyle='--', alpha=0.55)
    ax.set_xlabel('Weights pruned (%)')
    ax.set_ylabel('ACC (%)')
    ax.set_title(f'{title}  (cliff = {int(cliff * 100)}% of unpruned ACC)')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    _finish(fig, save_path)


def load_pruning_results(directory: str) -> Dict[str, Dict]:
    """Collect every JSON written by ``pruning.py`` in a directory."""
    out = {}
    for fn in sorted(os.listdir(directory)):
        if fn.endswith('.json'):
            with open(os.path.join(directory, fn)) as f:
                res = json.load(f)
            out[res.get('method', fn[:-5])] = res
    return out


def plot_shapley_heatmap(shapley_values: Dict[int, np.ndarray], groups,
                         save_path: Optional[str] = None,
                         title: str = 'Layer-wise Shapley Neuron Values', figsize=(13, 6)):
    """Mean phi per layer per task.

    ``groups`` is the list of ``NeuronGroup`` from ``build_neuron_index``.
    """
    tasks = sorted(shapley_values)
    data = np.zeros((len(tasks), len(groups)))
    for row, t in enumerate(tasks):
        phi = shapley_values[t]
        phi = phi.detach().cpu().numpy() if hasattr(phi, 'detach') else np.asarray(phi)
        for col, g in enumerate(groups):
            data[row, col] = phi[g.start:g.end].mean()

    fig, ax = plt.subplots(figsize=figsize)
    _heatmap(ax, data, '.3f', 'Greens', float(data.min()), float(data.max()),
             'mean phi')
    ax.set_xticks(np.arange(len(groups)) + (0.5 if _HAS_SNS else 0))
    ax.set_yticks(np.arange(len(tasks)) + (0.5 if _HAS_SNS else 0))
    ax.set_xticklabels([g.name for g in groups], rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels([f'T{t + 1}' for t in tasks], rotation=0)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Task')
    ax.set_title(title)
    _finish(fig, save_path)


def plot_mask_overlap(task_masks: Dict[int, np.ndarray], save_path: Optional[str] = None,
                      title: str = 'Overlap between task subnetworks', figsize=(7, 6)):
    """Jaccard similarity between the S_t -- the sharing Fig. 1 depicts."""
    tasks = sorted(task_masks)
    n = len(tasks)
    overlap = np.zeros((n, n))
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            a = np.asarray(task_masks[ti].cpu() if hasattr(task_masks[ti], 'cpu')
                           else task_masks[ti], dtype=bool)
            b = np.asarray(task_masks[tj].cpu() if hasattr(task_masks[tj], 'cpu')
                           else task_masks[tj], dtype=bool)
            union = np.sum(a | b)
            overlap[i, j] = np.sum(a & b) / union if union else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    _heatmap(ax, overlap, '.2f', 'Blues', 0, 1, 'Jaccard similarity')
    ax.set_xticks(np.arange(n) + (0.5 if _HAS_SNS else 0))
    ax.set_yticks(np.arange(n) + (0.5 if _HAS_SNS else 0))
    ax.set_xticklabels([f'T{t + 1}' for t in tasks])
    ax.set_yticklabels([f'T{t + 1}' for t in tasks], rotation=0)
    ax.set_title(title)
    _finish(fig, save_path)


def plot_capacity_growth(capacity_history: Sequence[float], save_path: Optional[str] = None,
                         title: str = 'Capacity used', figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    n = len(capacity_history)
    ax.plot(range(1, n + 1), capacity_history, '-o', color='seagreen', linewidth=2)
    ax.fill_between(range(1, n + 1), capacity_history, alpha=0.25, color='seagreen')
    ax.axhline(100, color='firebrick', linestyle='--', alpha=0.6, label='full capacity')
    ax.set_xlabel('Number of tasks learned')
    ax.set_ylabel('|B_t| / N  (%)')
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    _finish(fig, save_path)


# --------------------------------------------------------------------------- #
def results_table(results: Dict[str, Dict[str, Dict]], methods: Sequence[str],
                  columns: Sequence[str], metrics: Sequence[str] = ('ACC', 'BWT', 'PS')) -> str:
    """LaTeX table in the paper's ACC / BWT / PS layout.

    ``results[method][column][metric]`` holds ``{'mean': ..., 'std': ...}``;
    a missing cell prints ``---`` as the paper does.
    """
    ncols = len(columns) * len(metrics)
    lines = ['\\begin{tabular}{@{}l ' + 'c' * ncols + '@{}}', '\\toprule']
    lines.append(' & ' + ' & '.join(
        f'\\multicolumn{{{len(metrics)}}}{{c}}{{\\textbf{{{c}}}}}' for c in columns) + ' \\\\')
    lines.append('\\cmidrule(lr){2-' + str(ncols + 1) + '}')
    lines.append('\\textbf{Method} & ' + ' & '.join(
        f'{m}$\\uparrow$' for _ in columns for m in metrics) + ' \\\\')
    lines.append('\\midrule')

    for method in methods:
        cells = []
        for column in columns:
            entry = results.get(method, {}).get(column)
            for metric in metrics:
                if not entry or metric not in entry:
                    cells.append('---')
                    continue
                mean = entry[metric]['mean']
                std = entry[metric].get('std')
                scale = 100.0 if metric in ('ACC', 'BWT') else 1.0
                cell = f'{mean * scale:.2f}'
                if metric == 'ACC' and std is not None:
                    cell += f' \\scriptsize{{$\\pm$ {std * scale:.2f}}}'
                cells.append(cell)
        lines.append(f'{method} & ' + ' & '.join(cells) + ' \\\\')

    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines)


def load_experiment_results(output_dir: str) -> Dict:
    """Read back what ``train.py`` wrote for one configuration."""
    out = {}
    for key, fn in (('aggregated', 'aggregated_results.json'), ('config', 'config.json')):
        path = os.path.join(output_dir, fn)
        if os.path.exists(path):
            with open(path) as f:
                out[key] = json.load(f)
    path = os.path.join(output_dir, 'accuracy_matrices.npy')
    if os.path.exists(path):
        out['accuracy_matrices'] = np.load(path)
    return out


def save_experiment_config(config: Dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    return path
