"""
Utilities for SNV Continual Learning.

Includes:
- Visualization functions for accuracy matrices and Shapley values
- Analysis utilities for mask overlap and capacity
- Logging and experiment tracking

Anonymous submission for ICML 2026.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import torch


def plot_accuracy_matrix(
    accuracy_matrix: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Accuracy Matrix",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot the accuracy matrix as a heatmap.
    
    Args:
        accuracy_matrix: Accuracy matrix of shape (num_tasks, num_tasks)
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size
    """
    num_tasks = accuracy_matrix.shape[0]
    
    # Create mask for upper triangular (not yet trained)
    mask = np.triu(np.ones_like(accuracy_matrix, dtype=bool), k=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        accuracy_matrix * 100,  # Convert to percentage
        mask=mask,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={'label': 'Accuracy (%)'},
        square=True
    )
    
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('After Training on Task', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Set tick labels
    ax.set_xticklabels([f'T{i+1}' for i in range(num_tasks)])
    ax.set_yticklabels([f'T{i+1}' for i in range(num_tasks)])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_accuracy_progression(
    accuracy_matrix: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Accuracy Progression",
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot accuracy progression for each task over training.
    
    Args:
        accuracy_matrix: Accuracy matrix of shape (num_tasks, num_tasks)
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size
    """
    num_tasks = accuracy_matrix.shape[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))
    
    for task_id in range(num_tasks):
        # Get accuracies for this task over all subsequent training
        accs = accuracy_matrix[task_id:, task_id]
        x = list(range(task_id, num_tasks))
        
        ax.plot(x, accs * 100, 'o-', color=colors[task_id], 
               label=f'Task {task_id + 1}', linewidth=2, markersize=6)
    
    ax.set_xlabel('After Training on Task', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(num_tasks))
    ax.set_xticklabels([f'T{i+1}' for i in range(num_tasks)])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_average_accuracy_curve(
    accuracy_matrix: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Average Accuracy vs Tasks",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot average accuracy after each task.
    
    Args:
        accuracy_matrix: Accuracy matrix of shape (num_tasks, num_tasks)
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size
    """
    num_tasks = accuracy_matrix.shape[0]
    
    # Compute average accuracy after each task
    avg_accs = []
    for i in range(num_tasks):
        avg_acc = np.mean(accuracy_matrix[i, :i+1])
        avg_accs.append(avg_acc)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(range(1, num_tasks + 1), np.array(avg_accs) * 100, 
           'b-o', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Tasks Learned', fontsize=12)
    ax.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(1, num_tasks + 1))
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_shapley_heatmap(
    shapley_values: Dict[int, torch.Tensor],
    neuron_info: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Layer-wise Shapley Neuron Importance",
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Plot Shapley values as a heatmap across layers and tasks.
    
    Args:
        shapley_values: Dictionary mapping task_id to Shapley value tensors
        neuron_info: List of neuron information dictionaries
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size
    """
    num_tasks = len(shapley_values)
    num_layers = len(neuron_info)
    
    # Aggregate Shapley values by layer
    layer_shapley = np.zeros((num_tasks, num_layers))
    
    for task_id, values in shapley_values.items():
        values_np = values.cpu().numpy() if torch.is_tensor(values) else values
        
        for layer_idx, info in enumerate(neuron_info):
            start_idx = info['start_idx']
            end_idx = info['end_idx']
            layer_shapley[task_id, layer_idx] = np.mean(values_np[start_idx:end_idx])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        layer_shapley,
        annot=True,
        fmt='.3f',
        cmap='Greens',
        ax=ax,
        cbar_kws={'label': 'Average Shapley Value'},
        xticklabels=[info['name'] for info in neuron_info],
        yticklabels=[f'Task {i+1}' for i in range(num_tasks)]
    )
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Task', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_mask_overlap(
    task_masks: Dict[int, torch.Tensor],
    save_path: Optional[str] = None,
    title: str = "Mask Overlap Between Tasks",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot Jaccard similarity between task masks.
    
    Args:
        task_masks: Dictionary mapping task_id to binary mask tensors
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size
    """
    num_tasks = len(task_masks)
    overlap_matrix = np.zeros((num_tasks, num_tasks))
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            mask_i = task_masks[i].cpu().numpy() if torch.is_tensor(task_masks[i]) else task_masks[i]
            mask_j = task_masks[j].cpu().numpy() if torch.is_tensor(task_masks[j]) else task_masks[j]
            
            # Jaccard coefficient
            intersection = np.sum(mask_i & mask_j)
            union = np.sum(mask_i | mask_j)
            
            if union > 0:
                overlap_matrix[i, j] = intersection / union
            else:
                overlap_matrix[i, j] = 0
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Jaccard Similarity'},
        square=True
    )
    
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Task', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.set_xticklabels([f'T{i+1}' for i in range(num_tasks)])
    ax.set_yticklabels([f'T{i+1}' for i in range(num_tasks)])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_capacity_growth(
    capacity_history: List[float],
    save_path: Optional[str] = None,
    title: str = "Capacity Growth Over Tasks",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot capacity growth over tasks.
    
    Args:
        capacity_history: List of capacity percentages after each task
        save_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size
    """
    num_tasks = len(capacity_history)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(range(1, num_tasks + 1), capacity_history, 
           'g-o', linewidth=2, markersize=8)
    ax.fill_between(range(1, num_tasks + 1), capacity_history, alpha=0.3, color='green')
    
    ax.set_xlabel('Number of Tasks Learned', fontsize=12)
    ax.set_ylabel('Capacity Used (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(1, num_tasks + 1))
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Add horizontal line at 100%
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Full Capacity')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_results_table(
    results: Dict[str, Dict],
    methods: List[str],
    datasets: List[str],
    metrics: List[str] = ['ACC', 'BWT', 'PS']
) -> str:
    """
    Create a formatted LaTeX table of results.
    
    Args:
        results: Nested dictionary of results[method][dataset][metric]
        methods: List of method names
        datasets: List of dataset names
        metrics: List of metric names to include
        
    Returns:
        LaTeX table string
    """
    # Header
    table = "\\begin{tabular}{l" + "c" * (len(datasets) * len(metrics)) + "}\n"
    table += "\\toprule\n"
    
    # Dataset header
    table += " & " + " & ".join([f"\\multicolumn{{{len(metrics)}}}{{c}}{{{d}}}" 
                                 for d in datasets]) + " \\\\\n"
    
    # Metric header
    metric_headers = metrics * len(datasets)
    table += " & " + " & ".join(metric_headers) + " \\\\\n"
    table += "\\midrule\n"
    
    # Method rows
    for method in methods:
        row = method
        for dataset in datasets:
            for metric in metrics:
                if method in results and dataset in results[method]:
                    val = results[method][dataset].get(metric, {})
                    mean = val.get('mean', 0)
                    std = val.get('std', 0)
                    
                    if metric == 'ACC':
                        row += f" & {mean*100:.2f} (±{std*100:.2f})"
                    else:
                        row += f" & {mean:.2f} (±{std:.2f})"
                else:
                    row += " & -"
        row += " \\\\\n"
        table += row
    
    table += "\\bottomrule\n"
    table += "\\end{tabular}"
    
    return table


def save_experiment_config(
    config: Dict,
    output_dir: str
):
    """Save experiment configuration to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'config.json')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def load_experiment_results(
    output_dir: str
) -> Dict:
    """Load experiment results from directory."""
    results = {}
    
    # Load aggregated results
    agg_path = os.path.join(output_dir, 'aggregated_results.json')
    if os.path.exists(agg_path):
        with open(agg_path, 'r') as f:
            results['aggregated'] = json.load(f)
    
    # Load config
    config_path = os.path.join(output_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)
    
    # Load accuracy matrices
    matrices_path = os.path.join(output_dir, 'accuracy_matrices.npy')
    if os.path.exists(matrices_path):
        results['accuracy_matrices'] = np.load(matrices_path)
    
    return results


def compare_methods_barplot(
    results: Dict[str, Dict],
    metric: str = 'ACC',
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Create bar plot comparing methods on a single metric.
    
    Args:
        results: Dictionary mapping method names to result dictionaries
        metric: Metric to compare
        save_path: Optional path to save the figure
        title: Optional plot title
        figsize: Figure size
    """
    methods = list(results.keys())
    means = [results[m].get(metric, {}).get('mean', 0) for m in methods]
    stds = [results[m].get(metric, {}).get('std', 0) for m in methods]
    
    if metric == 'ACC':
        means = [m * 100 for m in means]
        stds = [s * 100 for s in stds]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel(f'{metric} {"(%)" if metric == "ACC" else ""}', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'{metric} Comparison', fontsize=14)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
