"""
Main Training Script for Shapley Neuron Valuation (SNV) Continual Learning.

This script runs the complete SNV continual learning experiments as described
in the paper for PMNIST, CIFAR-100, and TinyImageNet datasets.

Usage:
    python train.py --dataset cifar100 --num_tasks 10 --sparsity 0.1 --scenario class_il

Anonymous submission for ICML 2026.
"""

import argparse
import os
import json
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from snv_core import SNVContinualLearner, NeuronMaskManager
from models import create_model, count_parameters, count_neurons
from datasets import ContinualLearningBenchmark
from metrics import ContinualLearningMetrics, compute_per_task_accuracies


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SNV Continual Learning Training'
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar100',
                       choices=['pmnist', 'cifar100', 'tinyimagenet'],
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--num_tasks', type=int, default=10,
                       choices=[10, 20],
                       help='Number of tasks')
    
    # Model arguments
    parser.add_argument('--scenario', type=str, default='class_il',
                       choices=['class_il', 'task_il'],
                       help='Continual learning scenario')
    
    # SNV arguments
    parser.add_argument('--sparsity', type=float, default=0.1,
                       help='Sparsity ratio c (fraction of neurons per task)')
    parser.add_argument('--truncation', type=float, default=0.1,
                       help='Truncation threshold tau')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level alpha for MAB')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum epochs per task')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of runs for averaging')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def run_single_experiment(
    args,
    run_id: int,
    device: torch.device
) -> dict:
    """
    Run a single SNV continual learning experiment.
    
    Args:
        args: Command line arguments
        run_id: Run identifier
        device: Torch device
        
    Returns:
        Dictionary of results
    """
    # Set seed for this run
    seed = args.seed + run_id
    set_seed(seed)
    
    # Create benchmark
    benchmark = ContinualLearningBenchmark(
        dataset_name=args.dataset,
        num_tasks=args.num_tasks,
        data_root=args.data_root,
        seed=seed,
        scenario=args.scenario
    )
    
    # Determine total classes
    if args.dataset == 'pmnist':
        num_classes = 10
        batch_size = 10  # Paper specifies batch_size=10 for PMNIST
        epochs = 20      # Paper specifies 20 epochs for PMNIST
    else:
        num_classes = benchmark.get_cumulative_classes(args.num_tasks - 1)
        batch_size = args.batch_size
        epochs = args.epochs
    
    # Create model
    model = create_model(
        dataset=args.dataset,
        num_classes=num_classes,
        scenario=args.scenario
    )
    
    if args.verbose:
        print(f"\nRun {run_id + 1}/{args.num_runs}")
        print(f"Model parameters: {count_parameters(model):,}")
        print(f"Total neurons: {count_neurons(model)}")
    
    # Create SNV learner
    learner = SNVContinualLearner(
        model=model,
        device=device,
        sparsity_ratio=args.sparsity,
        truncation_threshold=args.truncation,
        confidence_level=args.confidence,
        lr=args.lr
    )
    
    # Initialize metrics tracker
    metrics_tracker = ContinualLearningMetrics(args.num_tasks)
    
    # Store test loaders for evaluation
    test_loaders = []
    
    # Training loop over tasks
    for task_id in range(args.num_tasks):
        if args.verbose:
            print(f"\n{'='*50}")
            print(f"Training Task {task_id + 1}/{args.num_tasks}")
            print(f"{'='*50}")
        
        # Get task data
        train_loader, val_loader, test_loader = benchmark.get_task_data(
            task_id, batch_size=batch_size
        )
        test_loaders.append(test_loader)
        
        # Train on task
        task_result = learner.train_task(
            task_id=task_id,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            patience=args.patience,
            verbose=args.verbose
        )
        
        # Evaluate on all tasks seen so far
        task_accuracies = compute_per_task_accuracies(
            model=learner.model,
            test_loaders=test_loaders,
            device=device,
            current_task=task_id
        )
        
        # Update metrics
        metrics_tracker.update(task_id, task_accuracies)
        
        if args.verbose:
            avg_acc = np.mean(task_accuracies)
            print(f"\nTask {task_id + 1} complete:")
            print(f"  Average accuracy: {avg_acc*100:.2f}%")
            print(f"  Capacity used: {task_result['capacity_used']:.2f}%")
            print(f"  Per-task accuracies: {[f'{a*100:.1f}%' for a in task_accuracies]}")
    
    # Compute final metrics
    final_metrics = metrics_tracker.get_all_metrics()
    
    # Add capacity metric
    final_metrics['CAP'] = learner.mask_manager.get_capacity_used()
    
    if args.verbose:
        metrics_tracker.print_summary()
    
    return {
        'metrics': final_metrics,
        'accuracy_matrix': metrics_tracker.get_accuracy_matrix(),
        'seed': seed,
        'run_id': run_id
    }


def main():
    """Main function to run SNV experiments."""
    args = parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.dataset}_{args.num_tasks}tasks_c{args.sparsity}_{args.scenario}"
    output_dir = os.path.join(args.output_dir, exp_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SNV Continual Learning Experiment")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Tasks: {args.num_tasks}")
    print(f"Scenario: {args.scenario}")
    print(f"Sparsity ratio (c): {args.sparsity}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Run experiments
    all_results = []
    start_time = time.time()
    
    for run_id in range(args.num_runs):
        result = run_single_experiment(args, run_id, device)
        all_results.append(result)
        
        # Save intermediate results
        with open(os.path.join(output_dir, f'run_{run_id}.json'), 'w') as f:
            result_save = {
                'metrics': result['metrics'],
                'accuracy_matrix': result['accuracy_matrix'].tolist(),
                'seed': result['seed'],
                'run_id': result['run_id']
            }
            json.dump(result_save, f, indent=2)
    
    elapsed_time = time.time() - start_time
    
    # Aggregate results
    metrics_keys = all_results[0]['metrics'].keys()
    aggregated = {}
    
    for key in metrics_keys:
        values = [r['metrics'][key] for r in all_results]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS (averaged over {} runs)".format(args.num_runs))
    print("=" * 60)
    
    print(f"\nAverage Accuracy (ACC):    {aggregated['ACC']['mean']*100:.2f}% (±{aggregated['ACC']['std']*100:.2f})")
    print(f"Backward Transfer (BWT):   {aggregated['BWT']['mean']:.4f} (±{aggregated['BWT']['std']:.4f})")
    print(f"Forward Transfer (FWT):    {aggregated['FWT']['mean']:.4f} (±{aggregated['FWT']['std']:.4f})")
    print(f"Plasticity-Stability (PS): {aggregated['PS']['mean']:.4f} (±{aggregated['PS']['std']:.4f})")
    print(f"Average Forgetting (AF):   {aggregated['AF']['mean']:.4f} (±{aggregated['AF']['std']:.4f})")
    print(f"Capacity Used (CAP):       {aggregated['CAP']['mean']:.2f}% (±{aggregated['CAP']['std']:.2f})")
    
    print(f"\nTotal time: {elapsed_time/60:.2f} minutes")
    print("=" * 60)
    
    # Save aggregated results
    with open(os.path.join(output_dir, 'aggregated_results.json'), 'w') as f:
        # Convert numpy types for JSON serialization
        agg_save = {}
        for key, val in aggregated.items():
            agg_save[key] = {
                'mean': float(val['mean']),
                'std': float(val['std']),
                'values': [float(v) for v in val['values']]
            }
        json.dump(agg_save, f, indent=2)
    
    # Save accuracy matrices
    accuracy_matrices = np.stack([r['accuracy_matrix'] for r in all_results])
    np.save(os.path.join(output_dir, 'accuracy_matrices.npy'), accuracy_matrices)
    
    print(f"\nResults saved to: {output_dir}")
    
    return aggregated


if __name__ == '__main__':
    main()
