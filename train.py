"""
Experiment driver for SNV and every baseline.

    python train.py --method snv --dataset imagenet1k --num_tasks 50 \
                    --scenario class_il --sparsity 0.02 --num_runs 10

Per-dataset defaults follow the experimental setup: 200 epochs and batch 64 for
CIFAR-100 / TinyImageNet, 100 epochs and batch 128 for ImageNet-1k, 20 epochs
and batch 10 for PMNIST; early stopping on validation loss throughout.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

from baselines import (ALL_METHODS, CUSTOM_BACKBONE, build_dytox_model, build_method,
                       default_buffer_size, requires_task_identity)
from datasets import ContinualLearningBenchmark
from metrics import ContinualLearningMetrics
from models import count_neurons, count_parameters, create_model

DATASET_DEFAULTS = {
    'pmnist':       {'epochs': 20,  'batch_size': 10,  'tasks': [10]},
    'cifar100':     {'epochs': 200, 'batch_size': 64,  'tasks': [10, 20]},
    'tinyimagenet': {'epochs': 200, 'batch_size': 64,  'tasks': [10, 20]},
    'imagenet1k':   {'epochs': 100, 'batch_size': 128, 'tasks': [10, 20, 50]},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description='SNV continual-learning experiments')
    p.add_argument('--method', default='snv', choices=ALL_METHODS)
    p.add_argument('--dataset', default='cifar100', choices=list(DATASET_DEFAULTS))
    p.add_argument('--data_root', default='./data')
    p.add_argument('--num_tasks', type=int, default=10,
                   help='10, 20 or 50 (50 is evaluated on ImageNet-1k)')
    p.add_argument('--scenario', default='class_il', choices=['class_il', 'task_il'])

    p.add_argument('--sparsity', type=float, default=0.1, help='capacity budget c')
    p.add_argument('--truncation', type=float, default=0.1, help='truncation threshold tau')
    p.add_argument('--confidence', type=float, default=0.95, help='MAB confidence alpha')
    p.add_argument('--max_permutations', type=int, default=200,
                   help='safety cap on EstimateSNV permutations')
    p.add_argument('--shapley_eval_batches', type=int, default=8,
                   help='validation batches used for each V(S)')

    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--buffer_size', type=int, default=None)

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_runs', type=int, default=10)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--output_dir', default='./results')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    d = DATASET_DEFAULTS[args.dataset]
    if args.epochs is None:
        args.epochs = d['epochs']
    if args.batch_size is None:
        args.batch_size = d['batch_size']
    if args.buffer_size is None:
        args.buffer_size = default_buffer_size(args.dataset, args.scenario)
    if args.num_tasks not in d['tasks']:
        print(f"warning: {args.num_tasks} tasks is outside the evaluated set "
              f"{d['tasks']} for {args.dataset}")
    return args


def build(args, benchmark, device):
    if args.method in CUSTOM_BACKBONE:
        model = build_dytox_model(args.dataset, benchmark.classes_per_task)
    else:
        model = create_model(args.dataset, benchmark.classes_per_task,
                             args.num_tasks, args.scenario)
    method = build_method(
        args.method, model, device, args.scenario, args.lr,
        sparsity=args.sparsity, truncation=args.truncation, confidence=args.confidence,
        max_permutations=args.max_permutations,
        shapley_eval_batches=args.shapley_eval_batches,
        buffer_size=args.buffer_size, num_classes=benchmark.num_classes)
    return model, method


@torch.no_grad()
def random_init_baseline(args, benchmark, device, test_loaders) -> List[float]:
    """b_t (RAC): accuracy of the randomly initialised model on each task."""
    model, _ = build(args, benchmark, device)
    model = model.to(device).eval()
    for t in range(args.num_tasks):
        model.ensure_head(t)
    model = model.to(device)
    baseline = []
    for t, loader in enumerate(test_loaders):
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, t) if args.scenario == 'task_il' else model(x)
            correct += out.argmax(1).eq(y).sum().item()
            total += y.numel()
        baseline.append(correct / total if total else 0.0)
    del model
    return baseline


def run_joint(args, device, run_id) -> Dict:
    """Joint training on the union of all tasks -- the upper bound."""
    import torch.nn as nn
    seed = args.seed + run_id
    set_seed(seed)
    benchmark = ContinualLearningBenchmark(args.dataset, args.num_tasks, args.data_root,
                                           seed, 'class_il', args.num_workers)
    model = create_model(args.dataset, benchmark.classes_per_task, args.num_tasks, 'class_il')
    model = model.to(device)
    for t in range(args.num_tasks):
        model.ensure_head(t)
    model = model.to(device)

    train_loader, _ = benchmark.get_joint_data(args.batch_size)
    test_loaders = [benchmark.get_task_data(t, args.batch_size)[2]
                    for t in range(args.num_tasks)]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            optimizer.step()
        if args.verbose:
            print(f'  joint epoch {epoch + 1}/{args.epochs}')

    model.eval()
    per_task = []
    with torch.no_grad():
        for t, loader in enumerate(test_loaders):
            correct = total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x, t) if args.scenario == 'task_il' else model(x)
                correct += out.argmax(1).eq(y).sum().item()
                total += y.numel()
            per_task.append(correct / total if total else 0.0)

    return {'metrics': {'ACC': float(np.mean(per_task)), 'BWT': 0.0, 'PS': float('nan'),
                        'P': float('nan'), 'S': 1.0, 'FWT': float('nan'), 'AF': 0.0,
                        'I': 0.0},
            'accuracy_matrix': np.tile(per_task, (args.num_tasks, 1)),
            'joint_accuracy': per_task, 'seed': seed, 'run_id': run_id}


def run_single_experiment(args, run_id: int, device: torch.device) -> Dict:
    seed = args.seed + run_id
    set_seed(seed)

    benchmark = ContinualLearningBenchmark(
        args.dataset, args.num_tasks, args.data_root, seed, args.scenario, args.num_workers)
    model, method = build(args, benchmark, device)

    if args.verbose:
        print(f'\nRun {run_id + 1}/{args.num_runs}  |  {count_parameters(model):,} params  |  '
              f'N = {count_neurons(model)} neurons  |  split '
              f'{benchmark.split_report(0)}')

    loaders = [benchmark.get_task_data(t, args.batch_size) for t in range(args.num_tasks)]
    test_loaders = [l[2] for l in loaders]

    tracker = ContinualLearningMetrics(args.num_tasks)
    tracker.set_random_baseline(random_init_baseline(args, benchmark, device, test_loaders))

    for task_id in range(args.num_tasks):
        train_loader, val_loader, _ = loaders[task_id]
        method.train_task(task_id, train_loader, val_loader, num_epochs=args.epochs,
                          patience=args.patience, verbose=args.verbose)

        row = method.evaluate_all_tasks(test_loaders, task_id)
        tracker.update(task_id, row)

        # A[t, t+1]: task t+1 before it is trained.  Its head is created first so
        # the measurement is defined; that head is untouched until task t+1 runs.
        if task_id + 1 < args.num_tasks:
            if hasattr(model, 'ensure_head'):
                model.ensure_head(task_id + 1)
                model.to(device)
            tracker.record_zero_shot(task_id, task_id + 1,
                                     method.evaluate(test_loaders[task_id + 1], task_id + 1))

        if args.verbose:
            print(f'  after T{task_id + 1}: ACC so far = {row.mean() * 100:.2f}%  '
                  f'per-task = {[f"{a * 100:.1f}" for a in row]}')

    metrics = tracker.get_all_metrics()
    if hasattr(method, 'mask_manager'):
        metrics['CAP'] = method.mask_manager.get_capacity_used()
    elif method.history and 'capacity_used' in method.history[-1]:
        metrics['CAP'] = method.history[-1]['capacity_used']
    else:
        metrics['CAP'] = float('nan')

    if args.verbose:
        tracker.print_summary()

    return {'metrics': metrics, 'accuracy_matrix': tracker.get_accuracy_matrix(),
            'seed': seed, 'run_id': run_id,
            'history': [{k: v for k, v in h.items()
                         if isinstance(v, (int, float, str, bool))}
                        for h in getattr(method, 'history', [])]}


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    if args.scenario == 'class_il' and requires_task_identity(args.method):
        raise SystemExit(f'{args.method} requires the task identity at test time and has '
                         'no class-incremental result; run it with --scenario task_il.')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp = f'{args.method}_{args.dataset}_{args.num_tasks}tasks_c{args.sparsity}_{args.scenario}'
    out_dir = os.path.join(args.output_dir, exp, stamp)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print('=' * 62)
    print(f'{args.method.upper()}  |  {args.dataset}  |  {args.num_tasks} tasks  |  '
          f'{args.scenario}  |  c = {args.sparsity}  |  {args.num_runs} runs')
    print('=' * 62)

    results, start = [], time.time()
    for run_id in range(args.num_runs):
        result = run_joint(args, device, run_id) if args.method == 'joint' \
            else run_single_experiment(args, run_id, device)
        results.append(result)
        with open(os.path.join(out_dir, f'run_{run_id}.json'), 'w') as f:
            json.dump({'metrics': {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                                       else float(v))
                                   for k, v in result['metrics'].items()},
                       'accuracy_matrix': np.nan_to_num(result['accuracy_matrix'], nan=-1).tolist(),
                       'seed': result['seed'], 'run_id': result['run_id']}, f, indent=2)

    aggregated = {}
    for key in results[0]['metrics']:
        vals = np.array([r['metrics'][key] for r in results], dtype=float)
        aggregated[key] = {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals)),
                           'values': [float(v) for v in vals]}

    print('\n' + '=' * 62)
    print(f'RESULTS over {args.num_runs} runs')
    print('=' * 62)
    for key, unit in (('ACC', '%'), ('BWT', '%'), ('PS', ''), ('P', ''), ('S', ''),
                      ('FWT', '%'), ('AF', '%'), ('CAP', '%')):
        if key not in aggregated:
            continue
        m, s = aggregated[key]['mean'], aggregated[key]['std']
        scale = 100.0 if unit == '%' and key != 'CAP' else 1.0
        print(f'  {key:<4} {m * scale:8.2f} +/- {s * scale:5.2f} {unit}')
    print(f'\ntotal time: {(time.time() - start) / 60:.1f} min')

    with open(os.path.join(out_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)
    np.save(os.path.join(out_dir, 'accuracy_matrices.npy'),
            np.stack([r['accuracy_matrix'] for r in results]))
    print(f'saved to {out_dir}')
    return aggregated


if __name__ == '__main__':
    main()
