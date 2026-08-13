"""
Weight-pruning analysis (paper Fig. 4, "Network parameter usage efficiency").

After a full continual-learning run, weights are pruned globally by magnitude at
increasing rates and the average accuracy is re-measured.  A *sharp* drop is the
good outcome: it means most weights carry task-relevant information.  Tolerance
to deep pruning exposes redundancy.

The critical pruning percentage reported here is the smallest rate at which
accuracy falls below ``--cliff`` of its unpruned value -- the dashed vertical
line in the figure.

    python pruning.py --method snv --dataset cifar100 --num_tasks 10 \\
                      --scenario class_il --sparsity 0.1
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from baselines import ALL_METHODS, CUSTOM_BACKBONE, build_dytox_model, build_method
from datasets import ContinualLearningBenchmark
from models import create_model
from train import DATASET_DEFAULTS, set_seed

PRUNE_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
               0.60, 0.70, 0.80, 0.90]


def prunable_weights(model: nn.Module) -> List[torch.Tensor]:
    """Conv and Linear weights of the backbone; heads and norms are left alone."""
    out = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and not name.startswith('heads'):
            w = getattr(module, 'weight', None)
            if w is None and hasattr(module, 'parametrizations'):
                w = module.parametrizations.weight.original
            if w is not None:
                out.append(w)
    return out


@torch.no_grad()
def apply_global_pruning(model: nn.Module, rate: float,
                         backup: List[torch.Tensor]) -> None:
    """Restore from ``backup`` then zero the globally smallest ``rate`` of weights."""
    weights = prunable_weights(model)
    for w, b in zip(weights, backup):
        w.copy_(b)
    if rate <= 0:
        return
    flat = torch.cat([w.abs().flatten() for w in weights])
    k = int(rate * flat.numel())
    if k <= 0:
        return
    threshold = torch.kthvalue(flat.cpu(), k).values.item()
    for w in weights:
        w.mul_((w.abs() > threshold).to(w.dtype))


@torch.no_grad()
def average_accuracy(method, test_loaders, scenario: str) -> float:
    accs = [method.evaluate(loader, t) for t, loader in enumerate(test_loaders)]
    return float(np.mean(accs))


def critical_rate(rates: List[float], accs: List[float], cliff: float) -> float:
    """Smallest rate whose accuracy is below ``cliff`` x the unpruned accuracy."""
    base = accs[0]
    for r, a in zip(rates, accs):
        if r > 0 and a < cliff * base:
            return r
    return float('nan')


def parse_args():
    p = argparse.ArgumentParser(description='Weight-pruning efficiency analysis')
    p.add_argument('--method', default='snv', choices=ALL_METHODS)
    p.add_argument('--dataset', default='cifar100', choices=list(DATASET_DEFAULTS))
    p.add_argument('--data_root', default='./data')
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--scenario', default='class_il', choices=['class_il', 'task_il'])
    p.add_argument('--sparsity', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--cliff', type=float, default=0.5,
                   help='fraction of unpruned accuracy that defines the cliff')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--output_dir', default='./results/pruning')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    d = DATASET_DEFAULTS[args.dataset]
    args.epochs = args.epochs or d['epochs']
    args.batch_size = args.batch_size or d['batch_size']
    return args


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    benchmark = ContinualLearningBenchmark(args.dataset, args.num_tasks, args.data_root,
                                           args.seed, args.scenario, args.num_workers)
    if args.method in CUSTOM_BACKBONE:
        model = build_dytox_model(args.dataset, benchmark.classes_per_task)
    else:
        model = create_model(args.dataset, benchmark.classes_per_task,
                             args.num_tasks, args.scenario)
    method = build_method(args.method, model, device, args.scenario, args.lr,
                          sparsity=args.sparsity, num_classes=benchmark.num_classes)

    loaders = [benchmark.get_task_data(t, args.batch_size) for t in range(args.num_tasks)]
    test_loaders = [l[2] for l in loaders]

    print(f'training {args.method} on {args.dataset} ({args.num_tasks} tasks) ...')
    for t in range(args.num_tasks):
        method.train_task(t, loaders[t][0], loaders[t][1], num_epochs=args.epochs,
                          patience=args.patience, verbose=args.verbose)

    backup = [w.detach().clone() for w in prunable_weights(method.model)]
    rates, accs = [], []
    print('\n  rate      ACC')
    for rate in PRUNE_RATES:
        apply_global_pruning(method.model, rate, backup)
        acc = average_accuracy(method, test_loaders, args.scenario)
        rates.append(rate)
        accs.append(acc)
        print(f'  {rate * 100:5.1f}%  {acc * 100:6.2f}%')
    apply_global_pruning(method.model, 0.0, backup)

    crit = critical_rate(rates, accs, args.cliff)
    print(f'\ncritical pruning percentage ({int(args.cliff * 100)}% of unpruned ACC): '
          f'{crit * 100:.0f}%' if not np.isnan(crit) else
          f'\nno cliff below {int(max(PRUNE_RATES) * 100)}% pruning -- '
          'the method is highly redundant')

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(
        args.output_dir,
        f'{args.method}_{args.dataset}_{args.num_tasks}tasks_{args.scenario}.json')
    with open(path, 'w') as f:
        json.dump({'method': args.method, 'dataset': args.dataset,
                   'num_tasks': args.num_tasks, 'scenario': args.scenario,
                   'rates': rates, 'accuracies': accs,
                   'critical_rate': None if np.isnan(crit) else crit}, f, indent=2)
    print(f'saved to {path}')


if __name__ == '__main__':
    main()
