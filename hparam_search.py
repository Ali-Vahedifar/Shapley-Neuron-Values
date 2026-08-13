"""
Hyperparameter selection under the two-phase protocol (GTEP).

Phase 1 -- grid search using *only* the first task's validation split.
Phase 2 -- the winning configuration is frozen and reused for every later task;
           it is never re-tuned on data the model has not been allowed to see.

The search therefore trains a single task per grid point and scores on that
task's validation accuracy.  Nothing downstream of task 1 influences the choice.

    python hparam_search.py --method ewc --dataset cifar100 --scenario class_il
"""

import argparse
import itertools
import json
import os
from typing import Dict, List

import torch

from baselines import ALL_METHODS, CUSTOM_BACKBONE, build_dytox_model, build_method
from datasets import ContinualLearningBenchmark
from models import create_model
from train import DATASET_DEFAULTS, set_seed

# Grids per method.  'lr' applies to every method.
GRIDS: Dict[str, Dict[str, List]] = {
    '_common':  {'lr': [1e-4, 5e-4, 1e-3, 5e-3]},
    'snv':      {'truncation': [0.05, 0.1, 0.2], 'confidence': [0.90, 0.95, 0.99]},
    'ewc':      {'ewc_lambda': [10.0, 100.0, 1000.0, 5000.0, 10000.0]},
    'si':       {'si_c': [0.01, 0.1, 0.5, 1.0]},
    'lwf':      {'temperature': [1.0, 2.0, 4.0]},
    'icarl':    {'temperature': [1.0, 2.0, 4.0]},
    'derpp':    {'alpha': [0.1, 0.5, 1.0], 'beta': [0.1, 0.5, 1.0]},
    'dytox':    {'temperature': [1.0, 2.0], 'kd_lambda': [0.5, 1.0]},
    'dcnet':    {'lambda_disc': [0.1, 0.5, 1.0], 'lambda_cons': [0.5, 1.0, 2.0]},
    'spacenet': {'rewire_fraction': [0.1, 0.2, 0.3]},
    'wsn':      {},
    'nispa':    {},
    'nfl':      {},
    'nfl+':     {},
    'pec':      {},
    'sgd':      {},
}


def grid_for(method: str) -> List[Dict]:
    space = dict(GRIDS['_common'])
    space.update(GRIDS.get(method, {}))
    keys = sorted(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]


def parse_args():
    p = argparse.ArgumentParser(description='GTEP-style first-task grid search')
    p.add_argument('--method', default='snv', choices=ALL_METHODS)
    p.add_argument('--dataset', default='cifar100', choices=list(DATASET_DEFAULTS))
    p.add_argument('--data_root', default='./data')
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--scenario', default='class_il', choices=['class_il', 'task_il'])
    p.add_argument('--sparsity', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--output_dir', default='./results/hparams')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    d = DATASET_DEFAULTS[args.dataset]
    args.epochs = args.epochs or d['epochs']
    args.batch_size = args.batch_size or d['batch_size']
    return args


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    grid = grid_for(args.method)
    print(f'{len(grid)} configurations for {args.method}; scoring on task 1 validation only')

    records = []
    for i, cfg in enumerate(grid):
        set_seed(args.seed)
        benchmark = ContinualLearningBenchmark(args.dataset, args.num_tasks, args.data_root,
                                               args.seed, args.scenario, args.num_workers)
        if args.method in CUSTOM_BACKBONE:
            model = build_dytox_model(args.dataset, benchmark.classes_per_task)
        else:
            model = create_model(args.dataset, benchmark.classes_per_task,
                                 args.num_tasks, args.scenario)
        method = build_method(args.method, model, device, args.scenario,
                              cfg.get('lr', 1e-3), sparsity=args.sparsity,
                              num_classes=benchmark.num_classes,
                              **{k: v for k, v in cfg.items() if k != 'lr'})

        train_loader, val_loader, _ = benchmark.get_task_data(0, args.batch_size)
        method.train_task(0, train_loader, val_loader, num_epochs=args.epochs,
                          patience=args.patience, verbose=args.verbose)
        score = method.evaluate(val_loader, 0)
        records.append({'config': cfg, 'task1_val_acc': score})
        print(f'  [{i + 1}/{len(grid)}] {cfg} -> {score * 100:.2f}%')
        del method, model
        torch.cuda.empty_cache()

    best = max(records, key=lambda r: r['task1_val_acc'])
    print(f'\nselected (frozen for tasks 2..T): {best["config"]}  '
          f'({best["task1_val_acc"] * 100:.2f}% on task 1 val)')

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir,
                        f'{args.method}_{args.dataset}_{args.scenario}.json')
    with open(path, 'w') as f:
        json.dump({'method': args.method, 'dataset': args.dataset,
                   'scenario': args.scenario, 'selected': best['config'],
                   'all': records}, f, indent=2)
    print(f'saved to {path}')


if __name__ == '__main__':
    main()
