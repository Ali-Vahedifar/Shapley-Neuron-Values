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
from typing import Dict, List, Optional

import numpy as np
import torch

from baselines import (ALL_METHODS, CUSTOM_BACKBONE, build_custom_model, build_method,
                       default_buffer_size, is_cil_only, requires_task_identity)
from cost import CostTracker
from utils import set_seed
from datasets import ContinualLearningBenchmark
from metrics import ContinualLearningMetrics
from models import count_neurons, count_parameters, create_model

DATASET_DEFAULTS = {
    'pmnist':       {'epochs': 20,  'batch_size': 10,  'tasks': [10]},
    'cifar10':      {'epochs': 50,  'batch_size': 64,  'tasks': [5]},
    'cifar20':      {'epochs': 200, 'batch_size': 64,  'tasks': [5, 10]},
    'cifar100':     {'epochs': 200, 'batch_size': 64,  'tasks': [10, 20]},
    'tinyimagenet': {'epochs': 200, 'batch_size': 64,  'tasks': [10, 20]},
    'imagenet1k':   {'epochs': 100, 'batch_size': 128, 'tasks': [10, 20, 50]},
}




def parse_args():
    p = argparse.ArgumentParser(description='SNV continual-learning experiments')
    p.add_argument('--method', default='snv', choices=ALL_METHODS)
    p.add_argument('--estimator_mode', choices=['manuscript', 'reverse_tmc'], default='manuscript')
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
    p.add_argument('--shapley_eval_batches', type=int, default=0,
                   help='validation batches used for each V(S); 0 uses the full split')
    p.add_argument('--payoff', default='accuracy', choices=['accuracy', 'loss'],
                   help='score V returns; loss is continuous and far lower-variance')
    p.add_argument('--selection', default='ablation',
                   choices=['ablation', 'inrun', 'dual'],
                   help="SNV only: which valuation decides S_t")
    p.add_argument('--inrun_val_every', type=int, default=1,
                   help='steps between refreshes of grad L_val for phi_run')
    p.add_argument('--freeze_old_heads', action=argparse.BooleanOptionalAction, default=False,
                   help="apply the documented head convention -- h_1..h_{t-1} stored, not "
                        "retrained -- to every method, not just SNV.  With old heads trainable "
                        "the CIL cross-entropy sees only new-class labels and drives every old "
                        "logit down.  MEASURED not to be the cause of the CIL zeros, though: "
                        "SGD CIL still gives exact 0.00 with it on, because the zeros come from "
                        "backbone drift.  Off by default so baselines keep published semantics; "
                        "")
    p.add_argument('--masked_inference', action=argparse.BooleanOptionalAction, default=True,
                   help='evaluate through the task subnetwork (default: enabled)')
    p.add_argument('--masked_training', action=argparse.BooleanOptionalAction, default=False,
                   help='after S_t is chosen, continue task t through its own subnetwork before '
                        'freezing it. This is an explicit non-paper ablation: Algorithm 1 '
                        'contains no post-selection optimization, so it is disabled by default.')
    p.add_argument('--consolidation_epochs', type=int, default=0,
                   help='epochs for that phase; 0 reuses --epochs (early stopping still applies)')
    p.add_argument('--consolidation_within_budget', action='store_true',
                   help='reserve consolidation_epochs from --epochs; separately labeled SNV variant')
    p.add_argument('--layer_floor', type=float, default=0.0,
                   help='minimum fraction of each layer that S_t must keep.  A global top-k can '
                        'empty a layer outright, which clamps it to a constant and severs the '
                        'subnetwork; 0 is the paper-faithful global top-k default.')
    p.add_argument('--dual_pool_factor', type=float, default=2.0,
                   help="selection='dual': candidate pool size as a multiple of k")

    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--buffer_size', type=int, default=None)

    p.add_argument('--no_truncation', action='store_true',
                   help='SNV ablation: disable TMAB truncation (plain MC marginals)')
    p.add_argument('--no_mab', action='store_true',
                   help='SNV ablation: disable the bandit stopping rule')
    # These were previously unreachable: build_method read them from kwargs that
    # train.py never supplied, so every baseline ran at its hard-coded default.
    p.add_argument('--ewc_lambda', type=float, default=5000.0)
    p.add_argument('--si_c', type=float, default=0.1)
    p.add_argument('--lwf_lambda', type=float, default=1.0)
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--ewc_gamma', type=float, default=1.0)
    p.add_argument('--si_xi', type=float, default=1e-3)
    p.add_argument('--nispa_s', type=float, default=None)
    p.add_argument('--nispa_prune_perc', type=float, default=90.0)
    p.add_argument('--nispa_recovery_perc', type=float, default=2.5)
    p.add_argument('--nispa_phase_epochs', type=int, default=5)
    p.add_argument('--nispa_max_phases', type=int, default=30)
    p.add_argument('--spacenet_s_init', type=float, default=None)
    p.add_argument('--spacenet_rewire_fraction', type=float, default=0.2,
                   help='SpaceNet fraction of active connections dropped and grown per epoch')
    p.add_argument('--wsn_density', type=float, default=0.5,
                   help='WSN per-layer alive fraction (official default: 0.5)')
    p.add_argument('--nispa_lambda_reg', type=float, default=1.0)
    p.add_argument('--spacenet_lambda_sp', type=float, default=1.0)
    p.add_argument('--pec_lambda', type=float, default=1.0)
    p.add_argument('--first_task_search', action='store_true',
                   help='deprecated alias for --search_tasks 1.  Scoring task 0 alone cannot '
                        'separate continual-learning hyperparameters: every one of them (SNV '
                        'sparsity/truncation/confidence, EWC lambda, ...) '
                        'first takes effect at task 1, so the whole grid collapses to a search '
                        'over lr and max() breaks the tie by iteration order.')
    p.add_argument('--search_tasks', type=int, default=0,
                   help='GTEP: keep the requested benchmark split but train and score only the '
                        'first K tasks.  K >= 2 is required for the score to see forgetting; '
                        'the reported SEARCH_ACC is the mean accuracy over those K tasks after '
                        'the last of them, so it prices retention as well as plasticity.')
    p.add_argument('--backbone', default=None, choices=['resnet18', 'resnet50'],
                   help='override the per-dataset default (cifar100: resnet18; '
                        'tinyimagenet and imagenet1k: resnet50)')
    p.add_argument('--gtep_half', type=int, default=None, choices=[1, 2],
                   help="GTEP high-similarity split: restrict the benchmark to one of two "
                        "DISJOINT halves of the dataset's classes.  1 = D^HT (tune here), "
                        "2 = D^E (evaluate here).  Half membership is fixed by --split_seed so "
                        "the two phases never share a class; --seed only varies the task "
                        "ordering, which is what GTEP's S trials average over.")
    p.add_argument('--split_seed', type=int, default=1234,
                   help='fixes which classes land in half 1 vs half 2; keep constant')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_runs', type=int, default=10)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--output_dir', default='./results')
    p.add_argument('--resume', action='store_true',
                   help='reuse the newest run directory for this config and continue '
                        'from its per-task checkpoints')
    p.add_argument('--run_dir', default=None,
                   help='write to (and resume from) this exact directory')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    if args.first_task_search and not args.search_tasks:
        args.search_tasks = 1
    if args.search_tasks:
        args.search_tasks = max(1, min(args.search_tasks, args.num_tasks))
        if args.search_tasks == 1:
            print('warning: --search_tasks 1 scores task 0 only, which no continual-learning '
                  'hyperparameter can influence; use --search_tasks 3 or more.')

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
        model = build_custom_model(args.method, args.dataset,
                                   benchmark.classes_per_task, args.scenario, args.num_tasks)
    else:
        model = create_model(args.dataset, benchmark.classes_per_task,
                             args.num_tasks, args.scenario,
                             backbone_name=args.backbone)
    method = build_method(
        args.method, model, device, args.scenario, args.lr,
        sparsity=args.sparsity, truncation=args.truncation, confidence=args.confidence,
        max_permutations=args.max_permutations,
        shapley_eval_batches=args.shapley_eval_batches,
        selection=args.selection, payoff=args.payoff,
        inrun_val_every=args.inrun_val_every,
        dual_pool_factor=args.dual_pool_factor,
        masked_inference=args.masked_inference,
        masked_training=args.masked_training,
        consolidation_epochs=args.consolidation_epochs,
        consolidation_within_budget=args.consolidation_within_budget,
        layer_floor=args.layer_floor,
        estimator_mode=args.estimator_mode,
        # 'inrun' needs no ablation pass at all -- that is the point of it.
        compute_ablation=args.selection != 'inrun',
        buffer_size=args.buffer_size, num_classes=benchmark.num_classes,
        freeze_old_heads=args.freeze_old_heads,
        use_truncation=not args.no_truncation, use_mab=not args.no_mab,
        ewc_lambda=args.ewc_lambda, si_c=args.si_c, lwf_lambda=args.lwf_lambda,
        temperature=args.temperature, ewc_gamma=args.ewc_gamma, si_xi=args.si_xi,
        nispa_s=args.nispa_s, spacenet_s_init=args.spacenet_s_init,
        spacenet_rewire_fraction=args.spacenet_rewire_fraction,
        nispa_prune_perc=args.nispa_prune_perc,
        nispa_recovery_perc=args.nispa_recovery_perc,
        nispa_phase_epochs=args.nispa_phase_epochs,
        nispa_max_phases=args.nispa_max_phases,
        wsn_density=args.wsn_density,
        nispa_lambda_reg=args.nispa_lambda_reg,
        spacenet_lambda_sp=args.spacenet_lambda_sp, pec_lambda=args.pec_lambda,
        weight_decay=args.weight_decay)
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


from method_loader import load as _load_method
_joint = _load_method("baselines/Joint", "joint")
run_joint = _joint.run_joint




def _ckpt_path(out_dir: str, run_id: int) -> str:
    return os.path.join(out_dir, f'ckpt_run{run_id}.pt')


def save_run_checkpoint(out_dir, run_id, last_task, model, method, tracker,
                        cost=None, task_costs=None) -> None:
    """Snapshot everything needed to resume this run after the given task.

    Written atomically (tmp + replace) so a kill mid-write cannot leave a
    truncated checkpoint behind.
    """
    state = {
        'run_id': run_id,
        'last_task': last_task,
        'model': model.state_dict(),
        'accuracy_matrix': tracker.accuracy_matrix,
        'random_baseline': tracker.random_baseline,
        'history': [h for h in getattr(method, 'history', [])],
        'rng': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    # Algorithm state that is not registered in the main model.  These values
    # are intentionally explicit: serialising method.__dict__ would also capture
    # the live model/device/criterion and module lookup tables, creating aliasing
    # bugs on restore.  torch.save preserves tensors, small helper objects and
    # frozen teacher modules losslessly.
    extra_names = (
        'fisher', 'anchor', 'omega', '_w', '_prev', '_task_start', '_grad',
        'teacher', 'student', 'old_model', 'frozen_backbone', 'old_width',
        'p_mask', 'mask_back', 'aggregation_means',
        'stable', '_snapshot', 'reserved', 'current',
        'stable_sets', 'freeze_masks',
        'accumulated', 'task_masks', 'task_bn',
        'exemplars', 'class_means', 'buffer',
        'memory_x', 'memory_y', 'memory_t',
        'valuation_seconds', 'phase_seconds',
    )
    state['method_extra'] = {
        name: getattr(method, name) for name in extra_names
        if hasattr(method, name)
    }
    # PEC owns a bank of students outside the shared backbone/model wrapper.
    if hasattr(method, 'students'):
        state['students'] = method.students.state_dict()
    if cost is not None:
        state['cost'] = {
            'train_seconds': cost.train_seconds,
            'task_seconds': list(cost.task_seconds),
            'util_samples': list(cost._util_samples),
            'peak_memory_mb': cost.summary()['GPU_MEM_PEAK_MB'],
        }
    state['task_costs'] = list(task_costs or [])
    mm = getattr(method, 'mask_manager', None)
    if mm is not None:
        state['cumulative_mask'] = mm.cumulative_mask.cpu()
        state['task_masks'] = {k: v.cpu() for k, v in mm.task_masks.items()}
    for attr in ('shapley_values', 'inrun_values'):
        vals = getattr(method, attr, None)
        if vals:
            state[attr] = {k: v.cpu() for k, v in vals.items()}
    tes = getattr(method, 'task_eval_state', None)
    if tes:
        state['task_eval_state'] = {
            k: (m.cpu(), {kk: vv.cpu() for kk, vv in d.items()}) for k, (m, d) in tes.items()}
    tns = getattr(method, 'task_norm_state', None)
    if tns:
        state['task_norm_state'] = {
            task: {name: {key: value.cpu() for key, value in module_state.items()}
                   for name, module_state in task_state.items()}
            for task, task_state in tns.items()}

    path = _ckpt_path(out_dir, run_id)
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)


def load_run_checkpoint(out_dir, run_id, model, method, tracker, device,
                        cost=None, task_costs=None):
    """Restore a run; returns the last completed task, or -1 if no checkpoint."""
    path = _ckpt_path(out_dir, run_id)
    if not os.path.exists(path):
        return -1
    state = torch.load(path, map_location=device, weights_only=False)
    last = state['last_task']

    # Heads must exist before the state dict can be loaded into them.  Task
    # last+1's head is created by the zero-shot measurement, so it may be there.
    if hasattr(model, 'ensure_head'):
        for t in range(min(last + 2, tracker.num_tasks)):
            model.ensure_head(t)
        model.to(device)
    model.load_state_dict(state['model'])

    mm = getattr(method, 'mask_manager', None)
    if mm is not None and 'cumulative_mask' in state:
        mm.cumulative_mask = state['cumulative_mask'].to(device)
        mm.task_masks = {k: v.to(device) for k, v in state['task_masks'].items()}
        mm._mask_cache = None
    for attr in ('shapley_values', 'inrun_values'):
        if attr in state and hasattr(method, attr):
            setattr(method, attr, dict(state[attr]))
    if 'task_eval_state' in state and hasattr(method, 'task_eval_state'):
        method.task_eval_state = {
            k: (m.to(device), {kk: vv.to(device) for kk, vv in d.items()})
            for k, (m, d) in state['task_eval_state'].items()}
    if 'task_norm_state' in state and hasattr(method, 'task_norm_state'):
        method.task_norm_state = {
            task: {name: {key: value.to(device) for key, value in module_state.items()}
                   for name, module_state in task_state.items()}
            for task, task_state in state['task_norm_state'].items()}
    method.history = list(state.get('history', []))

    for name, value in state.get('method_extra', {}).items():
        if isinstance(value, torch.nn.Module):
            value = value.to(device)
        elif hasattr(value, 'device') and value.__class__.__name__ == 'Reservoir':
            value.device = device
        setattr(method, name, value)
    if 'students' in state and hasattr(method, 'students'):
        method.students.load_state_dict(state['students'])

    tracker.accuracy_matrix = state['accuracy_matrix']
    tracker.random_baseline = state['random_baseline']
    if cost is not None and 'cost' in state:
        cost.train_seconds = float(state['cost'].get('train_seconds', 0.0))
        cost.task_seconds = list(state['cost'].get('task_seconds', []))
        cost._util_samples = list(state['cost'].get('util_samples', []))
        cost.prior_peak_mb = float(state['cost'].get('peak_memory_mb', 0.0))
    if task_costs is not None:
        task_costs.extend(state.get('task_costs', []))

    rng = state.get('rng', {})
    if rng:
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu() if hasattr(rng['torch'], 'cpu') else rng['torch'])
        if rng.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])
    return last


def run_single_experiment(args, run_id: int, device: torch.device,
                          out_dir: Optional[str] = None) -> Dict:
    seed = args.seed + run_id
    set_seed(seed)

    benchmark = ContinualLearningBenchmark(
        args.dataset, args.num_tasks, args.data_root, seed, args.scenario, args.num_workers,
        gtep_half=args.gtep_half, split_seed=args.split_seed)
    model, method = build(args, benchmark, device)

    if args.verbose:
        print(f'\nRun {run_id + 1}/{args.num_runs}  |  {count_parameters(model):,} params  |  '
              f'N = {count_neurons(model)} neurons  |  split '
              f'{benchmark.split_report(0)}')

    tasks_to_load = args.search_tasks or args.num_tasks
    loaders = [benchmark.get_task_data(t, args.batch_size) for t in range(tasks_to_load)]
    test_loaders = [l[2] for l in loaders]

    tracker = ContinualLearningMetrics(args.num_tasks)
    cost = CostTracker(device)
    task_costs = []

    start_task = 0
    if out_dir is not None and args.resume:
        start_task = load_run_checkpoint(out_dir, run_id, model, method, tracker, device,
                                         cost=cost, task_costs=task_costs) + 1
    if start_task == 0 and not args.search_tasks:
        tracker.set_random_baseline(random_init_baseline(args, benchmark, device, test_loaders))
    elif args.verbose:
        print(f'  resuming run {run_id} from task {start_task}')

    stop_task = args.search_tasks or args.num_tasks
    for task_id in range(start_task, stop_task):
        train_loader, val_loader, _ = loaders[task_id]
        valuation_before = float(getattr(method, 'valuation_seconds', 0.0))
        phase_before = dict(getattr(method, 'phase_seconds', {}))
        cost.start_train()
        method.train_task(task_id, train_loader, val_loader, num_epochs=args.epochs,
                          patience=args.patience, verbose=args.verbose)
        cost.stop_train()
        valuation_after = float(getattr(method, 'valuation_seconds', 0.0))
        valuation_delta = max(0.0, valuation_after - valuation_before)
        task_wall = cost.task_seconds[-1]
        phase_after = dict(getattr(method, 'phase_seconds', {}))
        task_costs.append({
            'task_id': task_id,
            'wall_seconds': task_wall,
            'training_seconds': max(0.0, task_wall - valuation_delta),
            'valuation_seconds': valuation_delta,
            'mc_seconds': max(0.0, phase_after.get('mc', 0.0)
                              - phase_before.get('mc', 0.0)),
            'trunc_mab_seconds': max(
                0.0, phase_after.get('trunc_mab', 0.0)
                - phase_before.get('trunc_mab', 0.0)),
            'mask_seconds': max(0.0, phase_after.get('mask_update', 0.0)
                                - phase_before.get('mask_update', 0.0)),
        })

        row = method.evaluate_all_tasks(test_loaders, task_id)
        tracker.update(task_id, row)

        if args.search_tasks and task_id == stop_task - 1:
            # Score the K-task prefix, not task 0.  SEARCH_ACC is the mean of the
            # final row -- retention of tasks 0..K-2 plus accuracy on K-1 -- so a
            # setting that learns the newest task by destroying the earlier ones
            # is penalised rather than rewarded.
            final = tracker.get_accuracy_matrix()[task_id, :task_id + 1]
            diag = np.array([tracker.get_accuracy_matrix()[k, k] for k in range(task_id + 1)])
            metrics = {'SEARCH_ACC': float(np.nanmean(final)),
                       'SEARCH_TASKS': float(task_id + 1),
                       'FIRST_TASK_VAL_ACC': method.evaluate(loaders[0][1], 0)}
            if task_id > 0:
                metrics['SEARCH_BWT'] = float(np.nanmean(final[:-1] - diag[:-1]))
                metrics['SEARCH_PRIOR_NONZERO'] = float(np.count_nonzero(final[:-1] > 0.0))
            cost.add_valuation(getattr(method, 'valuation_seconds', 0.0))
            metrics.update(cost.summary())
            if args.method == 'snv' and getattr(method, 'history', None):
                last = method.history[-1]
                metrics['SNV_CONVERGED'] = float(bool(last.get('converged', False)))
                metrics['SNV_PERMUTATIONS'] = float(last.get('permutations', 0))
                metrics['SNV_EVALUATIONS'] = float(last.get('evaluations', 0))
            return {'metrics': metrics, 'accuracy_matrix': tracker.get_accuracy_matrix(),
                    'seed': seed, 'run_id': run_id,
                    'task_costs': task_costs,
                    'shapley_values': {
                        str(k): v.detach().cpu().tolist()
                        for k, v in getattr(method, 'shapley_values', {}).items()},
                    'history': [{k: v for k, v in h.items()
                                 if isinstance(v, (int, float, str, bool))}
                                for h in getattr(method, 'history', [])]}

        # evaluate_all_tasks now returns the whole row -- every task, trained or
        # not -- so the old single superdiagonal cell A[t, t+1] is already
        # covered and writing it again here would just overwrite one column of
        # that row with a separately-measured value.

        # Grid candidates are already resume-safe at candidate granularity and
        # never need to be resumed mid-candidate.  Saving three full ResNet
        # checkpoints for every Cartesian candidate would consume hundreds of
        # gigabytes.
        if out_dir is not None and not args.search_tasks:
            # Append cost state to the atomic task checkpoint so a resumed run's
            # cost table covers the complete sequence, not just its suffix.
            save_run_checkpoint(out_dir, run_id, task_id, model, method, tracker,
                                cost=cost, task_costs=task_costs)

        if args.verbose:
            print(f'  after T{task_id + 1}: ACC so far = {row.mean() * 100:.2f}%  '
                  f'per-task = {[f"{a * 100:.1f}" for a in row]}')

    metrics = tracker.get_all_metrics()
    if hasattr(method, 'mask_manager'):
        metrics['CAP'] = method.mask_manager.get_capacity_used()
        # CAP counts neurons; CAP_W counts the weights those neurons own, which
        # is the unit WSN's capacity is expressed in.  Only CAP_W is comparable
        # across the two methods.
        if hasattr(method.mask_manager, 'weight_capacity_used'):
            metrics['CAP_W'] = method.mask_manager.weight_capacity_used()
    elif method.history and 'capacity_used' in method.history[-1]:
        metrics['CAP'] = method.history[-1]['capacity_used']
    else:
        metrics['CAP'] = float('nan')

    cost.add_valuation(getattr(method, 'valuation_seconds', 0.0))
    _ph = getattr(method, 'phase_seconds', None)
    if _ph:
        metrics['VAL_MC_MIN'] = _ph.get('mc', 0.0) / 60.0
        metrics['VAL_TRUNCMAB_MIN'] = _ph.get('trunc_mab', 0.0) / 60.0
        metrics['VAL_MASK_MIN'] = _ph.get('mask_update', 0.0) / 60.0
    for _k, _v in cost.summary().items():
        metrics[_k] = _v
    if task_costs:
        metrics['TASK_TRAIN_PER_TASK_MIN'] = (
            np.mean([row['training_seconds'] for row in task_costs]) / 60.0)
        valued = [row for row in task_costs if row['valuation_seconds'] > 0]
        metrics['VALUATION_TRANSITIONS'] = float(len(valued))
        if valued:
            metrics['VALUATION_PER_TRANSITION_MIN'] = (
                np.mean([row['valuation_seconds'] for row in valued]) / 60.0)
            metrics['MC_PER_TRANSITION_MIN'] = (
                np.mean([row['mc_seconds'] for row in valued]) / 60.0)
            metrics['TRUNCMAB_PER_TRANSITION_MIN'] = (
                np.mean([row['trunc_mab_seconds'] for row in valued]) / 60.0)
            metrics['MASK_PER_TRANSITION_MIN'] = (
                np.mean([row['mask_seconds'] for row in valued]) / 60.0)
    _xb = next(iter(test_loaders[0]))[0]
    metrics['GFLOPS_FWD'] = cost.gflops_forward(
        model, _xb, forward_fn=lambda x: method.predict(x, 0))
    model.eval()
    metrics['INFER_MS_PER_SAMPLE'] = cost.measure_inference(
        lambda x: method.predict(x, 0), test_loaders[0])

    if args.verbose:
        tracker.print_summary()
        print(f"  train {metrics['TRAIN_TIME_MIN']:.1f} min  |  "
              f"util {metrics['GPU_UTIL_MEAN']:.0f}%  |  "
              f"peak {metrics['GPU_MEM_PEAK_MB']:.0f} MiB  |  "
              f"{metrics['GFLOPS_FWD']:.3f} GFLOPs/sample")

    return {'metrics': metrics, 'accuracy_matrix': tracker.get_accuracy_matrix(),
            'seed': seed, 'run_id': run_id,
            'task_costs': task_costs,
            'shapley_values': {
                str(k): v.detach().cpu().tolist()
                for k, v in getattr(method, 'shapley_values', {}).items()},
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
    if args.scenario == 'task_il' and is_cil_only(args.method):
        raise SystemExit(f'{args.method} is defined for class-incremental learning only '
                         'and has no task-incremental result; run it with '
                         '--scenario class_il.')

    # Second resolution alone collides: two runs of the same config launched in
    # the same second share out_dir and overwrite each other's run_*.json,
    # config.json and accuracy_matrices.npy, silently leaving one result.
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    sel = f'_{args.selection}' if args.method == 'snv' else ''
    half = f'_half{args.gtep_half}' if args.gtep_half else ''
    exp = (f'{args.method}{sel}_{args.dataset}{half}_{args.num_tasks}tasks'
           f'_c{args.sparsity}_{args.scenario}')
    out_dir = args.run_dir or os.path.join(args.output_dir, exp, stamp)
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, 'config.json')
    if args.resume and os.path.exists(config_path):
        with open(config_path) as handle:
            previous_config = json.load(handle)
        ignored = {'gpu', 'num_workers', 'output_dir', 'run_dir', 'resume',
                   'verbose'}
        changed = {
            key: (previous_config.get(key), value)
            for key, value in vars(args).items()
            if key not in ignored and previous_config.get(key) != value
        }
        if changed:
            raise SystemExit(
                'refusing to resume a checkpoint with a different experiment '
                f'configuration: {changed}')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print('=' * 62)
    print(f'{args.method.upper()}  |  {args.dataset}  |  {args.num_tasks} tasks  |  '
          f'{args.scenario}  |  c = {args.sparsity}  |  {args.num_runs} runs')
    print('=' * 62)

    results, start = [], time.time()
    for run_id in range(args.num_runs):
        result = run_joint(args, device, run_id) if args.method == 'joint' \
            else run_single_experiment(args, run_id, device, out_dir=out_dir)
        results.append(result)
        with open(os.path.join(out_dir, f'run_{run_id}.json'), 'w') as f:
            json.dump({'metrics': {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                                       else float(v))
                                   for k, v in result['metrics'].items()},
                       'accuracy_matrix': np.nan_to_num(result['accuracy_matrix'], nan=-1).tolist(),
                       'seed': result['seed'], 'run_id': result['run_id'],
                       'task_costs': result.get('task_costs', []),
                       'shapley_values': result.get('shapley_values', {})},
                      f, indent=2)

    aggregated = {}
    for key in results[0]['metrics']:
        vals = np.array([r['metrics'][key] for r in results], dtype=float)
        aggregated[key] = {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals)),
                           'values': [float(v) for v in vals]}

    # Stability requested by the TMAB ablation table. Normalize each task/run
    # vector independently to [0,1], then average the across-run standard
    # deviation over neurons and tasks. One run cannot estimate stability.
    if args.method == 'snv' and len(results) >= 2:
        common_tasks = set(results[0].get('shapley_values', {}))
        for result in results[1:]:
            common_tasks &= set(result.get('shapley_values', {}))
        task_stds = []
        for task_id in sorted(common_tasks, key=int):
            normalized = []
            for result in results:
                values = np.asarray(result['shapley_values'][task_id], dtype=float)
                span = values.max() - values.min()
                normalized.append((values - values.min()) / span
                                  if span > 0 else np.zeros_like(values))
            task_stds.append(np.std(np.stack(normalized), axis=0).mean())
        if task_stds:
            value = float(np.mean(task_stds))
            aggregated['SHAPLEY_STD'] = {
                'mean': value, 'std': 0.0, 'values': task_stds}

    print('\n' + '=' * 62)
    print(f'RESULTS over {args.num_runs} runs')
    print('=' * 62)
    for key, unit in (('ACC', '%'), ('BWT', '%'), ('PS', ''), ('P', ''), ('S', ''),
                      ('FWT', '%'), ('AF', '%'), ('CAP', '%'), ('CAP_W', '%'),
                      ('TRAIN_TIME_MIN', ''), ('VALUATION_MIN', ''),
                      ('TASK_TRAIN_PER_TASK_MIN', ''),
                      ('VALUATION_PER_TRANSITION_MIN', ''),
                      ('MC_PER_TRANSITION_MIN', ''),
                      ('TRUNCMAB_PER_TRANSITION_MIN', ''),
                      ('MASK_PER_TRANSITION_MIN', ''),
                      ('VAL_MC_MIN', ''), ('VAL_TRUNCMAB_MIN', ''), ('VAL_MASK_MIN', ''),
                      ('GPU_UTIL_MEAN', ''), ('GPU_MEM_PEAK_MB', ''),
                      ('GFLOPS_FWD', ''), ('INFER_MS_PER_SAMPLE', '')):
        if key not in aggregated:
            continue
        m, s = aggregated[key]['mean'], aggregated[key]['std']
        scale = 100.0 if unit == '%' and key not in ('CAP', 'CAP_W') else 1.0
        print(f'  {key:<4} {m * scale:8.2f} +/- {s * scale:5.2f} {unit}')
    if 'SHAPLEY_STD' in aggregated:
        print(f"  SHAPLEY_STD {aggregated['SHAPLEY_STD']['mean']:.6f}")
    for _key in ('SEARCH_ACC', 'SEARCH_BWT', 'FIRST_TASK_VAL_ACC'):
        if _key in aggregated:
            m, s = aggregated[_key]['mean'], aggregated[_key]['std']
            print(f'  {_key} {m * 100:8.2f} +/- {s * 100:5.2f} %')
    if 'SEARCH_PRIOR_NONZERO' in aggregated:
        print(f"  SEARCH_PRIOR_NONZERO {aggregated['SEARCH_PRIOR_NONZERO']['mean']:.0f}")
    for key in ('SNV_CONVERGED', 'SNV_PERMUTATIONS', 'SNV_EVALUATIONS'):
        if key in aggregated:
            print(f'  {key} {aggregated[key]["mean"]:.6g}')
    print(f'\ntotal time: {(time.time() - start) / 60:.1f} min')

    with open(os.path.join(out_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)
    np.save(os.path.join(out_dir, 'accuracy_matrices.npy'),
            np.stack([r['accuracy_matrix'] for r in results]))
    print(f'saved to {out_dir}')
    return aggregated


if __name__ == '__main__':
    main()
