"""Joint training -- the upper bound.

Moved out of train.py.  Trains one model on the union of all tasks; under TIL
each sample is routed through the head of the task its class belongs to.
"""

from typing import Dict

import numpy as np
import torch

from cost import CostTracker
from datasets import ContinualLearningBenchmark
from models import create_model
from utils import set_seed


def run_joint(args, device, run_id) -> Dict:
    """Joint training on the union of all tasks -- the upper bound."""
    import torch.nn as nn
    seed = args.seed + run_id
    set_seed(seed)
    # The joint set is always built with global labels: under TIL `get_joint_data`
    # remaps every task into the same [0, classes_per_task) range, which would make
    # class 3 of task 0 and class 3 of task 2 the same label and destroy the upper
    # bound.  Global labels are split back into (task, local label) below.
    benchmark = ContinualLearningBenchmark(args.dataset, args.num_tasks, args.data_root,
                                           seed, 'class_il', args.num_workers)
    model = create_model(args.dataset, benchmark.classes_per_task, args.num_tasks,
                         args.scenario)
    model = model.to(device)
    effective_tasks = args.search_tasks or args.num_tasks
    for t in range(effective_tasks):
        model.ensure_head(t)
    model = model.to(device)

    train_loader, _ = benchmark.get_joint_data(args.batch_size,
                                                tasks=effective_tasks)
    test_loaders = [benchmark.get_task_data(t, args.batch_size)[2]
                    for t in range(effective_tasks)]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    cpt = benchmark.classes_per_task
    task_il = args.scenario == 'task_il'

    cost = CostTracker(device)
    cost.start_train()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if task_il:
                # Multi-head joint training: every sample goes through the head of
                # the task its class belongs to, scored against the local label.
                # The backbone is run ONCE over the whole batch and each head is
                # applied to its own slice of the features.  Calling
                # model(x[sel], t) inside the loop instead recomputes the full
                # backbone once per task present -- ~10x the work per batch for an
                # identical loss, which is why joint TIL was ~6x slower than joint
                # CIL rather than roughly equal.
                t_of = torch.div(y, cpt, rounding_mode='floor')
                features = model.get_features(x)
                loss = torch.zeros((), device=device)
                for t in t_of.unique():
                    sel = t_of == t
                    logits = model.heads[str(int(t))](features[sel])
                    loss = loss + criterion(logits, y[sel] % cpt) * \
                        (sel.sum() / y.numel())
            else:
                loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        if args.verbose:
            print(f'  joint epoch {epoch + 1}/{args.epochs}')

    cost.stop_train()

    model.eval()
    per_task = []
    with torch.no_grad():
        for t, loader in enumerate(test_loaders):
            correct = total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                if task_il:
                    out = model(x, t)
                    y = y % cpt          # test labels are global; head t is local
                else:
                    out = model(x)
                correct += out.argmax(1).eq(y).sum().item()
                total += y.numel()
            per_task.append(correct / total if total else 0.0)

    _m = {'ACC': float(np.mean(per_task)), 'BWT': 0.0, 'PS': float('nan'),
          'P': float('nan'), 'S': 1.0, 'FWT': float('nan'), 'AF': 0.0,
          'I': 0.0, 'CAP': float('nan')}
    if args.search_tasks:
        _m.update(SEARCH_ACC=float(np.mean(per_task)), SEARCH_BWT=0.0,
                  FIRST_TASK_VAL_ACC=float(per_task[0]))
    _m.update(cost.summary())
    _m['GFLOPS_FWD'] = cost.gflops_forward(
        model, next(iter(test_loaders[0]))[0],
        forward_fn=(lambda x: model(x, 0)) if task_il else None)
    model.eval()
    _m['INFER_MS_PER_SAMPLE'] = cost.measure_inference(
        lambda x: model(x, 0) if task_il else model(x), test_loaders[0])
    if args.verbose:
        print(f"  train {_m['TRAIN_TIME_MIN']:.1f} min  |  util {_m['GPU_UTIL_MEAN']:.0f}%  |  "
              f"peak {_m['GPU_MEM_PEAK_MB']:.0f} MiB  |  {_m['GFLOPS_FWD']:.3f} GFLOPs/sample")

    matrix = np.full((args.num_tasks, args.num_tasks), np.nan)
    matrix[:effective_tasks, :effective_tasks] = np.tile(
        per_task, (effective_tasks, 1))
    return {'metrics': _m,
            'accuracy_matrix': matrix,
            'joint_accuracy': per_task, 'seed': seed, 'run_id': run_id}
