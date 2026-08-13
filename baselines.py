"""
Method registry.

Every entry exposes the same three calls -- ``train_task``, ``evaluate`` and
``evaluate_all_tasks`` -- so ``train.py`` drives SNV and each baseline through
one code path.

    buffer-free   snv, sgd, ewc, si, lwf, pec, wsn, spacenet, nispa, dcnet,
                  nfl, nfl+
    memory-based  icarl, derpp, dytox
    bounds        sgd (lower), joint (upper, see train.py --method joint)
"""

from typing import Dict

import torch

from snv_core import SNVContinualLearner
from baselines_regularization import EWC, SI, LwF, SGDBaseline
from baselines_memory import DERpp, DyTox, DyToxNet, ICaRL
from baselines_sparse import DCNet, NFL, NFLPlus, NISPA, PEC, WSN, SpaceNet

BUFFER_FREE = ['snv', 'sgd', 'ewc', 'si', 'lwf', 'pec', 'wsn', 'spacenet',
               'nispa', 'dcnet', 'nfl', 'nfl+']
MEMORY_BASED = ['icarl', 'derpp', 'dytox']
ALL_METHODS = BUFFER_FREE + MEMORY_BASED + ['joint']

_BASELINES = {
    'sgd': SGDBaseline, 'ewc': EWC, 'si': SI, 'lwf': LwF,
    'wsn': WSN, 'spacenet': SpaceNet, 'nispa': NISPA, 'dcnet': DCNet,
    'nfl': NFL, 'nfl+': NFLPlus, 'pec': PEC,
    'icarl': ICaRL, 'derpp': DERpp, 'dytox': DyTox,
}

# Methods that need the task identity at test time and so have no CIL result.
TIL_ONLY = {'wsn'}

# Methods that replace the standard backbone with their own architecture.
CUSTOM_BACKBONE = {'dytox'}


def requires_task_identity(method: str) -> bool:
    return method in TIL_ONLY


def build_dytox_model(dataset: str, classes_per_task: int) -> DyToxNet:
    image_size = {'cifar100': 32, 'tinyimagenet': 64, 'imagenet1k': 224, 'pmnist': 28}[dataset]
    patch = 4 if image_size <= 64 else 16
    return DyToxNet(image_size=image_size, patch_size=patch, dim=384, depth=5,
                    heads=12, classes_per_task=classes_per_task)


def build_method(name: str, model, device: torch.device, scenario: str,
                 lr: float, **kwargs):
    """Instantiate a method by name."""
    name = name.lower()
    if name == 'snv':
        return SNVContinualLearner(
            model=model, device=device, scenario=scenario, lr=lr,
            sparsity_ratio=kwargs.get('sparsity', 0.1),
            truncation_threshold=kwargs.get('truncation', 0.1),
            confidence_level=kwargs.get('confidence', 0.95),
            max_permutations=kwargs.get('max_permutations', 200),
            shapley_eval_batches=kwargs.get('shapley_eval_batches', 8))

    if name not in _BASELINES:
        raise ValueError(f'unknown method {name!r}; expected one of {sorted(ALL_METHODS)}')

    cls = _BASELINES[name]
    kw: Dict = {}
    if name in ('wsn', 'spacenet', 'nispa', 'nfl', 'nfl+'):
        sparsity = kwargs.get('sparsity', 0.1)
        key = {'wsn': 'sparsity', 'spacenet': 'density', 'nispa': 'stable_fraction',
               'nfl': 'freeze_fraction', 'nfl+': 'freeze_fraction'}[name]
        kw[key] = sparsity
    if name in ('icarl', 'derpp', 'dytox'):
        kw['buffer_size'] = kwargs.get('buffer_size', 2000)
    if name == 'pec':
        kw['num_classes'] = kwargs['num_classes']
    if name == 'ewc':
        kw['ewc_lambda'] = kwargs.get('ewc_lambda', 5000.0)
    if name == 'si':
        kw['si_c'] = kwargs.get('si_c', 0.1)
    if name in ('lwf', 'icarl', 'dytox'):
        kw['temperature'] = kwargs.get('temperature', 2.0)

    return cls(model, device, scenario=scenario, lr=lr, **kw)


def default_buffer_size(dataset: str, scenario: str) -> int:
    """2,000 exemplars for CIFAR-100 / TinyImageNet and 20,000 for ImageNet-1k
    under CIL; 200 under TIL."""
    if scenario == 'task_il':
        return 200
    return 20000 if dataset == 'imagenet1k' else 2000
