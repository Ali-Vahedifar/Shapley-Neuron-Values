"""
Method registry.

Every entry exposes the same three calls -- ``train_task``, ``evaluate`` and
``evaluate_all_tasks`` -- so ``train.py`` and ``audited_gtep.py`` drive SNV and
each baseline through one code path.

    bounds        sgd (lower), joint (upper, see train.py --method joint)
    regularise    ewc, si, lwf
    sparse/arch   wsn (Task-IL only), spacenet, nispa, pec (Class-IL only)
    unlearning    uniclun  (continual learning + machine unlearning)
    ours          snv (SNV-A; see SNV/snv_adaptive.py and snv_adaptive_run.py)
"""

from typing import Dict

import torch

from snv_core import SNVContinualLearner
from method_loader import load as _load_method
UniCLUN = _load_method('baselines/UniCLUN', 'uniclun').UniCLUN
from baselines.regularization import EWC, SI, LwF, SGDBaseline
from baselines.sparse import NISPA, PEC, WSN, SpaceNet, build_nispa_model
BUFFER_FREE = ['snv', 'sgd', 'ewc', 'si', 'lwf', 'pec', 'wsn', 'spacenet',
               'nispa']
MEMORY_BASED = ['uniclun']
ALL_METHODS = BUFFER_FREE + MEMORY_BASED + ['joint']

_BASELINES = {
    'sgd': SGDBaseline, 'ewc': EWC, 'si': SI, 'lwf': LwF,
    'wsn': WSN, 'spacenet': SpaceNet, 'nispa': NISPA, 'pec': PEC,
    'uniclun': UniCLUN,
}

# Methods that need the task identity at test time and so have no CIL result.
TIL_ONLY = {'wsn'}

# PEC is defined and reported only for class-incremental learning.
CIL_ONLY = {'pec'}

# Methods that replace the standard backbone with their own architecture.
CUSTOM_BACKBONE = {'nispa', 'pec'}


def requires_task_identity(method: str) -> bool:
    return method in TIL_ONLY


def is_cil_only(method: str) -> bool:
    return method in CIL_ONLY


def build_custom_model(method: str, dataset: str, classes_per_task: int,
                       scenario: str, num_tasks: int = 10):
    if method == 'pec':
        return _load_method('baselines/PEC', 'pec').PECNet(classes_per_task * num_tasks, classes_per_task)
    if method == 'nispa':
        return build_nispa_model(dataset, classes_per_task, scenario)
    raise ValueError(f'no custom model registered for {method!r}')


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
            shapley_eval_batches=kwargs.get('shapley_eval_batches', 8),
            selection=kwargs.get('selection', 'ablation'),
            payoff=kwargs.get('payoff', 'accuracy'),
            track_inrun=kwargs.get('track_inrun', False),
            compute_ablation=kwargs.get('compute_ablation', True),
            inrun_val_every=kwargs.get('inrun_val_every', 1),
            dual_pool_factor=kwargs.get('dual_pool_factor', 2.0),
            use_truncation=kwargs.get('use_truncation', True),
            use_mab=kwargs.get('use_mab', True),
            masked_inference=kwargs.get('masked_inference', True),
            masked_training=kwargs.get('masked_training', False),
            consolidation_epochs=kwargs.get('consolidation_epochs', 0),
            consolidation_within_budget=kwargs.get('consolidation_within_budget', False),
            layer_floor=kwargs.get('layer_floor', 0.0),
            estimator_mode=kwargs.get('estimator_mode', 'manuscript'))

    if name not in _BASELINES:
        raise ValueError(f'unknown method {name!r}; expected one of {sorted(ALL_METHODS)}')

    cls = _BASELINES[name]
    kw: Dict = {}
    if name in ('wsn', 'spacenet', 'nispa'):
        sparsity = kwargs.get('sparsity', 0.1)
        key = {'wsn': 'sparsity', 'spacenet': 'density',
               'nispa': 'stable_fraction'}[name]
        kw[key] = sparsity
    if name == 'wsn':
        kw['sparsity'] = kwargs.get('wsn_density', 0.5)
    if name == 'uniclun':
        kw['buffer_size'] = kwargs.get('buffer_size', 2000)
    if name == 'pec':
        kw['num_classes'] = kwargs['num_classes']
        kw['lambda_pec'] = kwargs.get('pec_lambda', 1.0)
    if name == 'ewc':
        kw['ewc_lambda'] = kwargs.get('ewc_lambda', 5000.0)
        kw['gamma'] = kwargs.get('ewc_gamma', 1.0)
    if name == 'si':
        kw['si_c'] = kwargs.get('si_c', 0.1)
        kw['xi'] = kwargs.get('si_xi', 1e-3)
    if name == 'nispa':
        kw['prune_perc'] = kwargs.get('nispa_prune_perc', 90.0)
        kw['recovery_perc'] = kwargs.get('nispa_recovery_perc', 2.5)
        kw['phase_epochs'] = kwargs.get('nispa_phase_epochs', 5)
        kw['max_phases'] = kwargs.get('nispa_max_phases', 30)
        kw['lambda_reg'] = kwargs.get('nispa_lambda_reg', 1.0)
    if name == 'spacenet':
        kw['density'] = kwargs.get('spacenet_s_init', kwargs.get('sparsity', 0.1))
        kw['rewire_fraction'] = kwargs.get('spacenet_rewire_fraction', 0.2)
    if name == 'lwf':
        kw['temperature'] = kwargs.get('temperature', 2.0)
        kw['lwf_lambda'] = kwargs.get('lwf_lambda', 1.0)


    kw.setdefault('weight_decay', kwargs.get('weight_decay', 0.0))
    kw.setdefault('freeze_old_heads', kwargs.get('freeze_old_heads', False))
    return cls(model, device, scenario=scenario, lr=lr, **kw)


def default_buffer_size(dataset: str, scenario: str) -> int:
    """2,000 exemplars for CIFAR-100 / TinyImageNet, 20,000 for ImageNet-1k.

    The same budget is used under TIL and CIL.  Buzzega's 200-exemplar TIL setting
    is defined for CIFAR-10 (20 per class); carried to a 100-class benchmark it is
    2 per class, which handicaps the memory methods rather than testing them.
    """
    return 20000 if dataset == 'imagenet1k' else 2000
