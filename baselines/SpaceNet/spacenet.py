"""spacenet -- implementation lives here; baselines/sparse.py re-exports it.

Imported by the method registry in baselines/__init__.py; baselines/sparse.py re-exports these
names so existing imports keep working.
"""
"""
Sparse / subnetwork buffer-free baseline: SpaceNet.


Fidelity note.  WSN, SpaceNet and NISPA are re-implementations from
their published descriptions on the shared backbone and training loop used
throughout this repo, so that a comparison isolates the algorithm rather than
the surrounding infrastructure.  Where an original relies on machinery this
repo does not carry, the docstring says so.
"""

import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from wsn_helpers import _WeightMask, _maskable_modules

from cl_base import ContinualMethod



class SpaceNet(ContinualMethod):
    """SpaceNet (Sokar et al., 2021).

    Trains each task in a sparse subnetwork produced by adaptive drop-and-grow:
    the least important connections are dropped each epoch and regrown where the
    gradient is largest, so a task's representation compacts into few neurons.
    Connections belonging to earlier tasks are excluded from both the drop and
    the gradient.
    """

    name = 'spacenet'

    def __init__(self, *args, density: float = 0.1, rewire_fraction: float = 0.2,
                 lambda_sp: float = 1.0, **kw):
        super().__init__(*args, **kw)
        self.density = density
        self.rewire_fraction = rewire_fraction
        self.lambda_sp = lambda_sp
        self.parametrisations: Dict[str, _WeightMask] = {}
        self.reserved: Dict[str, torch.Tensor] = {}
        self.current: Dict[str, torch.Tensor] = {}
        self.weight_importance: Dict[str, torch.Tensor] = {}
        self._pre_step: Dict[str, torch.Tensor] = {}
        self._step_grad: Dict[str, torch.Tensor] = {}
        for name, module in _maskable_modules(self.model):
            p = _WeightMask(module.weight.shape, self.device)
            parametrize.register_parametrization(module, 'weight', p)
            self.parametrisations[name] = p
            self.reserved[name] = torch.zeros_like(p.mask, dtype=torch.bool)
            self.weight_importance[name] = torch.zeros_like(p.mask)

    def before_task(self, task_id, train_loader, val_loader):
        for name, p in self.parametrisations.items():
            free = ~self.reserved[name]
            n_free = int(free.sum())
            k = max(1, int(self.density * p.mask.numel()))
            k = min(k, n_free)
            idx = torch.nonzero(free.flatten()).flatten()
            pick = idx[torch.randperm(idx.numel(), device=idx.device)[:k]]
            m = torch.zeros_like(p.mask, dtype=torch.bool).flatten()
            m[pick] = True
            self.current[name] = m.view_as(p.mask)
            p.mask.copy_((self.current[name] | self.reserved[name]).float())
            self.weight_importance[name].zero_()

    def after_backward(self, task_id):
        for name, module in _maskable_modules(self.model):
            g = module.parametrizations.weight.original.grad
            if g is not None:
                g.mul_((~self.reserved[name]).float() * self.current[name].float())
                # SpaceNet's connection importance is accumulated as
                # |(w_after - w_before) * grad|.  Retain the pre-update values
                # and masked gradients for after_step().
                self._pre_step[name] = module.parametrizations.weight.original.detach().clone()
                self._step_grad[name] = g.detach().clone()

    @torch.no_grad()
    def after_step(self, task_id):
        for name, module in _maskable_modules(self.model):
            if name not in self._pre_step:
                continue
            w = module.parametrizations.weight.original
            delta = w.detach() - self._pre_step[name]
            self.weight_importance[name].add_(
                (delta * self._step_grad[name]).abs() * self.current[name])
            # Adam weight decay can alter reserved entries even with zeroed
            # gradients, so restore them exactly after every optimiser step.
            old = self._pre_step[name]
            w[self.reserved[name]] = old[self.reserved[name]]
        self._pre_step.clear()
        self._step_grad.clear()

    def after_epoch(self, task_id, epoch):
        # The reference implementation performs drop-and-grow after every
        # epoch except the final one, using importance accumulated during the
        # task.  This hook is called only when another epoch will follow.
        self._rewire()

    def after_task(self, task_id, train_loader, val_loader):
        for name in self.reserved:
            self.reserved[name] |= self.current[name]
            self.parametrisations[name].mask.copy_(self.reserved[name].float())
        used = sum(int(m.sum()) for m in self.reserved.values())
        total = sum(m.numel() for m in self.reserved.values())
        return {'capacity_used': 100.0 * used / total}

    def after_best_state_loaded(self, task_id):
        # ``mask`` is a registered buffer and is restored with the best model;
        # ``current`` is ordinary Python state and must follow it.
        for name, p in self.parametrisations.items():
            self.current[name] = p.mask.bool() & ~self.reserved[name]

    @torch.no_grad()
    def _rewire(self):
        for name, module in _maskable_modules(self.model):
            w = module.parametrizations.weight.original
            active = self.current[name]
            n_active = int(active.sum())
            if n_active < 2:
                continue
            n_drop = max(1, int(self.rewire_fraction * n_active))
            importance = self.weight_importance[name]
            ranked = importance.masked_fill(~active, float('inf')).flatten()
            drop = torch.topk(ranked, n_drop, largest=False).indices
            flat = active.flatten().clone()
            flat[drop] = False
            grow_pool = (~flat) & (~self.reserved[name].flatten())
            cand = torch.nonzero(grow_pool).flatten()
            if cand.numel():
                # The paper grows connections between important endpoint
                # neurons.  Reduce convolution kernels to an out-by-in matrix,
                # form endpoint importance, then broadcast back to kernels.
                reduced = importance
                if reduced.ndim > 2:
                    reduced = reduced.sum(dim=tuple(range(2, reduced.ndim)))
                out_score = reduced.sum(dim=1, keepdim=True)
                in_score = reduced.sum(dim=0, keepdim=True)
                endpoint = out_score * in_score
                while endpoint.ndim < importance.ndim:
                    endpoint = endpoint.unsqueeze(-1)
                endpoint = endpoint.expand_as(importance).flatten()
                take = min(n_drop, cand.numel())
                pick = cand[torch.topk(endpoint[cand], take, largest=True).indices]
                flat[pick] = True
            self.current[name] = flat.view_as(active)
            self.parametrisations[name].mask.copy_(
                (self.current[name] | self.reserved[name]).float())
