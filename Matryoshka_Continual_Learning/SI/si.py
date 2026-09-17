"""si -- implementation moved here from baselines_regularization.py.

Imported by the method registry in baselines.py; baselines_regularization.py re-exports these
names so existing imports keep working.
"""
"""
Regularisation-based baselines: SGD (lower bound), EWC, SI, LwF.
"""

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from cl_base import ContinualMethod



class SI(ContinualMethod):
    """Synaptic Intelligence (Zenke et al., 2017).

    Per-parameter importance Omega accumulates the path integral of the loss
    decrease, -sum_step g * delta_theta, normalised by the squared total drift.
    """

    name = 'si'

    def __init__(self, *args, si_c: float = 0.1, xi: float = 1e-3, **kw):
        super().__init__(*args, **kw)
        self.si_c = si_c
        self.xi = xi
        self.omega: Dict[str, torch.Tensor] = {}
        self.anchor: Dict[str, torch.Tensor] = {}
        self._w: Dict[str, torch.Tensor] = {}
        self._prev: Dict[str, torch.Tensor] = {}
        self._task_start: Dict[str, torch.Tensor] = {}
        self._grad: Dict[str, torch.Tensor] = {}

    def before_task(self, task_id, train_loader, val_loader):
        self._w = {n: torch.zeros_like(p) for n, p in self.regularized_parameters()}
        self._prev = {n: p.detach().clone() for n, p in self.regularized_parameters()}
        self._task_start = {n: p.detach().clone() for n, p in self.regularized_parameters()}

    def extra_loss(self, x, y, logits, task_id):
        if not self.omega:
            return torch.zeros((), device=self.device)
        penalty = torch.zeros((), device=self.device)
        for name, p in self.regularized_parameters():
            if name in self.omega:
                penalty = penalty + (self.omega[name] * (p - self.anchor[name]).pow(2)).sum()
        return self.si_c * penalty

    def before_loss_backward(self, task_loss, task_id):
        params = [(n, p) for n, p in self.regularized_parameters() if p.requires_grad]
        grads = torch.autograd.grad(task_loss, [p for _, p in params],
                                    retain_graph=True, allow_unused=True)
        self._grad = {n: None if g is None else g.detach().clone()
                      for (n, _), g in zip(params, grads)}

    def training_state_dict(self):
        return {'w': self._w, 'prev': self._prev}

    def load_training_state_dict(self, state):
        if state is not None:
            self._w, self._prev = state['w'], state['prev']

    def after_step(self, task_id):
        for name, p in self.regularized_parameters():
            g = self._grad.get(name)
            if g is None:
                continue
            delta = p.detach() - self._prev[name]
            self._w[name] -= g * delta
            self._prev[name] = p.detach().clone()

    def after_task(self, task_id, train_loader, val_loader):
        for name, p in self.regularized_parameters():
            drift = p.detach() - self._task_start[name]
            contrib = self._w[name] / (drift.pow(2) + self.xi)
            self.omega[name] = self.omega.get(name, torch.zeros_like(p)) + contrib.clamp(min=0)
        self.anchor = {n: p.detach().clone() for n, p in self.regularized_parameters()}
        return {'omega_trace': float(sum(o.sum() for o in self.omega.values()))}
