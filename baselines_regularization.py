"""
Regularisation-based baselines: SGD (lower bound), EWC, SI, LwF.
"""

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from cl_base import ContinualMethod


class SGDBaseline(ContinualMethod):
    """Plain sequential fine-tuning -- the lower bound."""

    name = 'sgd'


class EWC(ContinualMethod):
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Penalty  (lambda/2) sum_i F_i (theta_i - theta*_i)^2, with F the diagonal of
    the empirical Fisher accumulated across tasks and theta* the parameters at
    the end of the most recent task.
    """

    name = 'ewc'

    def __init__(self, *args, ewc_lambda: float = 5000.0, fisher_batches: int = 64, **kw):
        super().__init__(*args, **kw)
        self.ewc_lambda = ewc_lambda
        self.fisher_batches = fisher_batches
        self.fisher: Dict[str, torch.Tensor] = {}
        self.anchor: Dict[str, torch.Tensor] = {}

    def extra_loss(self, x, y, logits, task_id):
        if not self.fisher:
            return torch.zeros((), device=self.device)
        penalty = torch.zeros((), device=self.device)
        for name, p in self.backbone_parameters():
            if name in self.fisher:
                penalty = penalty + (self.fisher[name] * (p - self.anchor[name]).pow(2)).sum()
        return 0.5 * self.ewc_lambda * penalty

    def after_task(self, task_id, train_loader, val_loader):
        new_fisher = {n: torch.zeros_like(p) for n, p in self.backbone_parameters()}
        self.model.eval()
        batches = 0
        for batch in train_loader:
            if batches >= self.fisher_batches:
                break
            x = batch[0].to(self.device)
            self.model.zero_grad(set_to_none=True)
            out = self.logits(x, task_id)
            # Empirical Fisher: gradient of the log-likelihood of the model's own draw.
            sampled = torch.multinomial(F.softmax(out, dim=1), 1).squeeze(1)
            F.cross_entropy(out, sampled).backward()
            for name, p in self.backbone_parameters():
                if p.grad is not None:
                    new_fisher[name] += p.grad.detach().pow(2)
            batches += 1
        self.model.zero_grad(set_to_none=True)

        for name in new_fisher:
            new_fisher[name] /= max(batches, 1)
            self.fisher[name] = new_fisher[name] + self.fisher.get(name, 0)
        self.anchor = {n: p.detach().clone() for n, p in self.backbone_parameters()}
        return {'fisher_trace': float(sum(f.sum() for f in self.fisher.values()))}


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
        self._w = {n: torch.zeros_like(p) for n, p in self.backbone_parameters()}
        self._prev = {n: p.detach().clone() for n, p in self.backbone_parameters()}
        self._task_start = {n: p.detach().clone() for n, p in self.backbone_parameters()}

    def extra_loss(self, x, y, logits, task_id):
        if not self.omega:
            return torch.zeros((), device=self.device)
        penalty = torch.zeros((), device=self.device)
        for name, p in self.backbone_parameters():
            if name in self.omega:
                penalty = penalty + (self.omega[name] * (p - self.anchor[name]).pow(2)).sum()
        return self.si_c * penalty

    def after_backward(self, task_id):
        self._grad = {n: (p.grad.detach().clone() if p.grad is not None else None)
                      for n, p in self.backbone_parameters()}

    def after_step(self, task_id):
        for name, p in self.backbone_parameters():
            g = self._grad.get(name)
            if g is None:
                continue
            delta = p.detach() - self._prev[name]
            self._w[name] -= g * delta
            self._prev[name] = p.detach().clone()

    def after_task(self, task_id, train_loader, val_loader):
        for name, p in self.backbone_parameters():
            drift = p.detach() - self._task_start[name]
            contrib = self._w[name] / (drift.pow(2) + self.xi)
            self.omega[name] = self.omega.get(name, torch.zeros_like(p)) + contrib.clamp(min=0)
        self.anchor = {n: p.detach().clone() for n, p in self.backbone_parameters()}
        return {'omega_trace': float(sum(o.sum() for o in self.omega.values()))}


class LwF(ContinualMethod):
    """Learning without Forgetting (Li & Hoiem, 2017).

    Distils the previous model's responses on the *current* task's data into the
    old heads, so no stored exemplars are needed.
    """

    name = 'lwf'

    def __init__(self, *args, lwf_lambda: float = 1.0, temperature: float = 2.0, **kw):
        super().__init__(*args, **kw)
        self.lwf_lambda = lwf_lambda
        self.temperature = temperature
        self.teacher = None
        self.old_width = 0

    def extra_loss(self, x, y, logits, task_id):
        if self.teacher is None or self.old_width == 0:
            return torch.zeros((), device=self.device)
        with torch.no_grad():
            if self.scenario == 'task_il':
                old = torch.cat([self.teacher(x, t) for t in range(task_id)], dim=1)
            else:
                old = self.teacher(x)[:, :self.old_width]
        if self.scenario == 'task_il':
            new = torch.cat([self.model(x, t) for t in range(task_id)], dim=1)
        else:
            new = logits[:, :self.old_width]
        T = self.temperature
        kd = F.kl_div(F.log_softmax(new / T, dim=1), F.softmax(old / T, dim=1),
                      reduction='batchmean') * (T * T)
        return self.lwf_lambda * kd

    def after_task(self, task_id, train_loader, val_loader):
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.old_width = sum(self.model.heads[str(t)].out_features
                             for t in self.model.active_tasks())
        return {}
