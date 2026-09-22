"""ewc -- implementation lives here; baselines/regularization.py re-exports it.

Imported by the method registry in baselines/__init__.py; baselines/regularization.py re-exports these
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



class EWC(ContinualMethod):
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Penalty  (lambda/2) sum_i F_i (theta_i - theta*_i)^2, with F the diagonal of
    the empirical Fisher accumulated across tasks and theta* the parameters at
    the end of the most recent task.
    """

    name = 'ewc'

    def __init__(self, *args, ewc_lambda: float = 5000.0, fisher_batches: int = 64,
                 fisher_samples: int = 256,
                 gamma: float = 1.0, **kw):
        super().__init__(*args, **kw)
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        self.fisher_batches = fisher_batches
        self.fisher_samples = fisher_samples
        self.fisher: Dict[str, torch.Tensor] = {}
        self.anchor: Dict[str, torch.Tensor] = {}

    def extra_loss(self, x, y, logits, task_id):
        if not self.fisher:
            return torch.zeros((), device=self.device)
        penalty = torch.zeros((), device=self.device)
        for name, p in self.regularized_parameters():
            if name in self.fisher:
                penalty = penalty + (self.fisher[name] * (p - self.anchor[name]).pow(2)).sum()
        return 0.5 * self.ewc_lambda * penalty

    def after_task(self, task_id, train_loader, val_loader):
        new_fisher = {n: torch.zeros_like(p) for n, p in self.regularized_parameters()}
        self.model.eval()
        samples = 0
        for batch_id, batch in enumerate(train_loader):
            if batch_id >= self.fisher_batches or samples >= self.fisher_samples:
                break
            x = batch[0].to(self.device)
            # Model-sampled diagonal Fisher: average squared per-example
            # gradients. Squaring an averaged batch gradient introduces cross
            # terms and a batch-size-dependent scale instead.
            for xi in x[:self.fisher_samples - samples]:
                self.model.zero_grad(set_to_none=True)
                out = self.logits(xi.unsqueeze(0), task_id)
                sampled = torch.multinomial(F.softmax(out, dim=1), 1).squeeze(1)
                F.cross_entropy(out, sampled).backward()
                for name, p in self.regularized_parameters():
                    if p.grad is not None:
                        new_fisher[name] += p.grad.detach().pow(2)
                samples += 1
        self.model.zero_grad(set_to_none=True)

        for name in new_fisher:
            new_fisher[name] /= max(samples, 1)
            # F_cum <- gamma * F_prev + F_new; gamma = 1.0 is plain accumulation.
            self.fisher[name] = new_fisher[name] + self.gamma * self.fisher.get(name, 0)
        self.anchor = {n: p.detach().clone() for n, p in self.regularized_parameters()}
        return {'fisher_trace': float(sum(f.sum() for f in self.fisher.values())),
                'fisher_samples': samples, 'fisher_estimator': 'per-example model-sampled',
                'variant': 'online EWC'}
