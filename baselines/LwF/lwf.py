"""lwf -- implementation lives here; baselines/regularization.py re-exports it.

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

