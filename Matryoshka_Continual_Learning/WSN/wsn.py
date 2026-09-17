"""wsn -- implementation moved here from baselines_sparse.py.

Imported by the method registry in baselines.py; baselines_sparse.py re-exports these
names so existing imports keep working.
"""
"""
Sparse / subnetwork buffer-free baseline: WSN.

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

from cl_base import ContinualMethod



def _maskable_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and not name.startswith('heads'):
            yield name, module


class _GetSubnet(torch.autograd.Function):
    """Hard top-k forward with the WSN straight-through score gradient."""

    @staticmethod
    def forward(ctx, scores, density):
        k = max(1, min(scores.numel(), int(density * scores.numel())))
        indices = torch.topk(scores.flatten(), k).indices
        mask = torch.zeros_like(scores).flatten()
        mask[indices] = 1.0
        return mask.view_as(scores)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class _WeightMask(nn.Module):
    """Learnable WSN mask parametrisation.

    During training the binary subnet is recomputed from learnable popup scores
    on every forward pass. During task evaluation ``override`` holds the exact
    stored task mask.
    """

    def __init__(self, shape, device, density=None):
        super().__init__()
        self.density = density
        self.override = None
        self.last_mask = None
        if density is None:
            # Static-mask mode retained for SpaceNet, which shares this helper.
            self.register_buffer('mask', torch.ones(shape, device=device))
            self.score = None
        else:
            self.score = nn.Parameter(torch.empty(shape, device=device))
            nn.init.kaiming_uniform_(self.score, a=math.sqrt(5))

    def current_mask(self):
        if self.density is None:
            return self.mask
        if self.override is not None:
            return self.override
        return _GetSubnet.apply(self.score.abs(), self.density)

    def forward(self, w):
        self.last_mask = self.current_mask()
        return w * self.last_mask


class WSN(ContinualMethod):
    """Winning SubNetworks (Kang et al., 2022).

    A learnable score per weight; the task's subnetwork is the top-c fraction of
    scores, applied multiplicatively in the forward pass. Weights claimed by
    earlier tasks are frozen but remain eligible for reuse by later subnetworks,
    so the accumulated subnetworks never interfere. Selection is binary -- a weight is either in the winning
    ticket or out of it -- which is the distinction SNV's real-valued phi draws.

    WSN needs the task identity to pick the subnetwork at test time, so it is a
    TIL method; the paper's CIL column shows "---" for exactly this reason.
    """

    name = 'wsn'

    def __init__(self, *args, sparsity: float = 0.1, **kw):
        super().__init__(*args, **kw)
        self.sparsity = sparsity
        self.scores: Dict[str, nn.Parameter] = {}
        self.parametrisations: Dict[str, _WeightMask] = {}
        self.accumulated: Dict[str, torch.Tensor] = {}
        self.task_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        self.task_bn: Dict[int, Dict[str, Dict]] = {}
        self._current = None
        self._install()

    def _install(self):
        for name, module in _maskable_modules(self.model):
            p = _WeightMask(module.weight.shape, self.device, self.sparsity)
            parametrize.register_parametrization(module, 'weight', p)
            self.parametrisations[name] = p
            self.scores[name] = p.score
            self.accumulated[name] = torch.zeros_like(p.score, dtype=torch.bool)

    def build_optimizer(self):
        # Popup scores live inside the registered parametrisations and therefore
        # already occur exactly once in model.parameters().
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=self.weight_decay)

    def _select_masks(self) -> Dict[str, torch.Tensor]:
        # WSN ranks the whole tensor. Reuse of a previously consolidated weight
        # is legal; the union is used only to freeze weights, not as the forward
        # mask for the next task.
        return {name: p.current_mask().detach().bool().clone()
                for name, p in self.parametrisations.items()}

    def before_task(self, task_id, train_loader, val_loader):
        for parametrisation in self.parametrisations.values():
            parametrisation.override = None
        self._current = None

    def after_backward(self, task_id):
        # Weights inside a previous task's subnetwork are not updated.
        for name, module in _maskable_modules(self.model):
            original = module.parametrizations.weight.original
            if original.grad is not None:
                original.grad.mul_((~self.accumulated[name]).float())

    def after_task(self, task_id, train_loader, val_loader):
        masks = self._select_masks()
        for name in masks:
            self.accumulated[name] |= masks[name]
            self.parametrisations[name].override = masks[name].float()
        self.task_masks[task_id] = {k: v.clone() for k, v in masks.items()}
        # Conv/Linear weights are frozen once claimed, but BatchNorm affine
        # parameters and running statistics are neither masked nor frozen; if they
        # keep drifting the "frozen" subnetwork still forgets.  Snapshot them so
        # predict() can restore this task's normalisation state.
        self.task_bn[task_id] = copy.deepcopy(
            {n: m.state_dict() for n, m in self.model.named_modules()
             if isinstance(m, nn.modules.batchnorm._BatchNorm)})
        used = sum(int(m.sum()) for m in self.accumulated.values())
        total = sum(m.numel() for m in self.accumulated.values())
        return {'capacity_used': 100.0 * used / total}

    def predict(self, x, task_id):
        if task_id is not None and task_id in self.task_masks:
            saved = {n: p.override for n, p in self.parametrisations.items()}
            for n, m in self.task_masks[task_id].items():
                self.parametrisations[n].override = m.float()
            bn_now = None
            if task_id in self.task_bn:
                bn_mods = {n: m for n, m in self.model.named_modules()
                           if isinstance(m, nn.modules.batchnorm._BatchNorm)}
                bn_now = copy.deepcopy({n: m.state_dict() for n, m in bn_mods.items()})
                for n, sd in self.task_bn[task_id].items():
                    if n in bn_mods:
                        bn_mods[n].load_state_dict(sd)
            try:
                return super().predict(x, task_id)
            finally:
                for n, m in saved.items():
                    self.parametrisations[n].override = m
                if bn_now is not None:
                    for n, sd in bn_now.items():
                        bn_mods[n].load_state_dict(sd)
        return super().predict(x, task_id)
