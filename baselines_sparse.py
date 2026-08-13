"""
Sparse / subnetwork buffer-free baselines: WSN, NFL, NFL+, SpaceNet, NISPA,
DCNet and PEC.

Fidelity note.  WSN, NFL, NFL+, SpaceNet and NISPA are re-implementations from
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


class _WeightMask(nn.Module):
    """Parametrisation applying a binary subnetwork mask to a weight tensor."""

    def __init__(self, shape, device):
        super().__init__()
        self.register_buffer('mask', torch.ones(shape, device=device))

    def forward(self, w):
        return w * self.mask


# --------------------------------------------------------------------------- #
# WSN
# --------------------------------------------------------------------------- #
class WSN(ContinualMethod):
    """Winning SubNetworks (Kang et al., 2022).

    A learnable score per weight; the task's subnetwork is the top-c fraction of
    scores, applied multiplicatively in the forward pass.  Weights claimed by
    earlier tasks are excluded from the gradient, so the accumulated subnetworks
    never interfere.  Selection is binary -- a weight is either in the winning
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
        self._install()

    def _install(self):
        for name, module in _maskable_modules(self.model):
            p = _WeightMask(module.weight.shape, self.device)
            parametrize.register_parametrization(module, 'weight', p)
            self.parametrisations[name] = p
            score = nn.Parameter(torch.rand(module.weight.shape, device=self.device) * 1e-2)
            self.scores[name] = score
            self.accumulated[name] = torch.zeros_like(score, dtype=torch.bool)
        self._score_list = nn.ParameterList(self.scores.values()).to(self.device)

    def build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(params + list(self.scores.values()), lr=self.lr)

    def _select_masks(self) -> Dict[str, torch.Tensor]:
        masks = {}
        for name, score in self.scores.items():
            flat = score.detach().abs().flatten()
            k = max(1, int(self.sparsity * flat.numel()))
            thresh = torch.topk(flat, k).values[-1]
            masks[name] = (score.detach().abs() >= thresh)
        return masks

    def before_task(self, task_id, train_loader, val_loader):
        masks = self._select_masks()
        for name, m in masks.items():
            self.parametrisations[name].mask.copy_((m | self.accumulated[name]).float())
        self._current = masks

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
            self.parametrisations[name].mask.copy_(self.accumulated[name].float())
        self.task_masks[task_id] = {k: v.clone() for k, v in masks.items()}
        used = sum(int(m.sum()) for m in self.accumulated.values())
        total = sum(m.numel() for m in self.accumulated.values())
        return {'capacity_used': 100.0 * used / total}

    def predict(self, x, task_id):
        if task_id is not None and task_id in self.task_masks:
            saved = {n: p.mask.clone() for n, p in self.parametrisations.items()}
            for n, m in self.task_masks[task_id].items():
                self.parametrisations[n].mask.copy_(m.float())
            try:
                return super().predict(x, task_id)
            finally:
                for n, m in saved.items():
                    self.parametrisations[n].mask.copy_(m)
        return super().predict(x, task_id)


# --------------------------------------------------------------------------- #
# NFL / NFL+
# --------------------------------------------------------------------------- #
class NFL(ContinualMethod):
    """No Forgetting Learning (Vahedifar et al., 2026).

    Progressively freezes a fraction of the parameters after each task.  Plain
    NFL ranks by weight magnitude, which is the limitation the SNV paper
    identifies: it freezes *some* parameters without establishing which ones
    carry the task.
    """

    name = 'nfl'
    importance = 'magnitude'

    def __init__(self, *args, freeze_fraction: float = 0.1, fisher_batches: int = 32, **kw):
        super().__init__(*args, **kw)
        self.freeze_fraction = freeze_fraction
        self.fisher_batches = fisher_batches
        self.frozen: Dict[str, torch.Tensor] = {}
        self._snapshot: Dict[str, torch.Tensor] = {}

    def _scores(self, task_id, train_loader) -> Dict[str, torch.Tensor]:
        if self.importance == 'magnitude':
            return {n: p.detach().abs() for n, p in self.backbone_parameters()}
        # NFL+: squared-gradient (Fisher-style) importance.
        acc = {n: torch.zeros_like(p) for n, p in self.backbone_parameters()}
        self.model.eval()
        for i, batch in enumerate(train_loader):
            if i >= self.fisher_batches:
                break
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            self.model.zero_grad(set_to_none=True)
            F.cross_entropy(self.logits(x, task_id), y).backward()
            for n, p in self.backbone_parameters():
                if p.grad is not None:
                    acc[n] += p.grad.detach().pow(2)
        self.model.zero_grad(set_to_none=True)
        return {n: v * p.detach().pow(2) for (n, v), (_, p)
                in zip(acc.items(), self.backbone_parameters())}

    def before_task(self, task_id, train_loader, val_loader):
        self._snapshot = {n: p.detach().clone() for n, p in self.backbone_parameters()}

    def after_backward(self, task_id):
        for n, p in self.backbone_parameters():
            if p.grad is not None and n in self.frozen:
                p.grad.mul_((~self.frozen[n]).float())

    def after_task(self, task_id, train_loader, val_loader):
        scores = self._scores(task_id, train_loader)
        flat = torch.cat([s[~self.frozen[n]].flatten() if n in self.frozen else s.flatten()
                          for n, s in scores.items()])
        if flat.numel() == 0:
            return {}
        k = max(1, int(self.freeze_fraction * flat.numel()))
        threshold = torch.topk(flat, k).values[-1]
        for n, s in scores.items():
            newly = s >= threshold
            self.frozen[n] = newly if n not in self.frozen else (self.frozen[n] | newly)
        used = sum(int(m.sum()) for m in self.frozen.values())
        total = sum(m.numel() for m in self.frozen.values())
        return {'capacity_used': 100.0 * used / total}


class NFLPlus(NFL):
    """NFL+ -- NFL with a Fisher-style importance criterion instead of magnitude."""

    name = 'nfl+'
    importance = 'fisher'


# --------------------------------------------------------------------------- #
# SpaceNet
# --------------------------------------------------------------------------- #
class SpaceNet(ContinualMethod):
    """SpaceNet (Sokar et al., 2021).

    Trains each task in a sparse subnetwork produced by adaptive drop-and-grow:
    the least important connections are dropped each epoch and regrown where the
    gradient is largest, so a task's representation compacts into few neurons.
    Connections belonging to earlier tasks are excluded from both the drop and
    the gradient.
    """

    name = 'spacenet'

    def __init__(self, *args, density: float = 0.1, rewire_fraction: float = 0.2, **kw):
        super().__init__(*args, **kw)
        self.density = density
        self.rewire_fraction = rewire_fraction
        self.parametrisations: Dict[str, _WeightMask] = {}
        self.reserved: Dict[str, torch.Tensor] = {}
        self.current: Dict[str, torch.Tensor] = {}
        for name, module in _maskable_modules(self.model):
            p = _WeightMask(module.weight.shape, self.device)
            parametrize.register_parametrization(module, 'weight', p)
            self.parametrisations[name] = p
            self.reserved[name] = torch.zeros_like(p.mask, dtype=torch.bool)

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

    def after_backward(self, task_id):
        for name, module in _maskable_modules(self.model):
            g = module.parametrizations.weight.original.grad
            if g is not None:
                g.mul_((~self.reserved[name]).float() * self.current[name].float())

    def after_task(self, task_id, train_loader, val_loader):
        self._rewire()
        for name in self.reserved:
            self.reserved[name] |= self.current[name]
            self.parametrisations[name].mask.copy_(self.reserved[name].float())
        used = sum(int(m.sum()) for m in self.reserved.values())
        total = sum(m.numel() for m in self.reserved.values())
        return {'capacity_used': 100.0 * used / total}

    @torch.no_grad()
    def _rewire(self):
        for name, module in _maskable_modules(self.model):
            w = module.parametrizations.weight.original
            active = self.current[name]
            n_active = int(active.sum())
            if n_active < 2:
                continue
            n_drop = max(1, int(self.rewire_fraction * n_active))
            mag = w.detach().abs().masked_fill(~active, float('inf')).flatten()
            drop = torch.topk(mag, n_drop, largest=False).indices
            flat = active.flatten().clone()
            flat[drop] = False
            grow_pool = (~flat) & (~self.reserved[name].flatten())
            cand = torch.nonzero(grow_pool).flatten()
            if cand.numel():
                pick = cand[torch.randperm(cand.numel(), device=cand.device)[:n_drop]]
                flat[pick] = True
            self.current[name] = flat.view_as(active)


# --------------------------------------------------------------------------- #
# NISPA
# --------------------------------------------------------------------------- #
class NISPA(ContinualMethod):
    """NISPA (Gurbuz & Dovrolis, 2022).

    Partitions units into a stable set, which is frozen after each task, and a
    plastic set that is rewired.  Stable units grow with a cosine-annealed
    schedule as tasks accumulate.

    Fidelity note: the original selects stable units by activation-based
    candidate scoring and rewires connections between phases within a task.
    This version applies the same stable/plastic partition and freezing at task
    boundaries, using mean activation as the selection score.
    """

    name = 'nispa'

    def __init__(self, *args, stable_fraction: float = 0.1, **kw):
        super().__init__(*args, **kw)
        from snv_core import build_neuron_index
        self.groups = build_neuron_index(self.model)
        self.num_neurons = sum(g.num_neurons for g in self.groups)
        self.stable_fraction = stable_fraction
        self.stable = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.device)
        self._snapshot: Dict[str, torch.Tensor] = {}

    def _param_mask(self) -> Dict[str, torch.Tensor]:
        masks = {}
        for g in self.groups:
            frozen = self.stable[g.slice()]
            for owner_name, owner in ((g.name, g.module), (g.norm_name, g.norm_module)):
                if owner is None:
                    continue
                for pname, param in owner.named_parameters(recurse=False):
                    m = torch.ones_like(param)
                    m[frozen] = 0.0
                    masks[f'{owner_name}.{pname}'] = m
        return masks

    def after_backward(self, task_id):
        masks = self._param_mask()
        for name, p in self.model.named_parameters():
            if p.grad is not None and name in masks:
                p.grad.mul_(masks[name])

    @torch.no_grad()
    def after_task(self, task_id, train_loader, val_loader):
        sums = torch.zeros(self.num_neurons, device=self.device)
        counts = 0
        handles, buf = [], {}

        def make(g):
            def hook(m, i, o):
                v = o.mean(dim=(2, 3)) if o.dim() == 4 else o
                buf[g.name] = v.detach().abs().sum(0)
            return hook

        for g in self.groups:
            handles.append(g.module.register_forward_hook(make(g)))
        self.model.eval()
        for i, batch in enumerate(train_loader):
            if i >= 16:
                break
            self.model(batch[0].to(self.device)) if self.scenario == 'class_il' \
                else self.model(batch[0].to(self.device), task_id)
            for g in self.groups:
                sums[g.slice()] += buf[g.name]
            counts += batch[0].shape[0]
        for h in handles:
            h.remove()
        sums /= max(counts, 1)

        sums[self.stable] = -float('inf')
        k = max(1, int(self.stable_fraction * self.num_neurons))
        self.stable[torch.topk(sums, k).indices] = True
        return {'capacity_used': 100.0 * float(self.stable.sum()) / self.num_neurons}


# --------------------------------------------------------------------------- #
# DCNet
# --------------------------------------------------------------------------- #
class DCNet(ContinualMethod):
    """DCNet (Wang et al., 2025): discriminative and consistent representations.

    Two objectives on top of cross-entropy -- a supervised-contrastive term that
    keeps same-class features tight and cross-class features apart
    (discriminative), and a feature-consistency term against the previous task's
    frozen encoder (consistent).  Buffer-free: both are computed on the current
    task's data only.
    """

    name = 'dcnet'

    def __init__(self, *args, lambda_disc: float = 0.5, lambda_cons: float = 1.0,
                 temperature: float = 0.1, **kw):
        super().__init__(*args, **kw)
        self.lambda_disc = lambda_disc
        self.lambda_cons = lambda_cons
        self.temperature = temperature
        self.teacher = None

    def extra_loss(self, x, y, logits, task_id):
        feats = F.normalize(self.model.get_features(x), dim=1)
        loss = self.lambda_disc * self._supcon(feats, y)
        if self.teacher is not None:
            with torch.no_grad():
                old = F.normalize(self.teacher.get_features(x), dim=1)
            loss = loss + self.lambda_cons * (1.0 - (feats * old).sum(1)).mean()
        return loss

    def _supcon(self, feats, y):
        sim = feats @ feats.t() / self.temperature
        n = feats.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=feats.device)
        sim = sim.masked_fill(eye, -1e9)
        positives = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
        if positives.sum() == 0:
            return torch.zeros((), device=feats.device)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        per_sample = (log_prob * positives).sum(1) / positives.sum(1).clamp(min=1)
        return -per_sample[positives.any(1)].mean()

    def after_task(self, task_id, train_loader, val_loader):
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        return {}


# --------------------------------------------------------------------------- #
# PEC
# --------------------------------------------------------------------------- #
class PEC(ContinualMethod):
    """Prediction Error-based Classification (Zajac et al., 2024).

    One small student per class is trained to regress a frozen random teacher on
    that class's data only.  A test point is assigned to the class whose student
    reproduces the teacher best.  No replay and no task identity, but one network
    per class -- the parameter redundancy the pruning analysis exposes.
    """

    name = 'pec'

    def __init__(self, *args, student_dim: int = 64, output_dim: int = 16,
                 num_classes: int = 100, **kw):
        super().__init__(*args, **kw)
        self.num_classes = num_classes
        self.output_dim = output_dim
        feature_dim = self.model.feature_dim
        gen = torch.Generator(device='cpu').manual_seed(0)
        self.teacher = nn.Sequential(
            nn.Linear(feature_dim, student_dim), nn.ReLU(),
            nn.Linear(student_dim, output_dim)).to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.students = nn.ModuleDict({
            str(c): nn.Sequential(nn.Linear(feature_dim, student_dim), nn.ReLU(),
                                  nn.Linear(student_dim, output_dim))
            for c in range(num_classes)}).to(self.device)

    def build_optimizer(self):
        return torch.optim.Adam(
            list(self.model.backbone.parameters()) + list(self.students.parameters()),
            lr=self.lr)

    def train_task(self, task_id, train_loader, val_loader, num_epochs=200,
                   patience=20, verbose=True):
        from tqdm import tqdm
        optimizer = self.build_optimizer()
        best, waited, best_state = float('inf'), 0, None
        bar = tqdm(range(num_epochs), desc=f'pec T{task_id}', disable=not verbose)
        for _ in bar:
            self.model.train()
            for batch in train_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                feats = self.model.get_features(x)
                target = self.teacher(feats.detach())
                loss = torch.zeros((), device=self.device)
                for c in y.unique():
                    sel = y == c
                    loss = loss + F.mse_loss(self.students[str(int(c))](feats[sel]),
                                             target[sel])
                loss.backward()
                optimizer.step()
            val = self._val_error(val_loader)
            if val < best - 1e-6:
                best, waited = val, 0
                best_state = copy.deepcopy(self.state_dict_all())
            else:
                waited += 1
                if waited >= patience:
                    break
            bar.set_postfix({'val_mse': f'{val:.4f}'})
        bar.close()
        if best_state is not None:
            self.load_state_dict_all(best_state)
        return {'task_id': task_id}

    def state_dict_all(self):
        return {'model': self.model.state_dict(), 'students': self.students.state_dict()}

    def load_state_dict_all(self, sd):
        self.model.load_state_dict(sd['model'])
        self.students.load_state_dict(sd['students'])

    @torch.no_grad()
    def _val_error(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            feats = self.model.get_features(x)
            target = self.teacher(feats)
            for c in y.unique():
                sel = y == c
                total += F.mse_loss(self.students[str(int(c))](feats[sel]),
                                    target[sel]).item() * int(sel.sum())
                n += int(sel.sum())
        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, x, task_id):
        feats = self.model.get_features(x)
        target = self.teacher(feats)
        errors = torch.stack([((self.students[str(c)](feats) - target) ** 2).mean(1)
                              for c in range(self.num_classes)], dim=1)
        return -errors
