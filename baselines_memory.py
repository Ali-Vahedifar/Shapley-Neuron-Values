"""
Memory-based baselines: iCaRL, DER++, DyTox.

Buffer sizes follow the setup: 2,000 exemplars for CIFAR-100 and TinyImageNet
and 20,000 for ImageNet-1k under CIL; 200 exemplars under TIL.
"""

import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cl_base import ContinualMethod, Reservoir


class DERpp(ContinualMethod):
    """Dark Experience Replay++ (Buzzega et al., 2020).

    L = CE(f(x), y) + alpha * ||f(x') - z'||^2 + beta * CE(f(x''), y''),
    where (x', z') and (x'', y'') are independent draws from a reservoir buffer
    that stores the logits recorded at insertion time.
    """

    name = 'derpp'

    def __init__(self, *args, buffer_size: int = 2000, alpha: float = 0.5,
                 beta: float = 0.5, minibatch: int = 32, **kw):
        super().__init__(*args, **kw)
        self.buffer = Reservoir(buffer_size, self.device)
        self.alpha, self.beta, self.minibatch = alpha, beta, minibatch
        self._pending = None

    def extra_loss(self, x, y, logits, task_id):
        self._pending = (x, y, logits)
        if len(self.buffer) == 0:
            return torch.zeros((), device=self.device)
        loss = torch.zeros((), device=self.device)

        drawn = self.buffer.sample(self.minibatch)
        if drawn is not None and drawn[2] is not None:
            bx, _, bz, bzw = drawn
            out = self.logits(bx, task_id)
            k = min(out.shape[1], bz.shape[1])
            # Only regress against columns that were real predictions when the
            # row was stored; later columns are padding from a wider head.
            cols = torch.arange(k, device=self.device).unsqueeze(0)
            valid = cols < bzw.unsqueeze(1)
            if valid.any():
                diff = (out[:, :k] - bz[:, :k]) * valid
                loss = loss + self.alpha * diff.pow(2).sum() / valid.sum()

        drawn = self.buffer.sample(self.minibatch)
        if drawn is not None:
            bx, by, _, _ = drawn
            out = self.logits(bx, task_id)
            valid = by < out.shape[1]
            if valid.any():
                loss = loss + self.beta * F.cross_entropy(out[valid], by[valid])
        return loss

    def after_step(self, task_id):
        if self._pending is not None:
            x, y, z = self._pending
            self.buffer.add(x, y, z)
            self._pending = None


class ICaRL(ContinualMethod):
    """iCaRL (Rebuffi et al., 2017): herded exemplars, distillation, NME classifier.

    Exemplars are selected by herding on the feature mean, the memory budget is
    divided evenly across the classes seen so far, and prediction at test time is
    nearest-mean-of-exemplars in feature space rather than the linear head.
    """

    name = 'icarl'

    def __init__(self, *args, buffer_size: int = 2000, temperature: float = 2.0, **kw):
        super().__init__(*args, **kw)
        self.buffer_size = buffer_size
        self.temperature = temperature
        self.exemplars: Dict[int, torch.Tensor] = {}     # class -> stacked inputs
        self.class_means: Dict[int, torch.Tensor] = {}
        self.teacher = None
        self.old_width = 0

    def extra_loss(self, x, y, logits, task_id):
        if self.teacher is None or self.old_width == 0:
            return torch.zeros((), device=self.device)
        # Under TIL the current logits cover only head `task_id`, so the old
        # responses have to be gathered head by head rather than sliced off.
        if self.scenario == 'task_il':
            with torch.no_grad():
                old = torch.cat([self.teacher(x, t) for t in range(task_id)], dim=1)
            new = torch.cat([self.model(x, t) for t in range(task_id)], dim=1)
        else:
            with torch.no_grad():
                old = self.teacher(x)[:, :self.old_width]
            new = logits[:, :self.old_width]
        T = self.temperature
        return F.kl_div(F.log_softmax(new / T, dim=1), F.softmax(old / T, dim=1),
                        reduction='batchmean') * (T * T)

    def after_task(self, task_id, train_loader, val_loader):
        self._build_exemplars(train_loader)
        self._compute_class_means()
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.old_width = sum(self.model.heads[str(t)].out_features
                             for t in self.model.active_tasks())
        return {'exemplar_classes': len(self.exemplars)}

    @torch.no_grad()
    def _build_exemplars(self, loader):
        self.model.eval()
        by_class: Dict[int, List[torch.Tensor]] = {}
        feats: Dict[int, List[torch.Tensor]] = {}
        for batch in loader:
            x, y = batch[0].to(self.device), batch[1]
            f = F.normalize(self.model.get_features(x), dim=1).cpu()
            for i in range(y.shape[0]):
                c = int(y[i])
                by_class.setdefault(c, []).append(batch[0][i])
                feats.setdefault(c, []).append(f[i])

        for c, samples in by_class.items():
            xs = torch.stack(samples)
            fs = torch.stack(feats[c])
            target = fs.mean(0)
            chosen, running = [], torch.zeros_like(target)
            available = list(range(len(xs)))
            for step in range(len(xs)):
                cand = (target - (running + fs[available]) / (step + 1)).norm(dim=1)
                pick = available[int(cand.argmin())]
                chosen.append(pick)
                running = running + fs[pick]
                available.remove(pick)
            self.exemplars[c] = xs[chosen]

        per_class = max(1, self.buffer_size // max(len(self.exemplars), 1))
        for c in self.exemplars:
            self.exemplars[c] = self.exemplars[c][:per_class]

    @torch.no_grad()
    def _compute_class_means(self):
        self.model.eval()
        for c, xs in self.exemplars.items():
            f = F.normalize(self.model.get_features(xs.to(self.device)), dim=1)
            self.class_means[c] = F.normalize(f.mean(0), dim=0)

    def predict(self, x, task_id):
        if not self.class_means:
            return super().predict(x, task_id)
        classes = sorted(self.class_means)
        if self.scenario == 'task_il' and task_id is not None:
            width = self.model.heads[str(task_id)].out_features
            classes = [c for c in classes if task_id * width <= c < (task_id + 1) * width]
            if not classes:
                return super().predict(x, task_id)
        means = torch.stack([self.class_means[c] for c in classes]).to(self.device)
        f = F.normalize(self.model.get_features(x), dim=1)
        scores = -torch.cdist(f, means)
        if self.scenario == 'task_il':
            return scores
        full = torch.full((x.shape[0], max(classes) + 1), -1e9, device=self.device)
        full[:, torch.tensor(classes, device=self.device)] = scores
        return full


# --------------------------------------------------------------------------- #
# DyTox
# --------------------------------------------------------------------------- #
class _SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class _TaskAttentionBlock(nn.Module):
    """Task-Attention Block: a task token queries the patch tokens."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, patches, task_token):
        q = task_token.expand(patches.shape[0], -1, -1)
        kv = self.norm1(torch.cat([q, patches], dim=1))
        out = q + self.attn(self.norm1(q), kv, kv, need_weights=False)[0]
        return (out + self.mlp(self.norm2(out))).squeeze(1)


class DyToxNet(nn.Module):
    """Compact DyTox: shared ConViT-style encoder, one task token + head per task."""

    def __init__(self, image_size=32, patch_size=4, dim=384, depth=5, heads=12,
                 classes_per_task=10):
        super().__init__()
        self.dim = dim
        self.classes_per_task = classes_per_task
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([_SelfAttentionBlock(dim, heads) for _ in range(depth)])
        self.tab = _TaskAttentionBlock(dim, heads)
        self.task_tokens = nn.ParameterDict()
        self.heads = nn.ModuleDict()
        self.feature_dim = dim

    def ensure_head(self, task_id, num_classes=None):
        key = str(task_id)
        if key not in self.heads:
            device = self.pos_embed.device
            token = nn.Parameter(torch.zeros(1, 1, self.dim, device=device))
            nn.init.trunc_normal_(token, std=0.02)
            self.task_tokens[key] = token
            self.heads[key] = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, num_classes or self.classes_per_task)).to(device)
        return self.heads[key]

    def freeze_heads_before(self, task_id):
        for key in self.heads:
            flag = int(key) >= task_id
            for p in self.heads[key].parameters():
                p.requires_grad_(flag)
            self.task_tokens[key].requires_grad_(flag)

    def active_tasks(self):
        return sorted(int(k) for k in self.heads)

    def encode(self, x):
        p = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            p = blk(p)
        return p

    def get_features(self, x, task_id=None):
        patches = self.encode(x)
        t = self.active_tasks()[-1] if task_id is None else task_id
        return self.tab(patches, self.task_tokens[str(t)])

    def forward(self, x, task_id=None):
        patches = self.encode(x)
        if task_id is not None:
            return self.heads[str(task_id)](self.tab(patches, self.task_tokens[str(task_id)]))
        return torch.cat([self.heads[str(t)](self.tab(patches, self.task_tokens[str(t)]))
                          for t in self.active_tasks()], dim=1)


class DyTox(ContinualMethod):
    """DyTox (Douillard et al., 2022): dynamic task tokens plus rehearsal.

    Fidelity note: this is a compact re-implementation -- shared encoder, one
    Task-Attention Block, a task token and classifier per task, a rehearsal
    buffer and knowledge distillation.  The original additionally uses a
    ConViT/GPSA encoder, a divergence head and a finetuning phase with a
    balanced buffer; those are omitted, so absolute numbers may sit below the
    published ones.
    """

    name = 'dytox'

    def __init__(self, *args, buffer_size: int = 2000, minibatch: int = 32,
                 kd_lambda: float = 1.0, temperature: float = 2.0, **kw):
        super().__init__(*args, **kw)
        self.buffer = Reservoir(buffer_size, self.device)
        self.minibatch = minibatch
        self.kd_lambda = kd_lambda
        self.temperature = temperature
        self.teacher = None
        self.old_width = 0
        self._pending = None

    def before_task(self, task_id, train_loader, val_loader):
        if hasattr(self.model, 'freeze_heads_before'):
            self.model.freeze_heads_before(0)   # DyTox keeps every token trainable

    def extra_loss(self, x, y, logits, task_id):
        self._pending = (x, y, logits)
        loss = torch.zeros((), device=self.device)

        if self.teacher is not None and self.old_width:
            with torch.no_grad():
                old = self.teacher(x)[:, :self.old_width]
            T = self.temperature
            loss = loss + self.kd_lambda * F.kl_div(
                F.log_softmax(logits[:, :self.old_width] / T, dim=1),
                F.softmax(old / T, dim=1), reduction='batchmean') * (T * T)

        drawn = self.buffer.sample(self.minibatch)
        if drawn is not None:
            bx, by = drawn[0], drawn[1]
            out = self.logits(bx, task_id)
            valid = by < out.shape[1]
            if valid.any():
                loss = loss + F.cross_entropy(out[valid], by[valid])
        return loss

    def after_step(self, task_id):
        if self._pending is not None:
            x, y, _ = self._pending
            self.buffer.add(x, y)
            self._pending = None

    def after_task(self, task_id, train_loader, val_loader):
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.old_width = sum(self.model.heads[str(t)][-1].out_features
                             for t in self.model.active_tasks())
        return {}
