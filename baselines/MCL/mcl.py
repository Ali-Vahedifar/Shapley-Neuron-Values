"""MCL -- Matryoshka Continual Learning, on this repository's harness.

Implements the two-term objective of the manuscript's Method section exactly:

    L = sum_m c_m * CE( z^(m)|_{A_t}, y )                              (i)
      + lambda_S * sum_m w_m * tau^2 * KL( sig(zt^(m)|_{O_t}/tau) || sig(z^(m)|_{O_t}/tau) )   (ii)

with  c_m = 1/|M|  (the mean over dolls, not MRL's sum -- Appendix app:cm),
      w_m = (d/m)^beta / sum_m' (d/m')^beta                      (eq:wm),
      A_t = C_t in BOTH scenarios (sec:cil: "Under Task-IL ... A_t = C_t holds
            there by construction"),
      O_t = union of classes of tasks k < t,
      zt  = logits of the model frozen at the end of task t-1 (sec:sd), evaluated
            on the CURRENT task's inputs only -- nothing is stored.

Two terms, and nothing else.  There is deliberately NO EWC / Fisher term: the
manuscript's sec:no-ewc ablation shows the graded weight penalty *increases*
Class-IL forgetting on CIFAR-100/10 by 12.5 BWT points while buying no accuracy.
An earlier port of this file carried that penalty (from the superseded
METHOD.tex draft); it has been removed.

Head (eq:head).  One weight matrix serves every doll: the doll of width m reads
only the first m columns of W and the first m coordinates of f(x)  (MRL-E weight
tying, no extra parameters).
    Task-IL : linear,  z^(m) = W[:, :m] f[:m] + b
    Class-IL: cosine,  z^(m) = s * <W[:, :m]/||W[:, :m]||_row , f[:m]/||f[:m]||>, s = 16
Note the row norm and the feature norm are taken over the PREFIX of width m,
not over the full d coordinates.  The cosine head takes magnitude out of the
decision so an old class does not lose merely because its weight row stopped
growing (sec:cil).

Harness note.  This repo keeps one nn.Linear per task rather than one growing
C x d matrix.  Row-wise that is the same object: the cosine head normalises each
class row independently and the linear head is row-separable, so concatenating
per-task heads over a class subset is identical to slicing the rows of a single
W.  Weight tying across dolls -- the thing that matters -- holds within every
head.

Variants (``mcl`` is the method reported in the paper; the other two are the
ablation arms of the score-importance study):
    mcl          beta = 0 + FeCAM-style density readout (alpha = 0.25) + nested
                 weight aligning.  The configuration the grid search selected
                 and the one the manuscript numbers come from.
    mcl_uniform  beta = 0, no density readout, no weight aligning: the plain
                 two-term objective, every doll anchored alike.
    mcl_g        beta = 1, coarse dolls held 16x harder (sec:variants).
"""

import copy
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cl_base import ContinualMethod


def default_granularities(feature_dim: int) -> tuple:
    """Nesting widths as fractions of d: d/16 .. d.  ResNet-18 (d=512) gives
    (32, 64, 128, 256, 512); ResNet-50 (d=2048) gives (128, ..., 2048)."""
    return tuple(feature_dim // f for f in (16, 8, 4, 2, 1))


def density_features(features: torch.Tensor, width: int) -> torch.Tensor:
    """FeCAM-style stable coordinates for the exemplar-free density head:
    signed square root, then L2-normalise, on the first ``width`` dims."""
    features = features[:, :width]
    features = features.sign() * features.abs().clamp_min(1e-8).sqrt()
    return F.normalize(features, dim=1)


def _standardize(value: torch.Tensor) -> torch.Tensor:
    """Zero-mean unit-std across the class axis, so the cosine logits and the
    negative distances are on one scale before they are blended."""
    centered = value - value.mean(dim=1, keepdim=True)
    return centered / centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)


def nesting_importance(feature_dim: int, granularities: Sequence[int]) -> torch.Tensor:
    """u(j) = fraction of dolls that read coordinate j (sec:reach).

    Explanatory only: it is the reason w_m is graded, but it does not appear in
    the loss.  Kept for analysis/plotting.
    """
    idx = torch.arange(feature_dim, dtype=torch.float)
    counts = sum((idx < float(m)).float() for m in granularities)
    return counts / float(len(granularities))


def granularity_weights(granularities: Sequence[int], beta: float) -> Dict[int, float]:
    """w_m = (d/m)^beta, normalised to sum 1 (eq:wm).  beta=1 on ResNet-18 gives
    16/31, 8/31, 4/31, 2/31, 1/31."""
    widest = max(granularities)
    raw = {m: (widest / m) ** beta for m in granularities}
    total = sum(raw.values())
    return {m: v / total for m, v in raw.items()}


class MCLBase(ContinualMethod):
    """Matryoshka head + doll-weighted self-distillation.  Two terms.

    The plain objective without the density readout or weight aligning; used as
    the ``mcl_uniform`` ablation arm.  The reported method is ``MCL`` below.
    """

    name = 'mcl_uniform'

    def __init__(self, *args, granularities: Optional[Sequence[int]] = None,
                 sdft_lambda: float = 1.0, temperature: float = 2.0,
                 sdft_beta: float = 0.0, cosine_scale: float = 16.0,
                 density_alpha: float = 0.0, density_metric: str = 'ncm',
                 density_shrinkage: float = 0.1, weight_align: bool = False, **kw):
        super().__init__(*args, **kw)
        d = self.model.feature_dim
        granularities = tuple(sorted(granularities or default_granularities(d)))
        if granularities[-1] != d or min(granularities) < 1:
            raise ValueError(f'granularities {granularities} must end at feature_dim {d}')
        self.granularities = granularities
        self.sdft_lambda = sdft_lambda
        self.temperature = temperature
        self.sdft_beta = sdft_beta
        self.cosine_scale = cosine_scale
        self.weights = granularity_weights(granularities, sdft_beta)
        self.u = nesting_importance(d, granularities)
        if not 0.0 <= density_alpha <= 1.0:
            raise ValueError('density_alpha must be in [0, 1]')
        if density_metric not in ('ncm', 'diag'):
            raise ValueError("density_metric must be 'ncm' or 'diag'")
        # Density readout: one mean (and shrinkage-diagonal variance) per class,
        # in density_features space at the widest doll.  No images are kept, but
        # this IS per-class stored state -- d floats per class -- so the
        # "we store nothing" claim in sec:setting needs a carve-out for it.
        self.density_alpha = density_alpha
        self.density_metric = density_metric
        self.density_shrinkage = density_shrinkage
        self.density_stats: Dict[int, tuple] = {}
        self.weight_align = weight_align
        self.teacher: Optional[nn.Module] = None
        self._features: Optional[torch.Tensor] = None
        self._task: int = 0
        # (i) is a LOCAL cross-entropy: A_t = C_t.  The base training loop calls
        # self.criterion(out, y) with the harness's labels -- task-local under
        # Task-IL already, global under Class-IL -- so the criterion re-indexes y
        # into C_t itself.  self.logits() returns the widest doll of the current
        # head only, so the pair is consistent.
        self.criterion = self._local_ce

    # -- heads --------------------------------------------------------------- #
    def _doll(self, head: nn.Linear, features: torch.Tensor, m: int) -> torch.Tensor:
        """z^(m) through one head (eq:head).  Prefix-width normalisation for cosine."""
        w = head.weight[:, :m]
        f = features[:, :m]
        if self.scenario == 'class_il':
            w = F.normalize(w, dim=1)
            f = F.normalize(f, dim=1)
            return self.cosine_scale * F.linear(f, w)          # no bias in the cosine head
        return F.linear(f, w, head.bias)

    def _heads_of(self, model: nn.Module, tasks: List[int], features, m: int) -> torch.Tensor:
        """Logits restricted to the classes of ``tasks``, in task order."""
        return torch.cat([self._doll(model.heads[str(t)], features, m) for t in tasks], dim=1)

    def _old_tasks(self, task_id: int) -> List[int]:
        return list(range(task_id))

    # -- (i) local cross-entropy ------------------------------------------- #
    def _offset(self, task_id: int) -> int:
        """Global label of the first class of task t under Class-IL; 0 under Task-IL."""
        if self.scenario != 'class_il':
            return 0
        return sum(self.model.heads[str(t)].out_features for t in range(task_id))

    def _local_ce(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out, y - self._offset(self._task))

    def before_task(self, task_id, train_loader, val_loader):
        self._task = task_id

    def logits(self, x, task_id: int, for_training: bool = True):
        self._task = task_id
        self._features = self.model.get_features(x)
        # Widest doll, current head only  (A_t = C_t).  The base loop adds CE on
        # this; extra_loss turns it into the mean over dolls.
        return self._doll(self.model.heads[str(task_id)], self._features, self.granularities[-1])

    def _density_scores(self, features: torch.Tensor, classes: List[int]) -> torch.Tensor:
        """-distance to each stored class mean, in density_features space."""
        f = density_features(features, self.granularities[-1])
        means = torch.stack([self.density_stats[c][0] for c in classes])
        delta = f[:, None, :] - means[None, :, :]
        if self.density_metric == 'ncm':
            distance = delta.square().mean(dim=2)
        else:
            variances = torch.stack([self.density_stats[c][1] for c in classes])
            # One shared diagonal, not per-class: per-class covariances let a
            # recently-seen, tighter class win on scale alone, which is the
            # recency bias this readout exists to remove.
            shared = variances.mean(dim=0).clamp_min(1e-5)
            distance = (delta.square() / shared[None, None, :]).mean(dim=2)
        return -distance

    def predict(self, x, task_id: Optional[int]):
        """Widest doll.  Task-IL: the given head.  Class-IL: every head seen so
        far, optionally blended with the density readout."""
        features = self.model.get_features(x)
        m = self.granularities[-1]
        if self.scenario == 'task_il':
            return self._doll(self.model.heads[str(task_id)], features, m)

        tasks = self.model.active_tasks()
        cosine = self._heads_of(self.model, tasks, features, m)
        if self.density_alpha <= 0.0 or not self.density_stats:
            return cosine
        classes = sorted(self.density_stats)
        if len(classes) != cosine.shape[1]:      # stats not yet complete
            return cosine
        density = self._density_scores(features, classes)
        return ((1.0 - self.density_alpha) * _standardize(cosine)
                + self.density_alpha * _standardize(density))

    # -- loss ---------------------------------------------------------------- #
    def extra_loss(self, x, y, logits, task_id) -> torch.Tensor:
        features = self._features if self._features is not None else self.model.get_features(x)
        head = self.model.heads[str(task_id)]
        y_loc = y - self._offset(task_id)

        # (i)  mean over dolls.  The base loop already contributed CE at the
        # widest doll, so subtract it and add the full mean:  net = (1/|M|) sum_m.
        ce = [F.cross_entropy(self._doll(head, features, m), y_loc) for m in self.granularities]
        loss = sum(ce) / len(ce) - F.cross_entropy(logits, y_loc)

        # (ii) doll-weighted self-distillation over O_t, current inputs only.
        old = self._old_tasks(task_id)
        if self.teacher is not None and old and self.sdft_lambda > 0:
            with torch.no_grad():
                tf = self.teacher.get_features(x)
            tau = self.temperature
            distil = torch.zeros((), device=self.device)
            for m in self.granularities:
                with torch.no_grad():
                    zt = self._heads_of(self.teacher, old, tf, m)            # zt^(m)|_{O_t}
                zs = self._heads_of(self.model, old, features, m)            #  z^(m)|_{O_t}
                # exact KL( sig(zt/tau) || sig(zs/tau) ), batch-mean
                kl = F.kl_div(F.log_softmax(zs / tau, dim=1),
                              F.softmax(zt / tau, dim=1), reduction='batchmean')
                distil = distil + self.weights[m] * (tau ** 2) * kl
            loss = loss + self.sdft_lambda * distil
        return loss

    # -- teacher ------------------------------------------------------------- #
    @torch.no_grad()
    def _update_density_stats(self, task_id, loader) -> None:
        """One mean and shrinkage-diagonal variance per class of this task."""
        if self.density_alpha <= 0.0:
            return
        offset = self._offset(task_id)
        n_new = self.model.heads[str(task_id)].out_features
        sums, squares, counts = {}, {}, {}
        was_training = self.model.training
        self.model.eval()
        for batch in loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            f = density_features(self.model.get_features(x), self.granularities[-1])
            local = y - offset if self.scenario == 'class_il' else y
            for j in range(n_new):
                sel = f[local == j]
                if not sel.numel():
                    continue
                c = offset + j
                sums[c] = sums.get(c, 0) + sel.sum(0)
                squares[c] = squares.get(c, 0) + sel.square().sum(0)
                counts[c] = counts.get(c, 0) + sel.shape[0]
        if was_training:
            self.model.train()
        for c, n in counts.items():
            mean = sums[c] / n
            var = (squares[c] / n - mean.square()).clamp_min(1e-5)
            # Shrink unreliable coordinates toward the class-average variance.
            target = var.mean().expand_as(var)
            var = ((1.0 - self.density_shrinkage) * var
                   + self.density_shrinkage * target).clamp_min(1e-5)
            self.density_stats[c] = (mean, var)

    @torch.no_grad()
    def _apply_weight_align(self, task_id: int) -> float:
        """Nested weight aligning: rescale the new task's rows to the old norm.

        Standard WA compares whole-row norms; that is the wrong quantity for a
        Matryoshka head, because a row's norm is not the norm of the prefix any
        given doll reads.  The ratio is computed per granularity and averaged
        over the dolls, which reduces to standard WA when |M| = 1.
        """
        if not self.weight_align or task_id == 0:
            return 1.0
        new = self.model.heads[str(task_id)].weight.data
        old = [self.model.heads[str(t)].weight.data for t in range(task_id)]
        ratios = []
        for m in self.granularities:
            old_norm = torch.cat([w[:, :m].norm(dim=1) for w in old]).mean()
            new_norm = new[:, :m].norm(dim=1).mean()
            if torch.isfinite(new_norm) and new_norm.item() > 0.0:
                ratios.append(float(old_norm.item() / new_norm.item()))
        if not ratios:
            return 1.0
        gamma = float(sum(ratios) / len(ratios))
        new.mul_(gamma)
        return gamma

    def after_task(self, task_id, train_loader, val_loader) -> Dict:
        gamma = self._apply_weight_align(task_id)
        self._update_density_stats(task_id, train_loader)
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        return {'task_id': task_id, 'beta': self.sdft_beta, 'wa_gamma': gamma,
                'density_classes': len(self.density_stats),
                'granularities': str(self.granularities)}


class MCLGraded(MCLBase):
    """MCL-G: identical model, beta = 1 (coarse dolls held 16x harder)."""

    name = 'mcl_g'

    def __init__(self, *args, sdft_beta: float = 1.0, **kw):
        super().__init__(*args, sdft_beta=sdft_beta, **kw)


class MCL(MCLBase):
    """MCL: beta = 0, density readout blended at alpha = 0.25, weight aligning.

    The configuration the grid search selected and the one the manuscript's
    numbers correspond to.  Earlier drafts and the run directories of the
    CIFAR-100 campaign call this arm ``mcl3``; the method registry keeps that
    spelling as an alias so those runs stay readable.
    """

    name = 'mcl'

    def __init__(self, *args, density_alpha: float = 0.25, sdft_beta: float = 0.0,
                 weight_align: bool = True, **kw):
        super().__init__(*args, density_alpha=density_alpha, sdft_beta=sdft_beta,
                         weight_align=weight_align, **kw)


# Legacy spelling kept so result directories written as ``mcl3_*`` resolve.
MCL3 = MCL
