"""
In-Run Neuron Shapley -- trajectory-based neuron valuation in one training run.

The neuron analogue of In-Run Data Shapley (Wang et al., ICLR 2025).  Where
``snv_core`` values a neuron by *ablating* it in the finished model, this values
a neuron by how much its own updates reduced the task's validation loss over the
course of training.

Construction.  At training step s the parameters move by Delta theta^(s).  A
first-order expansion of the validation loss gives

    L_val(theta^(s+1)) - L_val(theta^(s))  ~=  < grad L_val(theta^(s)), Delta theta^(s) >

and because parameters partition by neuron -- a filter owns its weights, its
bias, and the affine parameters of the norm that scales it -- that inner product
decomposes exactly into a sum of per-neuron blocks.  The per-step game over
neurons is therefore *additive*, and an additive game's Shapley value is just
the player's own contribution.  No subsets, no permutations:

    phi_run_i  =  - sum_s < grad_{theta_i} L_val(theta^(s)), Delta theta_i^(s) >

The sign is chosen so that larger means "reduced validation loss more", matching
the direction of phi_abl.

Two deliberate choices:

*   The realised ``Delta theta`` is used rather than ``-eta * grad L_train``.
    The published derivation assumes SGD; with Adam the step is not proportional
    to the gradient, and using the actual displacement keeps the expansion exact
    to first order for whatever optimiser is in use.

*   The validation gradient is taken in eval mode, so BatchNorm uses running
    statistics.  That is the loss the metric actually reports.

A frozen neuron has ``Delta theta_i = 0`` by construction, so ``phi_run_i`` is
exactly zero -- the valuation restricts itself to the plastic pool with no
availability mask.  ``test_snv.py`` pins that property.

Cost: one extra backward pass over a small validation batch every
``val_every`` steps, plus two parameter-sized buffers.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class InRunNeuronShapley:
    """Accumulates phi_run over a task's training run.

    Usage inside a training loop::

        tracker.begin_step()          # snapshot theta, refresh grad L_val if due
        ...   loss.backward(); optimizer.step()   ...
        tracker.end_step()            # accumulate the per-neuron contribution
    """

    def __init__(
        self,
        model: nn.Module,
        groups: Sequence,
        device: torch.device,
        val_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        task_id: Optional[int] = None,
        scenario: str = 'class_il',
        val_every: int = 1,
        criterion: Optional[nn.Module] = None,
    ):
        self.model = model
        self.groups = list(groups)
        self.device = device
        self.val_batches = [(x.to(device), y.to(device)) for x, y in val_batches]
        self.task_id = task_id
        self.scenario = scenario
        self.val_every = max(1, val_every)
        self.criterion = criterion or nn.CrossEntropyLoss()

        self.num_neurons = sum(g.num_neurons for g in self.groups)
        self.phi = torch.zeros(self.num_neurons, dtype=torch.float64, device=device)

        self._tracked = self._tracked_parameters()
        self._val_grad: Dict[str, torch.Tensor] = {}
        self._theta_before: Dict[str, torch.Tensor] = {}
        self.steps = 0
        self.val_refreshes = 0

    # -- bookkeeping -------------------------------------------------------- #
    def _tracked_parameters(self) -> Dict[str, Tuple[torch.nn.Parameter, int, int]]:
        """Map parameter name -> (parameter, neuron start, neuron end).

        Only backbone parameters that belong to a neuron are tracked; task heads
        are not neurons and carry no phi.
        """
        tracked = {}
        for g in self.groups:
            for owner_name, owner in ((g.name, g.module), (g.norm_name, g.norm_module)):
                if owner is None:
                    continue
                for pname, param in owner.named_parameters(recurse=False):
                    name = f'{owner_name}.{pname}' if owner_name else pname
                    tracked[name] = (param, g.start, g.end)
        return tracked

    def _forward(self, x):
        head = self.task_id if self.scenario == 'task_il' else None
        if head is None:
            return self.model(x)
        try:
            return self.model(x, head)
        except TypeError:
            return self.model(x)

    # -- validation gradient ------------------------------------------------ #
    def _refresh_val_grad(self) -> None:
        """grad L_val(theta^(s)), taken in eval mode."""
        was_training = self.model.training
        self.model.eval()

        # Every parameter has to be saved and restored, not only the tracked
        # ones: the validation backward writes into the task heads too, and
        # leaving that behind would add the validation gradient to the training
        # step.
        saved = {name: (p.grad.detach().clone() if p.grad is not None else None)
                 for name, p in self.model.named_parameters()}
        for _, p in self.model.named_parameters():
            p.grad = None

        total = sum(y.numel() for _, y in self.val_batches)
        for x, y in self.val_batches:
            loss = self.criterion(self._forward(x), y) * (y.numel() / max(total, 1))
            loss.backward()

        self._val_grad = {name: (p.grad.detach().clone() if p.grad is not None
                                 else torch.zeros_like(p))
                          for name, (p, _, _) in self._tracked.items()}

        for name, p in self.model.named_parameters():
            p.grad = saved[name]
        if was_training:
            self.model.train()
        self.val_refreshes += 1

    # -- step hooks --------------------------------------------------------- #
    def begin_step(self) -> None:
        if self.steps % self.val_every == 0:
            self._refresh_val_grad()
        self._theta_before = {name: p.detach().clone()
                              for name, (p, _, _) in self._tracked.items()}

    @torch.no_grad()
    def end_step(self) -> None:
        """phi_i -= < grad_{theta_i} L_val, Delta theta_i >, blockwise per neuron."""
        if not self._theta_before or not self._val_grad:
            self.steps += 1
            return

        for name, (param, start, end) in self._tracked.items():
            gval = self._val_grad.get(name)
            before = self._theta_before.get(name)
            if gval is None or before is None:
                continue
            delta = param.detach() - before
            if not torch.any(delta):
                continue
            # dim 0 of every tracked parameter indexes the neuron.
            block = (gval * delta).reshape(delta.shape[0], -1).sum(dim=1)
            self.phi[start:end] -= block.double()

        self._theta_before = {}
        self.steps += 1

    # -- results ------------------------------------------------------------ #
    @property
    def values(self) -> torch.Tensor:
        return self.phi.float()

    def select_top_k_neurons(self, sparsity_ratio: float) -> torch.Tensor:
        """S_t as the top floor(c*N) neurons by phi_run."""
        import math
        k = max(1, min(int(math.floor(sparsity_ratio * self.num_neurons)), self.num_neurons))
        mask = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.phi.device)
        mask[torch.topk(self.phi, k).indices] = True
        return mask

    def summary(self) -> Dict[str, float]:
        nonzero = int((self.phi != 0).sum())
        return {
            'steps': self.steps,
            'val_refreshes': self.val_refreshes,
            'nonzero_neurons': nonzero,
            'zero_neurons': self.num_neurons - nonzero,
            'phi_sum': float(self.phi.sum()),
            'phi_max': float(self.phi.max()),
            'phi_min': float(self.phi.min()),
        }


# --------------------------------------------------------------------------- #
# Comparing the two valuations
# --------------------------------------------------------------------------- #
def _rankdata(x: torch.Tensor) -> torch.Tensor:
    """Average ranks, ties shared -- the ranking Spearman needs."""
    x = x.detach().cpu().double()
    n = x.numel()
    order = torch.argsort(x)
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float64)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().double()
    b = b.detach().cpu().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float('nan')


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def top_k_agreement(a: torch.Tensor, b: torch.Tensor, k: int) -> Dict[str, float]:
    """Overlap of the two top-k sets -- the quantity that decides what gets frozen."""
    k = max(1, min(k, a.numel()))
    sa = set(torch.topk(a, k).indices.tolist())
    sb = set(torch.topk(b, k).indices.tolist())
    inter = len(sa & sb)
    return {
        'k': k,
        'overlap': inter,
        'overlap_frac': inter / k,
        'jaccard': inter / len(sa | sb),
        'only_ablation': sorted(sa - sb),
        'only_inrun': sorted(sb - sa),
    }


def compare_valuations(phi_abl: torch.Tensor, phi_run: torch.Tensor, groups,
                       budgets: Sequence[float] = (0.03, 0.05, 0.1, 0.3, 0.5),
                       restrict: Optional[torch.Tensor] = None) -> Dict:
    """Full agreement report between the ablation and trajectory valuations.

    ``restrict`` is an optional boolean mask limiting the comparison to a subset
    of neurons -- pass the plastic pool for tasks after the first, where frozen
    neurons have phi_run = 0 by construction and would otherwise drag the
    correlation down for a reason that has nothing to do with the valuations
    disagreeing.
    """
    a, b = phi_abl.detach().cpu(), phi_run.detach().cpu()
    if restrict is not None:
        restrict = restrict.detach().cpu()
        a, b = a[restrict], b[restrict]

    report = {
        'n_neurons': int(a.numel()),
        'spearman': spearman(a, b),
        'pearson': _pearson(a, b),
        'top_k': {str(c): top_k_agreement(a, b, int(c * a.numel())) for c in budgets},
    }

    if restrict is None:
        per_layer = []
        for g in groups:
            ga, gb = phi_abl[g.slice()].cpu(), phi_run[g.slice()].cpu()
            per_layer.append({
                'layer': g.name,
                'num_neurons': g.num_neurons,
                'spearman': spearman(ga, gb) if g.num_neurons > 2 else float('nan'),
                'mean_abl': float(ga.mean()),
                'mean_run': float(gb.mean()),
            })
        report['per_layer'] = per_layer

    return report


def verdict(spearman_value: float) -> str:
    """Plain reading of the falsification test."""
    if spearman_value != spearman_value:                      # NaN
        return 'undefined -- one valuation is constant'
    if spearman_value >= 0.85:
        return ('HIGH agreement: the two valuations measure essentially the same thing. '
                'A dual-criterion paper does not survive this; the contribution would '
                'reduce to making the estimator cheaper.')
    if spearman_value >= 0.6:
        return ('MODERATE agreement: partly redundant. Worth checking whether the '
                'disagreeing neurons are the ones that matter for capacity.')
    return ('LOW agreement: the valuations rank neurons differently. The '
            'disagreement is the contribution -- characterise it and show it '
            'changes the continual-learning outcome.')
