"""
Shapley Neuron Valuation (SNV) for Continual Learning -- core implementation.

Implements the method of:
    "Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?"

Correspondence with the paper:

  Section 2          A "Neuron" m_i is a convolutional filter (or a hidden unit
                     of a non-classifier Linear layer).  N = sum_l C_l.
  Section 2          V(S) is the model's accuracy after every neuron in M \\ S has
                     had its output replaced by its mean response over the
                     validation data.  The model is never retrained.
  Eq. (9)            phi_i = E_pi[ V(S_i^pi u {i}) - V(S_i^pi) ]  (Monte Carlo).
  Section 2.4 (ii)   Truncation: marginals are skipped while V(S) <= tau.
  Section 2.4 (iii)  Multi-armed bandit: A <- {i : |phi_i - phi^(k)| < delta_i},
                     delta_i = z_alpha * sigma_i / sqrt(n_i); loop while A != {}.
  Section 2.2        S_t = top-k of phi with k = floor(c * N).
  Section 2.3        B_t = B_{t-1} u S_t;  theta <- theta - eta (dL/dtheta . M_{t-1}).
  Fig. 1             A neuron may enter the top-r% for several tasks, so top-k is
                     taken over all N neurons -- selection is NOT restricted to
                     neurons that are still unfrozen.
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import time as _time

import torch
import torch.nn as nn
from tqdm import tqdm

_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)


# --------------------------------------------------------------------------- #
# Neuron indexing
# --------------------------------------------------------------------------- #
@dataclass
class NeuronGroup:
    """One layer's worth of neurons, plus the norm layer that scales them.

    The paper defines the freezing mask over "all trainable weights" belonging to
    a neuron.  For a convolutional filter that includes the affine parameters of
    the BatchNorm that immediately follows it: a frozen filter whose gamma/beta
    keep training does not compute a fixed function, so forgetting is not
    prevented.  ``norm_module`` records that pairing.
    """

    name: str
    module: nn.Module
    kind: str            # 'conv' | 'linear'
    num_neurons: int
    start: int
    end: int
    norm_name: Optional[str] = None
    norm_module: Optional[nn.Module] = None

    def slice(self) -> slice:
        return slice(self.start, self.end)


def _is_classifier(name: str) -> bool:
    tail = name.split('.')[-1]
    return tail in ('fc', 'classifier') or 'head' in name


def build_neuron_index(model: nn.Module) -> List[NeuronGroup]:
    """Enumerate the neurons of ``model`` and pair each layer with its norm.

    Neurons are ordered by module registration order, which is the order used
    for every mask in this file.  The final classifier is excluded -- it is
    task-specific, not part of the shared backbone the masks protect.
    """
    groups: List[NeuronGroup] = []
    idx = 0
    pending: Optional[NeuronGroup] = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            pending = NeuronGroup(name, module, 'conv', module.out_channels, idx,
                                  idx + module.out_channels)
            groups.append(pending)
            idx += module.out_channels
        elif isinstance(module, nn.Linear) and not _is_classifier(name):
            pending = NeuronGroup(name, module, 'linear', module.out_features, idx,
                                  idx + module.out_features)
            groups.append(pending)
            idx += module.out_features
        elif isinstance(module, _NORM_TYPES) and pending is not None:
            n_feat = getattr(module, 'num_features', None) or getattr(module, 'num_channels', None)
            if n_feat == pending.num_neurons and pending.norm_module is None:
                pending.norm_name = name
                pending.norm_module = module
                pending = None

    return groups


def _total_weights(groups: List[NeuronGroup]) -> int:
    return sum(g.module.weight.numel() for g in groups)


def subnetwork_weight_fraction(groups: List[NeuronGroup], mask: torch.Tensor) -> float:
    """Fraction (0-1) of backbone weights owned by the neurons in ``mask``."""
    return _owned_weights(groups, mask) / max(_total_weights(groups), 1)


def _owned_weights(groups: List[NeuronGroup], mask: torch.Tensor) -> int:
    """Weights on the input rows of the neurons selected by ``mask``.

    A neuron owns its whole row -- every incoming weight -- which is the set the
    gradient mask in ``create_gradient_mask`` actually pins, and therefore the
    set directly comparable with the weights WSN marks as used.
    """
    total = 0
    for g in groups:
        kept = int(mask[g.slice()].sum().item())
        if kept:
            total += kept * (g.module.weight.numel() // g.num_neurons)
    return total


class NeuronMaskManager:
    """Tracks B_t and turns it into parameter-level freezing masks M_t.

    Also owns the *exactness* guarantee: a frozen neuron's weights, biases,
    norm affine parameters and norm running statistics are all restored to the
    values they held when the neuron was frozen.  Gradient masking alone leaves
    BatchNorm running statistics free to drift, because they are updated by the
    forward pass rather than by the optimizer.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.groups = build_neuron_index(model)
        self.num_neurons = sum(g.num_neurons for g in self.groups)

        self.cumulative_mask = torch.zeros(self.num_neurons, dtype=torch.bool, device=device)
        self.task_masks: Dict[int, torch.Tensor] = {}

        self._mask_cache: Optional[Dict[str, torch.Tensor]] = None
        self._frozen_snapshot: Dict[str, torch.Tensor] = {}

    # -- masks ------------------------------------------------------------- #
    def create_gradient_mask(self) -> Dict[str, torch.Tensor]:
        """Parameter-level mask M_{t-1}, keyed by exact ``named_parameters`` name.

        ``(M)_j = 0`` when weight theta_j belongs to a neuron m_i with
        ``B_{t-1}[i] = 1``, and 1 otherwise.
        """
        if self._mask_cache is not None:
            return self._mask_cache

        masks: Dict[str, torch.Tensor] = {}
        for g in self.groups:
            frozen_rows = self.cumulative_mask[g.slice()]
            for owner_name, owner in ((g.name, g.module), (g.norm_name, g.norm_module)):
                if owner is None:
                    continue
                for pname, param in owner.named_parameters(recurse=False):
                    full = f"{owner_name}.{pname}" if owner_name else pname
                    mask = torch.ones_like(param)
                    mask[frozen_rows] = 0.0          # dim 0 == out_channels / out_features
                    masks[full] = mask

        self._mask_cache = masks
        return masks

    def apply_gradient_mask(self) -> None:
        """theta <- theta - eta (dL/dtheta . M_{t-1}); zeroes the masked gradients."""
        masks = self.create_gradient_mask()
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            mask = masks.get(name)
            if mask is not None:
                param.grad.mul_(mask)

    # -- exactness --------------------------------------------------------- #
    def snapshot_frozen_state(self) -> None:
        """Record every frozen quantity so it can be restored bit-for-bit."""
        snap: Dict[str, torch.Tensor] = {}
        for g in self.groups:
            frozen_rows = self.cumulative_mask[g.slice()]
            if not bool(frozen_rows.any()):
                continue
            for owner_name, owner in ((g.name, g.module), (g.norm_name, g.norm_module)):
                if owner is None:
                    continue
                for pname, param in owner.named_parameters(recurse=False):
                    snap[f"{owner_name}.{pname}"] = param.detach()[frozen_rows].clone()
                for bname, buf in owner.named_buffers(recurse=False):
                    if buf is None or buf.dim() == 0 or buf.shape[0] != g.num_neurons:
                        continue
                    snap[f"{owner_name}.{bname}"] = buf.detach()[frozen_rows].clone()
        self._frozen_snapshot = snap

    def restore_frozen_state(self) -> None:
        """Undo any drift in frozen weights, norm affines and norm running stats."""
        if not self._frozen_snapshot:
            return
        with torch.no_grad():
            for g in self.groups:
                frozen_rows = self.cumulative_mask[g.slice()]
                if not bool(frozen_rows.any()):
                    continue
                for owner_name, owner in ((g.name, g.module), (g.norm_name, g.norm_module)):
                    if owner is None:
                        continue
                    for pname, param in owner.named_parameters(recurse=False):
                        saved = self._frozen_snapshot.get(f"{owner_name}.{pname}")
                        if saved is not None:
                            param[frozen_rows] = saved
                    for bname, buf in owner.named_buffers(recurse=False):
                        saved = self._frozen_snapshot.get(f"{owner_name}.{bname}")
                        if saved is not None:
                            buf[frozen_rows] = saved

    def max_frozen_drift(self) -> float:
        """Largest absolute change in any frozen quantity since the snapshot."""
        worst = 0.0
        for g in self.groups:
            frozen_rows = self.cumulative_mask[g.slice()]
            if not bool(frozen_rows.any()):
                continue
            for owner_name, owner in ((g.name, g.module), (g.norm_name, g.norm_module)):
                if owner is None:
                    continue
                for pname, param in owner.named_parameters(recurse=False):
                    saved = self._frozen_snapshot.get(f"{owner_name}.{pname}")
                    if saved is not None:
                        worst = max(worst, (param.detach()[frozen_rows] - saved).abs().max().item())
                for bname, buf in owner.named_buffers(recurse=False):
                    saved = self._frozen_snapshot.get(f"{owner_name}.{bname}")
                    if saved is not None:
                        worst = max(worst, (buf.detach()[frozen_rows] - saved).abs().max().item())
        return worst

    # -- bookkeeping ------------------------------------------------------- #
    def update_cumulative_mask(self, task_id: int, task_mask: torch.Tensor) -> None:
        """B_t <- B_{t-1} u S_t."""
        task_mask = task_mask.to(self.cumulative_mask.device)
        self.task_masks[task_id] = task_mask.clone()
        self.cumulative_mask = self.cumulative_mask | task_mask
        self._mask_cache = None

    def get_available_neurons(self) -> torch.Tensor:
        return ~self.cumulative_mask

    def get_capacity_used(self) -> float:
        return self.cumulative_mask.sum().item() / self.num_neurons * 100.0

    def weight_capacity_used(self) -> float:
        """Percentage of backbone weights owned by the frozen neurons.

        ``get_capacity_used`` counts *neurons*, WSN's capacity counts *weights*,
        and the two are not interchangeable: 4800 neurons are spread over layers
        whose rows differ in width by three orders of magnitude, so a top-k over
        neurons does not claim k/N of the parameters.  Reporting both is what
        makes an SNV-vs-WSN capacity column mean the same thing in each row.
        """
        return 100.0 * _owned_weights(self.groups, self.cumulative_mask) / max(
            _total_weights(self.groups), 1)

    def task_weight_capacity(self, task_id: int) -> float:
        """The same measure for one task's S_t alone."""
        mask = self.task_masks.get(task_id)
        if mask is None:
            return float('nan')
        return 100.0 * _owned_weights(self.groups, mask) / max(_total_weights(self.groups), 1)

    def reuse_stats(self, task_id: int) -> Dict[str, int]:
        """How much of S_t was already frozen (the sharing Fig. 1 describes)."""
        s_t = self.task_masks[task_id]
        previously = torch.zeros_like(s_t)
        for prev, mask in self.task_masks.items():
            if prev < task_id:
                previously |= mask
        return {
            'selected': int(s_t.sum().item()),
            'reused': int((s_t & previously).sum().item()),
            'newly_frozen': int((s_t & ~previously).sum().item()),
        }


# --------------------------------------------------------------------------- #
# Mean activations  (mu_i in EstimateSNV)
# --------------------------------------------------------------------------- #
class MeanActivationComputer:
    """mu_i = (1/|D_val|) sum_x a_i(x), the per-neuron mean response.

    Masking a neuron replaces its output with mu_i rather than with zero, which
    "blocks the flow of information through that filter while preserving the
    average statistics of the signal passed to subsequent layers".
    """

    def __init__(self, model: nn.Module, groups: List[NeuronGroup], device: torch.device):
        self.model = model
        self.groups = groups
        self.device = device
        self.mean_activations: Dict[str, torch.Tensor] = {}

    def compute(self, batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                task_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        sums: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        handles = []

        def make_hook(name: str):
            def hook(module, inputs, output):
                if output.dim() == 4:
                    per_sample = output.mean(dim=(2, 3))     # [B, C]
                else:
                    per_sample = output
                acc = per_sample.detach().sum(dim=0).float()
                sums[name] = acc if name not in sums else sums[name] + acc
                counts[name] = counts.get(name, 0) + output.shape[0]
            return hook

        for g in self.groups:
            handles.append(g.module.register_forward_hook(make_hook(g.name)))

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for x, _ in batches:
                _forward(self.model, x.to(self.device), task_id)
        if was_training:
            self.model.train()

        for h in handles:
            h.remove()

        self.mean_activations = {k: sums[k] / max(counts[k], 1) for k in sums}
        return self.mean_activations


def _forward(model: nn.Module, x: torch.Tensor, task_id: Optional[int]) -> torch.Tensor:
    """Call ``model`` with a task id when the model accepts one."""
    if task_id is None:
        return model(x)
    try:
        return model(x, task_id)
    except TypeError:
        return model(x)


# --------------------------------------------------------------------------- #
# Shapley Neuron Value estimation
# --------------------------------------------------------------------------- #
class ShapleyNeuronEstimator:
    """Truncated Monte-Carlo Shapley estimation with a top-k bandit stopping rule.

    ``V`` is evaluated on a fixed, cached set of validation batches.  Caching
    matters for correctness as well as speed: the bandit compares marginals
    across permutations, so ``V`` has to be a deterministic function of the
    subset rather than a fresh random sample each call.
    """

    def __init__(
        self,
        model: nn.Module,
        groups: List[NeuronGroup],
        mean_activations: Dict[str, torch.Tensor],
        device: torch.device,
        truncation_threshold: float = 0.1,
        confidence_level: float = 0.95,
        task_id: Optional[int] = None,
        payoff: str = 'accuracy',
        use_truncation: bool = True,
        use_mab: bool = True,
        estimator_mode: str = 'manuscript',
    ):
        """``payoff`` selects the score V returns.

        The paper defines V as "a black box that ... returns a score, such as
        accuracy, loss, or disparity".  Accuracy is a step function of the
        predictions, so on a validation set of size n every marginal is a
        multiple of 1/n and most are exactly zero -- the estimator spends its
        samples resolving ties.  Negative loss is continuous, so every marginal
        carries information, which cuts the variance of phi at fixed cost.
        """
        if payoff not in ('accuracy', 'loss'):
            raise ValueError(f'unknown payoff {payoff!r}')
        self.payoff = payoff
        if estimator_mode not in ('manuscript', 'reverse_tmc'):
            raise ValueError(estimator_mode)
        self.estimator_mode = estimator_mode
        self.model = model
        self.groups = groups
        self.mean_activations = mean_activations
        self.device = device
        self.tau = truncation_threshold
        self.confidence_level = confidence_level
        self.task_id = task_id
        self.num_neurons = sum(g.num_neurons for g in groups)
        self.z_alpha = float(_norm_ppf((1.0 + confidence_level) / 2.0))
        # Guard against retiring on a degenerate variance estimate: with a
        # discrete payoff a handful of samples are often all equal, giving
        # sigma = 0 and delta = 0, which retires an entirely unresolved neuron.
        # A sample variance is defined after two observations. Unsampled and
        # singly-sampled neurons remain active for exploration.
        self.min_samples_to_retire = 2
        self.eval_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.num_evaluations = 0
        # Ablation switches for the TMAB approximation layers.  With both off the
        # estimator is plain Monte-Carlo Shapley: every marginal is evaluated and
        # the loop always runs the full permutation budget.
        self.use_truncation = use_truncation
        self.use_mab = use_mab
        # Wall-clock split of the valuation phase, in seconds.
        self.phase_seconds = {'mc': 0.0, 'trunc_mab': 0.0}

    # -- payoff ------------------------------------------------------------ #
    def set_eval_batches(self, batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                         max_chunk: int = 2048) -> None:
        """Cache the batches V(S) is measured on, coalesced into few large ones.

        V is called once per neuron per permutation -- tens of thousands of times
        -- so per-call overhead dominates rather than arithmetic.  Merging the
        loader's small batches cuts the number of forward calls proportionally.
        """
        xs = [x for x, _ in batches]
        ys = [y for _, y in batches]
        if not xs:
            self.eval_batches = []
            return
        x_all = torch.cat(xs).to(self.device)
        y_all = torch.cat(ys).to(self.device)
        self.eval_batches = [(x_all[i:i + max_chunk], y_all[i:i + max_chunk])
                             for i in range(0, x_all.shape[0], max_chunk)]
        self._install_hooks()

    def _install_hooks(self) -> None:
        """Register masking hooks once and drive them from a mutable buffer.

        Re-registering hooks per evaluation costs a Python round-trip per layer
        per call.  Instead each layer keeps a persistent ``keep`` buffer that
        ``_set_active`` writes into, so switching subsets is a device-side copy.
        """
        self.remove_hooks()
        self._keep: Dict[str, torch.Tensor] = {}
        self._keep_flat = torch.ones(self.num_neurons, dtype=torch.bool, device=self.device)
        self._skip: Dict[str, bool] = {}

        for g in self.groups:
            mean = self.mean_activations.get(g.name)
            if mean is None:
                continue
            keep = self._keep_flat[g.slice()]
            self._keep[g.name] = keep
            self._skip[g.name] = True
            mean = mean.to(self.device)
            skip = self._skip
            name = g.name

            def hook(module, inputs, output, keep=keep, mean=mean, name=name, skip=skip):
                if skip[name]:                       # all neurons active: no-op
                    return output
                if output.dim() == 4:
                    k, m = keep.view(1, -1, 1, 1), mean.to(output.dtype).view(1, -1, 1, 1)
                else:
                    k, m = keep.view(1, -1), mean.to(output.dtype).view(1, -1)
                return torch.where(k, output, m.expand_as(output))

            self._handles.append(g.module.register_forward_hook(hook))

    def remove_hooks(self) -> None:
        for h in getattr(self, '_handles', []):
            h.remove()
        self._handles: List = []

    def _set_active(self, active: torch.Tensor) -> None:
        active_cpu = active.detach().cpu()
        self._keep_flat.copy_(active)
        for g in self.groups:
            if g.name not in self._keep:
                continue
            block = active_cpu[g.slice()]
            self._skip[g.name] = bool(block.all())

    def evaluate_subset(self, active: torch.Tensor) -> float:
        """V(S): accuracy with M minus S replaced by mean responses.  No retraining."""
        _t_eval0 = _time.time()
        if not getattr(self, '_handles', None):
            self._install_hooks()
        self._set_active(active)
        self.model.eval()
        score = total = 0.0
        with torch.no_grad():
            for x, y in self.eval_batches:
                logits = _forward(self.model, x, self.task_id)
                if self.payoff == 'accuracy':
                    score += logits.argmax(1).eq(y).sum().item()
                else:                      # negative loss: higher is still better
                    score -= nn.functional.cross_entropy(
                        logits, y, reduction='sum').item()
                total += y.numel()
        self.num_evaluations += 1
        self.phase_seconds['mc'] += _time.time() - _t_eval0
        return score / total if total else 0.0

    def estimate_shapley_values(
        self,
        k: int,
        max_permutations: int = 200,
        min_permutations: int = 5,
        verbose: bool = True,
    ) -> Dict[str, object]:
        """Run EstimateSNV.

        Returns a dict with ``phi``, ``counts``, ``permutations``, ``evaluations``
        and ``converged`` -- ``converged`` is False when ``max_permutations`` cut
        the loop short, so a truncated run is never silently reported as a
        confident one.
        """
        estimate_started = _time.perf_counter()
        if self.estimator_mode == 'reverse_tmc':
            return self._estimate_reverse(k, max_permutations, min_permutations, verbose)
        N = self.num_neurons
        # Scalar control on CUDA caused synchronization for every neuron and
        # dozens of tiny kernels per marginal. Only model evaluation needs GPU.
        phi = torch.zeros(N, dtype=torch.float64)
        counts = torch.zeros(N, dtype=torch.float64)
        m2 = torch.zeros(N, dtype=torch.float64)
        active = torch.ones(N, dtype=torch.bool)

        empty = torch.zeros(N, dtype=torch.bool, device=self.device)
        v_empty = self.evaluate_subset(empty)

        # tau is an absolute accuracy floor in the published method.  Negative
        # loss lives on a different, task-dependent scale, so there tau is read
        # as a fraction of the achievable range [V(0), V(M)].
        tau = self.tau
        if self.payoff == 'loss':
            v_full = self.evaluate_subset(torch.ones(N, dtype=torch.bool,
                                                     device=self.device))
            tau = v_empty + self.tau * (v_full - v_empty)

        permutation = 0
        converged = False
        bar = tqdm(total=max_permutations, desc='EstimateSNV', disable=not verbose)

        while permutation < max_permutations:
            # Preserve the original device RNG sequence, transfer once.
            perm = torch.randperm(N, device=self.device).cpu().tolist()
            subset = torch.zeros(N, dtype=torch.bool)
            v_prev = v_empty
            v_prev_valid = True

            for pos in range(N):
                i = perm[pos]

                if bool(active[i]):
                    if not v_prev_valid:
                        v_prev = self.evaluate_subset(subset)   # true V(S_i^pi)
                        v_prev_valid = True
                    if (not self.use_truncation) or v_prev > tau:  # (ii) truncation
                        subset[i] = True
                        v_new = self.evaluate_subset(subset)
                        delta = v_new - v_prev
                        counts[i] += 1
                        d1 = delta - phi[i]
                        phi[i] += d1 / counts[i]
                        m2[i] += d1 * (delta - phi[i])
                        v_prev = v_new
                        continue

                subset[i] = True
                v_prev_valid = False

            permutation += 1
            bar.update(1)
            if verbose:
                bar.set_postfix(evaluations=self.num_evaluations,
                                seconds=f'{_time.perf_counter()-estimate_started:.1f}')

            if permutation < min_permutations:
                continue

            if not self.use_mab:
                continue                    # plain MC: run the whole budget
            active = self._bandit_active_set(phi, counts, m2, k)
            n_active = int(active.sum().item())
            bar.set_postfix({'active': n_active, 'V(0)': f'{v_empty:.3f}'})
            if n_active == 0:                                    # (iii) A == {}
                converged = True
                break

        bar.close()
        self.remove_hooks()          # never leave a stale mask on the model
        # Everything outside actual V(S) forward evaluation is approximation
        # control: permutation construction, truncation decisions, online
        # moments, confidence intervals and active-set updates. Measuring the
        # complete residual makes MC + control add up to estimator wall time;
        # timing only `_bandit_active_set` hid nearly all truncation overhead.
        estimate_wall = _time.perf_counter() - estimate_started
        self.phase_seconds['trunc_mab'] = max(
            0.0, estimate_wall - self.phase_seconds['mc'])
        if not converged and verbose:
            print(f"  [EstimateSNV] stopped at the {max_permutations}-permutation cap with "
                  f"{int(active.sum().item())} neurons still unresolved; "
                  f"top-{k} separation is NOT confidence-certified for this task.")

        return {
            'phi': phi.float().to(self.device),
            'counts': counts.float().to(self.device),
            'permutations': permutation,
            'evaluations': self.num_evaluations,
            'converged': converged,
            # Pointwise normal intervals with adaptive sampling are a
            # heuristic; they are not a simultaneous top-k certificate.
            'confidence_certified': False,
            'v_empty': v_empty,
            'phase_seconds': dict(self.phase_seconds),
        }

    def _estimate_reverse(self, k, max_permutations, min_permutations, verbose):
        """Explicit approximation: remove in reverse order, zero the skipped tail.

        No monotonicity is assumed as a theorem: stopping near the empty
        baseline is a heuristic and can miss recoveries and negative marginals.
        Every permutation contributes once to every arm, including skipped zeros.
        The fixed budget avoids calling zero empirical variance a certificate.
        """
        started = _time.perf_counter()
        n = self.num_neurons
        empty = torch.zeros(n, dtype=torch.bool)
        v_empty = self.evaluate_subset(empty)
        v_full = self.evaluate_subset(~empty)
        tolerance = self.tau * abs(v_full - v_empty)
        phi = torch.zeros(n, dtype=torch.float64)
        m2 = torch.zeros_like(phi)
        measured = torch.zeros(n, dtype=torch.int64)
        stopped = 0
        try:
            for p in range(max_permutations):
                subset = torch.ones(n, dtype=torch.bool)
                prev = v_full
                delta = torch.zeros(n, dtype=torch.float64)
                for i in torch.randperm(n, device=self.device).cpu().tolist():
                    if self.use_truncation and abs(prev - v_empty) <= tolerance:
                        stopped += 1
                        break
                    subset[i] = False
                    current = self.evaluate_subset(subset)
                    delta[i] = prev - current
                    measured[i] += 1
                    prev = current
                d = delta - phi
                phi += d / (p + 1)
                m2 += d * (delta - phi)
                if verbose:
                    print(f'  reverse-TMC {p+1}/{max_permutations}: '
                          f'evaluations={self.num_evaluations}, seconds={_time.perf_counter()-started:.1f}', flush=True)
        finally:
            self.remove_hooks()
        self.phase_seconds['trunc_mab'] = max(0., _time.perf_counter()-started-self.phase_seconds['mc'])
        return {'phi':phi.float().to(self.device),
                'counts':torch.full((n,),float(max_permutations),device=self.device),
                'measured_counts':measured, 'permutations':max_permutations,
                'evaluations':self.num_evaluations, 'converged':False,
                'confidence_certified':False, 'estimator_mode':'reverse_tmc',
                'truncated_permutations':stopped, 'v_empty':v_empty,'v_full':v_full,
                'efficiency_residual':abs(float(phi.sum())-(v_full-v_empty)),
                'phase_seconds':dict(self.phase_seconds)}

    def _bandit_active_set(self, phi, counts, m2, k) -> torch.Tensor:
        """The supplied algorithm's exact active-set rule.

        A <- {i : |phi_i - phi^(k)| < delta_i}, where phi^(k) is the k-th
        largest estimate and delta_i = z_alpha sigma_i / sqrt(n_i). Neurons with
        fewer than two samples remain active because their variance is unknown.
        """
        delta = torch.full_like(phi, float('inf'))
        seen = counts >= self.min_samples_to_retire
        var = torch.zeros_like(phi)
        var[seen] = m2[seen] / (counts[seen] - 1)
        delta[seen] = self.z_alpha * torch.sqrt(var[seen] / counts[seen])

        n = phi.numel()
        k = max(1, min(k, n))
        phi_k = torch.topk(phi, k).values[k - 1]
        return (phi - phi_k).abs() < delta

    # -- selection --------------------------------------------------------- #
    def select_top_k_neurons(self, phi: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
        """S_t(i) = 1 iff phi_i is among the floor(c*N) largest values.

        Taken over all N neurons.  Per Fig. 1 a neuron may be in the top-r% for
        more than one task, so already-frozen neurons remain eligible.
        """
        k = int(math.floor(sparsity_ratio * self.num_neurons))
        k = max(1, min(k, self.num_neurons))
        mask = torch.zeros(self.num_neurons, dtype=torch.bool, device=phi.device)
        mask[torch.topk(phi, k).indices] = True
        return mask

    # -- exact reference (verification only) -------------------------------- #
    def exact_shapley_values(self) -> torch.Tensor:
        """Eq. (5) evaluated by enumeration -- exponential, for small N only.

        phi_i = sum_{S subset M\\{i}} |S|!(|M|-|S|-1)!/|M|! [V(S u {i}) - V(S)]

        Used by the test-suite to check the axioms against the closed form and to
        confirm the Monte-Carlo estimator converges to it.
        """
        n = self.num_neurons
        if n > 16:
            raise ValueError(f'exact enumeration needs 2^{n} evaluations; use the estimator')

        cache: Dict[int, float] = {}

        def value(bits: int) -> float:
            if bits not in cache:
                active = torch.tensor([(bits >> j) & 1 for j in range(n)],
                                      dtype=torch.bool, device=self.device)
                cache[bits] = self.evaluate_subset(active)
            return cache[bits]

        phi = torch.zeros(n, dtype=torch.float64)
        factorial = [math.factorial(i) for i in range(n + 1)]
        for i in range(n):
            for bits in range(1 << n):
                if (bits >> i) & 1:
                    continue
                size = bin(bits).count('1')
                weight = factorial[size] * factorial[n - size - 1] / factorial[n]
                phi[i] += weight * (value(bits | (1 << i)) - value(bits))
        self.remove_hooks()
        return phi.float()

    # -- axiom check (used by the test-suite) ------------------------------- #
    def efficiency_residual(self, phi: torch.Tensor) -> float:
        """|sum_i phi_i - (V(M) - V(0))|.

        The Efficiency axiom states sum_i phi_i = V(M).  With V(0) != 0 -- which
        is the case here, since masking everything to its mean leaves a
        well-defined baseline -- the payoff being divided is V(M) - V(0), and it
        is that quantity the estimator is unbiased for.
        """
        full = torch.ones(self.num_neurons, dtype=torch.bool, device=self.device)
        empty = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.device)
        residual = abs(float(phi.sum())
                       - (self.evaluate_subset(full) - self.evaluate_subset(empty)))
        self.remove_hooks()
        return residual


def _norm_ppf(p: float) -> float:
    try:
        from scipy import stats
        return float(stats.norm.ppf(p))
    except Exception:
        # Acklam's rational approximation; accurate to ~1e-9 over (0, 1).
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > ph:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q, r = p - 0.5, (p - 0.5) ** 2
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def install_subnetwork_mask(groups, active, means, device):
    """Mask every neuron outside ``active`` to its stored mean response.

    Used for per-task inference: Section 2 defines the task's subnetwork as
    ``M* = M (.) S_t*`` and Eq. (1) scores it as ``f(x; M (.) S_t)``, i.e. the
    masked network -- while the Train/Evaluate pseudocode abbreviates this as
    ``f_theta(x, k)``. The active set is the task mask S_t; because S_t is added
    to B_t immediately, all parameters used by that subnetwork remain frozen.
    """
    handles = []
    for g in groups:
        mean = means.get(g.name)
        if mean is None:
            continue
        keep = active[g.slice()].to(device)
        if bool(keep.all()):
            continue
        mean = mean.to(device)

        def hook(module, inputs, output, keep=keep, mean=mean):
            if output.dim() == 4:
                k, m = keep.view(1, -1, 1, 1), mean.to(output.dtype).view(1, -1, 1, 1)
            else:
                k, m = keep.view(1, -1), mean.to(output.dtype).view(1, -1)
            return torch.where(k, output, m.expand_as(output))

        handles.append(g.module.register_forward_hook(hook))
    return handles


# --------------------------------------------------------------------------- #
# The Train procedure
# --------------------------------------------------------------------------- #
class SNVContinualLearner:
    """Algorithm 1 (Train / EstimateSNV / Evaluate).

    Head handling.  The algorithm stores a task head h_t and evaluates with
    f_theta(x, k).  Heads of finished tasks are frozen.  Under TIL the head of
    the queried task is used; under CIL, where no task identity is available at
    test time, the logits of every head seen so far are concatenated and the
    argmax is taken over the union -- the concatenation order reproduces the
    global class index.  The paper does not state the CIL head rule explicitly;
    this is the standard reading and is recorded here as an assumption.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        sparsity_ratio: float = 0.1,
        truncation_threshold: float = 0.1,
        confidence_level: float = 0.95,
        lr: float = 0.001,
        scenario: str = 'class_il',
        max_permutations: int = 200,
        shapley_eval_batches: int = 0,
        selection: str = 'ablation',
        payoff: str = 'accuracy',
        track_inrun: bool = False,
        compute_ablation: bool = True,
        inrun_val_every: int = 1,
        dual_pool_factor: float = 2.0,
        use_truncation: bool = True,
        use_mab: bool = True,
        masked_inference: bool = True,
        masked_training: bool = False,
        consolidation_epochs: int = 0,
        layer_floor: float = 0.0,
        estimator_mode: str = 'manuscript',
        consolidation_within_budget: bool = False,
    ):
        """
        selection
            'ablation'  S_t = top-k of phi_abl  (the published method)
            'inrun'     S_t = top-k of phi_run  (trajectory valuation only)
            'dual'      candidate pool = top (dual_pool_factor * k) by phi_run --
                        the neurons this task actually moved -- and S_t = the k
                        most load-bearing of those by phi_abl.  Budget is charged
                        by what the task claimed; protection is decided by what
                        the function needs.
        track_inrun     accumulate phi_run during training (required by 'inrun'
                        and 'dual'; enable it alongside 'ablation' to compare).
        compute_ablation
            set False with selection='inrun' to skip the expensive estimator.
        """
        if selection not in ('ablation', 'inrun', 'dual'):
            raise ValueError(f'unknown selection {selection!r}')
        if selection in ('inrun', 'dual'):
            track_inrun = True
        if selection == 'dual' and not compute_ablation:
            raise ValueError("selection='dual' needs the ablation valuation")

        self.model = model.to(device)
        self.device = device
        self.sparsity_ratio = sparsity_ratio
        self.truncation_threshold = truncation_threshold
        self.confidence_level = confidence_level
        self.lr = lr
        self.scenario = scenario
        self.max_permutations = max_permutations
        self.shapley_eval_batches = shapley_eval_batches
        self.selection = selection
        self.payoff = payoff
        self.estimator_mode = estimator_mode
        self.track_inrun = track_inrun
        self.compute_ablation = compute_ablation or selection != 'inrun'
        self.inrun_val_every = inrun_val_every
        self.dual_pool_factor = dual_pool_factor
        self.use_truncation = use_truncation
        self.use_mab = use_mab
        # Wall-clock split of SNV's valuation phase, accumulated over tasks.
        self.phase_seconds = {'mc': 0.0, 'trunc_mab': 0.0, 'mask_update': 0.0}

        self.masked_inference = masked_inference
        # Eq. (1) scores task t as f(x; M (.) S_t), but Train fits theta on the
        # dense network -- so the model that is scored is not the model that was
        # optimised.  Measured on CIFAR-100 task 0 (diag_masked_subnetwork.py):
        # the same phi-ranked mask gives 8.67% installed after dense training and
        # 73.50% when the kept neurons are trained through it, against 72.92%
        # dense.  masked_training closes that gap by continuing task t's own
        # training through its own subnetwork before the neurons are frozen.
        self.masked_training = masked_training
        self.consolidation_epochs = consolidation_epochs
        self.consolidation_within_budget = consolidation_within_budget
        if consolidation_within_budget and (not masked_training or consolidation_epochs <= 0):
            raise ValueError('within-budget consolidation requires masked_training and positive consolidation_epochs')
        # A global top-k over phi has no per-layer floor and can empty a layer
        # outright; every neuron in it is then clamped to a constant and the
        # network is severed.  A magnitude ranking empties 12 of 20 layers at
        # c=0.1 -- including the stem -- and no amount of downstream fine-tuning
        # recovers it (10.50% vs 69.08% for a layer-spread mask of equal size).
        self.layer_floor = layer_floor
        self.task_eval_state: Dict[int, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = {}
        self.task_norm_state: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}

        self.mask_manager = NeuronMaskManager(self.model, device)
        self.shapley_values: Dict[int, torch.Tensor] = {}
        self.inrun_values: Dict[int, torch.Tensor] = {}
        self.history: List[Dict] = []

    # -- training ---------------------------------------------------------- #
    def train_task(
        self,
        task_id: int,
        train_loader,
        val_loader,
        num_epochs: int = 200,
        patience: int = 20,
        verbose: bool = True,
    ) -> Dict:
        dense_epochs = num_epochs
        if self.consolidation_within_budget:
            dense_epochs -= self.consolidation_epochs
            if dense_epochs < 1:
                raise ValueError('consolidation_epochs must be smaller than the total epoch budget')
        dense_epochs_run = 0
        if hasattr(self.model, 'ensure_head'):
            self.model.ensure_head(task_id)
            self.model.freeze_heads_before(task_id)
        self.model.to(self.device)

        self.mask_manager.snapshot_frozen_state()
        criterion = nn.CrossEntropyLoss()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        from training_policy import optimizer_for, EpochPolicy
        optimizer = optimizer_for(self, trainable)
        schedule = EpochPolicy(self, optimizer, dense_epochs, patience)
        epoch_log = []

        eval_batches = self._cache_val_batches(val_loader)
        tracker = None
        if self.track_inrun:
            from inrun import InRunNeuronShapley
            tracker = InRunNeuronShapley(
                self.model, self.mask_manager.groups, self.device, eval_batches,
                task_id=None if self.scenario == 'class_il' else task_id,
                scenario=self.scenario, val_every=self.inrun_val_every,
                criterion=criterion)

        best_val_loss, best_state, best_phi_run, waited = float('inf'), None, None, 0
        bar = tqdm(range(dense_epochs), desc=f'Task {task_id}', disable=not verbose)

        for epoch in bar:
            dense_epochs_run += 1
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                if tracker is not None:
                    tracker.begin_step()
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self._train_logits(x, task_id), y)
                loss.backward()
                self.mask_manager.apply_gradient_mask()     # (dL/dtheta . M_{t-1})
                optimizer.step()
                self.mask_manager.restore_frozen_state()    # keeps norm buffers fixed too
                if tracker is not None:
                    tracker.end_step()

            val_loss, val_acc = self._validate(val_loader, task_id, criterion)
            epoch_log.append(dict(epoch=epoch+1, validation_loss=val_loss, validation_accuracy=val_acc))
            if val_loss < best_val_loss:
                best_val_loss, waited = val_loss, 0
                best_state = copy.deepcopy(self.model.state_dict())
                # phi_run must describe the model that is actually returned, so
                # the accumulation is cut at the checkpoint early stopping keeps.
                if tracker is not None:
                    best_phi_run = tracker.phi.clone()
            else:
                waited += 1
                if schedule.should_stop(epoch+1, waited):
                    if verbose:
                        bar.write(f'  early stop at epoch {epoch + 1}')
                    break
            schedule.step()
            bar.set_postfix({'val_loss': f'{val_loss:.4f}', 'val_acc': f'{val_acc:.4f}'})
        bar.close()

        if best_state is not None:
            self.model.load_state_dict(best_state)
        if tracker is not None and best_phi_run is not None:
            tracker.phi = best_phi_run
        self.mask_manager.restore_frozen_state()

        drift = self.mask_manager.max_frozen_drift()
        if drift > 1e-6:
            raise RuntimeError(
                f'Frozen state drifted by {drift:.3e} during task {task_id}; '
                'the freezing mask is not doing its job.')

        _t_val = _time.time()
        result = self._value_and_freeze(task_id, eval_batches, tracker, verbose,
                                        train_loader=train_loader, val_loader=val_loader,
                                        num_epochs=num_epochs, patience=patience)
        self.valuation_seconds = getattr(self, 'valuation_seconds', 0.0) + (_time.time() - _t_val)
        result['frozen_drift'] = drift
        result['dense_epochs_run'] = dense_epochs_run
        result['dense_epoch_log'] = epoch_log
        result['dense_epoch_cap'] = dense_epochs
        result['dense_cap_reached_while_improving'] = dense_epochs_run==dense_epochs and waited<patience
        result['total_training_epochs_run'] = dense_epochs_run + result.get('consolidation_epochs_run', 0)
        result['consolidation_within_budget'] = self.consolidation_within_budget
        if self.consolidation_within_budget and result['total_training_epochs_run'] > num_epochs:
            raise RuntimeError('SNV exceeded its training epoch budget')
        self.history.append(result)
        return result

    def _cache_val_batches(self, val_loader) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batches = []
        for i, (x, y) in enumerate(val_loader):
            if self.shapley_eval_batches > 0 and i >= self.shapley_eval_batches:
                break
            batches.append((x, y))
        return batches

    def _train_logits(self, x, task_id):
        """CIL trains over every class seen so far; TIL over the current head."""
        if self.scenario == 'class_il' and hasattr(self.model, 'forward'):
            return _forward(self.model, x, None)
        return _forward(self.model, x, task_id)

    def _set_frozen_norms_eval(self) -> None:
        """Norm layers with any frozen channel stay in eval mode during training.

        BatchNorm updates its running statistics in the forward pass, outside the
        optimizer, so ``M_{t-1}`` cannot reach them.  Leaving them in train mode
        rewrites the statistics a frozen filter depends on.
        """
        for g in self.mask_manager.groups:
            if g.norm_module is None:
                continue
            if bool(self.mask_manager.cumulative_mask[g.slice()].any()):
                g.norm_module.eval()

    def _validate(self, loader, task_id, criterion) -> Tuple[float, float]:
        self.model.eval()
        loss_sum = correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self._train_logits(x, task_id)
                loss_sum += criterion(logits, y).item() * y.numel()
                correct += logits.argmax(1).eq(y).sum().item()
                total += y.numel()
        return loss_sum / max(total, 1), correct / max(total, 1)

    # -- valuation --------------------------------------------------------- #
    def _consolidate(self, task_id: int, task_mask: torch.Tensor, means,
                     train_loader, val_loader, num_epochs: int, patience: int,
                     verbose: bool) -> Dict:
        """Continue task t's training through M (.) S_t, then stop.

        Exactness is unaffected: neurons outside S_t are clamped to the stored
        mean responses, so they contribute a constant no matter what happens to
        their weights afterwards, and every neuron inside S_t is frozen from the
        end of this call onwards.  Randomising every out-of-mask weight after
        consolidation leaves the accuracy bit-identical -- that is the check
        diag_masked_subnetwork.py runs in its ``after-scramble`` column.
        """
        epochs = self.consolidation_epochs or num_epochs
        if epochs <= 0:
            return {'consolidation_epochs_run': 0}
        handles = install_subnetwork_mask(self.mask_manager.groups, task_mask,
                                          means, self.device)
        criterion = nn.CrossEntropyLoss()
        try:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable:
                return {'consolidation_epochs_run': 0}
            from training_policy import optimizer_for, EpochPolicy
            optimizer = optimizer_for(self, trainable)
            schedule = EpochPolicy(self, optimizer, epochs, patience)
            epoch_log = []
            best_loss, before_acc = self._validate(val_loader, task_id, criterion)
            best_state, waited, epochs_run = copy.deepcopy(self.model.state_dict()), 0, 0
            bar = tqdm(range(epochs), desc=f'Task {task_id} consolidate',
                       disable=not verbose)
            for _ in bar:
                epochs_run += 1
                self.model.train()
                self._set_frozen_norms_eval()
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(self._train_logits(x, task_id), y)
                    loss.backward()
                    self.mask_manager.apply_gradient_mask()
                    optimizer.step()
                    self.mask_manager.restore_frozen_state()
                val_loss, val_acc = self._validate(val_loader, task_id, criterion)
                epoch_log.append(dict(epoch=epochs_run, validation_loss=val_loss, validation_accuracy=val_acc))
                if val_loss < best_loss:
                    best_loss, waited = val_loss, 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    waited += 1
                    if schedule.should_stop(epochs_run, waited):
                        break
                schedule.step()
                bar.set_postfix({'val_loss': f'{val_loss:.4f}', 'val_acc': f'{val_acc:.4f}'})
            bar.close()
            if best_state is not None:
                self.model.load_state_dict(best_state)
            self.mask_manager.restore_frozen_state()
            after_loss, after_acc = self._validate(val_loader, task_id, criterion)
            drift = self.mask_manager.max_frozen_drift()
            if drift > 1e-6:
                raise RuntimeError(f'Frozen state drifted during consolidation: {drift:.3e}')
            return {'consolidation_epochs_run': epochs_run,
                    'consolidation_epoch_log': epoch_log, 'consolidation_epoch_cap': epochs,
                    'consolidation_cap_reached_while_improving': epochs_run==epochs and waited<patience,
                    'masked_val_acc_before': before_acc,
                    'masked_val_acc_after': after_acc,
                    'masked_val_loss_after': after_loss,
                    'consolidation_frozen_drift': drift}
        finally:
            for h in handles:
                h.remove()

    def _value_and_freeze(self, task_id: int, batches, tracker, verbose: bool,
                          train_loader=None, val_loader=None,
                          num_epochs: int = 0, patience: int = 0) -> Dict:
        head = None if self.scenario == 'class_il' else task_id
        N = self.mask_manager.num_neurons
        k = max(1, int(math.floor(self.sparsity_ratio * N)))
        result: Dict = {'task_id': task_id, 'k': k, 'selection': self.selection}

        phi_run = None
        if tracker is not None:
            phi_run = tracker.values
            self.inrun_values[task_id] = phi_run.cpu()
            result['inrun'] = tracker.summary()

        means = None
        if self.compute_ablation or self.masked_inference or self.masked_training:
            mean_started = _time.perf_counter()
            means = MeanActivationComputer(self.model, self.mask_manager.groups, self.device)
            means.compute(batches, head)
            # Mean-response estimation is itself a validation forward pass and
            # belongs in the table's MC/forward-pass phase.
            self.phase_seconds['mc'] += _time.perf_counter() - mean_started

        phi_abl = None
        if self.compute_ablation:
            estimator = ShapleyNeuronEstimator(
                self.model, self.mask_manager.groups, means.mean_activations, self.device,
                self.truncation_threshold, self.confidence_level, task_id=head,
                payoff=self.payoff, use_truncation=self.use_truncation,
                use_mab=self.use_mab, estimator_mode=self.estimator_mode)
            estimator.set_eval_batches(batches)
            out = estimator.estimate_shapley_values(
                k=k, max_permutations=self.max_permutations, verbose=verbose)
            phi_abl = out['phi']
            estimator.remove_hooks()
            self.shapley_values[task_id] = phi_abl.cpu()
            result.update(permutations=out['permutations'], evaluations=out['evaluations'],
                          converged=out['converged'], estimator_mode=self.estimator_mode,
                          confidence_certified=False,
                          efficiency_residual=out.get('efficiency_residual'))
            for _p, _v in out.get('phase_seconds', {}).items():
                self.phase_seconds[_p] = self.phase_seconds.get(_p, 0.0) + _v

        _t_mask = _time.time()
        task_mask = self._select(phi_abl, phi_run, k)
        result['layer_coverage'] = self.layer_coverage(task_mask)
        result['empty_layers'] = sum(1 for v in result['layer_coverage'].values() if v == 0)

        # Consolidation: train task t through its own Eq.(1) subnetwork before
        # its neurons are frozen, so the model that is scored is the model that
        # was optimised.  B_{t-1} is still gradient-masked, and the mean-response
        # hook routes gradient only to neurons inside S_t, so the only weights
        # that move are the ones this task is about to claim.
        if self.masked_training and means is not None and train_loader is not None:
            result.update(self._consolidate(task_id, task_mask, means.mean_activations,
                              train_loader, val_loader, num_epochs, patience, verbose))

        self.mask_manager.update_cumulative_mask(task_id, task_mask)
        # Extend the exact frozen-state snapshot to neurons first claimed by
        # this task. Without this, post-task drift checks index B_t against a
        # snapshot containing only B_{t-1}.
        self.mask_manager.snapshot_frozen_state()
        self.phase_seconds['mask_update'] += _time.time() - _t_mask

        if self.masked_inference and means is not None:
            # Eq. (1) defines the task subnetwork as M (.) S_t, not the
            # cumulative freezing union B_t. S_t's parameters are included in
            # B_t immediately below and therefore remain invariant later.
            self.task_eval_state[task_id] = (
                task_mask.clone(),
                {k: v.clone() for k, v in means.mean_activations.items()})
            # Mean replacement happens at filter outputs, before a paired norm.
            # Preserve that norm's task-specific transform too; otherwise
            # plastic channels' later BN updates alter the old masked function.
            self.task_norm_state[task_id] = copy.deepcopy({
                name: module.state_dict()
                for name, module in self.model.named_modules()
                if isinstance(module, _NORM_TYPES)
            })

        stats = self.mask_manager.reuse_stats(task_id)
        result.update(stats)
        result['task_mask'] = task_mask.cpu()
        result['capacity_used'] = self.mask_manager.get_capacity_used()
        if phi_abl is not None:
            result['shapley_values'] = phi_abl.cpu()
        if phi_run is not None:
            result['inrun_values'] = phi_run.cpu()

        if verbose:
            print(f'  Task {task_id}: k={k}  selected={stats["selected"]}  '
                  f'reused={stats["reused"]}  new={stats["newly_frozen"]}  '
                  f'capacity={self.mask_manager.get_capacity_used():.2f}%  '
                  f'via {self.selection}')
        return result

    def _select(self, phi_abl, phi_run, k: int) -> torch.Tensor:
        """S_t under the configured criterion."""
        N = self.mask_manager.num_neurons

        if self.selection == 'ablation':
            source = phi_abl
        elif self.selection == 'inrun':
            source = phi_run
        else:                                     # dual
            # Pool: what this task actually moved.  A frozen neuron has
            # phi_run = 0 exactly, so it is excluded without an availability mask.
            pool_size = max(k, min(N, int(self.dual_pool_factor * k)))
            pool = torch.zeros(N, dtype=torch.bool, device=phi_run.device)
            pool[torch.topk(phi_run, pool_size).indices] = True
            pool &= phi_run > 0

            # Eq. (1) constrains ||S_t||_0 <= floor(c*N) -- an upper bound, not an
            # equality.  When a task claims fewer than k neurons the budget is
            # deliberately left unspent rather than padded with neurons this task
            # never moved; that unspent capacity is what stays plastic for later
            # tasks, and it is the mechanism the dual criterion trades on.
            available = int(pool.sum())
            if available == 0:
                return torch.zeros(N, dtype=torch.bool, device=phi_run.device)
            k = min(k, available)
            source = phi_abl.clone()
            source[~pool.to(source.device)] = float('-inf')

        if source is None:
            raise RuntimeError(f"selection={self.selection!r} has no valuation to rank by")

        return self._topk_with_layer_floor(source, k)

    def _topk_with_layer_floor(self, source: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k over all N neurons, but never leaving a layer empty.

        Eq. (1) reads phi as a single global ranking, and Fig. 1 needs it to be
        global so a neuron can be claimed by several tasks.  Taken literally,
        though, nothing stops the top-k from selecting no neuron at all in some
        layer: at Eq. (1) inference every neuron there is replaced by its mean
        response, the layer emits a constant, and the subnetwork is cut in two.
        When that layer is the stem, no image information enters the network and
        the subnetwork sits at chance no matter how it is trained.

        The floor reserves ``layer_floor`` of each layer for that layer's own
        best-ranked neurons and spends the remaining budget globally, so the
        ranking still decides almost everything while connectivity is guaranteed.
        """
        N = self.mask_manager.num_neurons
        eligible = torch.isfinite(source)
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0 or k <= 0:
            return torch.zeros(N, dtype=torch.bool, device=source.device)
        k = min(k, N, eligible_count)
        mask = torch.zeros(N, dtype=torch.bool, device=source.device)
        groups = self.mask_manager.groups

        if self.layer_floor > 0 and groups:
            quota = [min(int(eligible[g.slice()].sum().item()),
                         max(1, int(math.ceil(self.layer_floor * g.num_neurons))))
                     for g in groups]
            if sum(quota) > k:
                # Budget too small to honour the requested floor.  Connectivity
                # is not negotiable, so keep one neuron per layer and warn: the
                # ranking then decides the rest of a much tighter budget.
                quota = [int(eligible[g.slice()].any()) for g in groups]
            if sum(quota) <= k:
                for g, q in zip(groups, quota):
                    if q == 0:
                        continue
                    block = source[g.slice()]
                    q = min(q, int(torch.isfinite(block).sum().item()))
                    idx = torch.topk(block, q).indices
                    mask[g.start + idx] = True

        remaining = k - int(mask.sum().item())
        if remaining > 0:
            rest = source.clone().float()
            rest[mask] = float('-inf')
            mask[torch.topk(rest, remaining).indices] = True
        return mask

    def layer_coverage(self, mask: torch.Tensor) -> Dict[str, int]:
        """Neurons kept per layer -- zero anywhere means a severed subnetwork."""
        return {g.name: int(mask[g.slice()].sum().item()) for g in self.mask_manager.groups}

    # -- evaluation -------------------------------------------------------- #
    def _install_task_subnetwork(self, task_id):
        """Install task_id's Eq.(1) subnetwork; returns (handles, saved_norm_state).

        Shared by evaluate() and predict() so a timing harness measures the same
        forward pass the accuracy numbers come from.
        """
        handles, norm_now, norm_modules = [], None, None
        if self.masked_inference and task_id is not None and task_id in self.task_eval_state:
            active, means = self.task_eval_state[task_id]
            handles = install_subnetwork_mask(self.mask_manager.groups, active, means,
                                              self.device)
            if task_id in self.task_norm_state:
                norm_modules = {name: module for name, module in self.model.named_modules()
                                if isinstance(module, _NORM_TYPES)}
                norm_now = copy.deepcopy({name: module.state_dict()
                                          for name, module in norm_modules.items()})
                for name, state in self.task_norm_state[task_id].items():
                    if name in norm_modules:
                        norm_modules[name].load_state_dict(state)
        return handles, norm_now, norm_modules

    @staticmethod
    def _remove_task_subnetwork(handles, norm_now, norm_modules):
        for h in handles:
            h.remove()
        if norm_now is not None:
            for name, state in norm_now.items():
                norm_modules[name].load_state_dict(state)

    def predict(self, x, task_id: Optional[int] = None):
        """CIL scores every candidate head through its own mask, without task ID."""
        self.model.eval()
        x = x.to(self.device)
        candidates = (self.model.active_tasks() if self.scenario == 'class_il'
                      else [task_id])
        outputs = []
        for candidate in candidates:
            handles, norm_now, norm_modules = self._install_task_subnetwork(candidate)
            try:
                with torch.no_grad():
                    outputs.append(_forward(self.model, x, candidate))
            finally:
                self._remove_task_subnetwork(handles, norm_now, norm_modules)
        return torch.cat(outputs, dim=1) if self.scenario == 'class_il' else outputs[0]

    def evaluate(self, test_loader, task_id: Optional[int] = None) -> float:
        correct = total = 0
        for x, y in test_loader:
            logits = self.predict(x, task_id)
            correct += logits.argmax(1).eq(y.to(self.device)).sum().item()
            total += y.numel()
        return correct / total if total else 0.0

    def evaluate_all_tasks(self, test_loaders, current_task: int) -> np.ndarray:
        """Measure the full row, including future tasks required by FWT and PS."""
        row = np.full(len(test_loaders), np.nan)
        for k in range(current_task + 1):
            row[k] = self.evaluate(test_loaders[k], k)
        if hasattr(self.model, 'full_output_space'):
            with self.model.full_output_space(len(test_loaders)):
                for k in range(current_task + 1, len(test_loaders)):
                    row[k] = self.evaluate(test_loaders[k], k)
        return row
