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
    ):
        self.model = model
        self.groups = groups
        self.mean_activations = mean_activations
        self.device = device
        self.tau = truncation_threshold
        self.confidence_level = confidence_level
        self.task_id = task_id
        self.num_neurons = sum(g.num_neurons for g in groups)
        self.z_alpha = float(_norm_ppf((1.0 + confidence_level) / 2.0))
        self.eval_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.num_evaluations = 0

    # -- payoff ------------------------------------------------------------ #
    def set_eval_batches(self, batches: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.eval_batches = [(x.to(self.device), y.to(self.device)) for x, y in batches]

    def _mask_hooks(self, active: torch.Tensor) -> List:
        """Replace the output of every neuron outside S with its mean response."""
        handles = []
        for g in self.groups:
            mean = self.mean_activations.get(g.name)
            if mean is None:
                continue
            keep = active[g.slice()].to(self.device)
            if bool(keep.all()):
                continue
            mean = mean.to(self.device)

            def hook(module, inputs, output, keep=keep, mean=mean):
                if output.dim() == 4:
                    k = keep.view(1, -1, 1, 1)
                    m = mean.to(output.dtype).view(1, -1, 1, 1)
                else:
                    k = keep.view(1, -1)
                    m = mean.to(output.dtype).view(1, -1)
                return torch.where(k, output, m.expand_as(output))

            handles.append(g.module.register_forward_hook(hook))
        return handles

    def evaluate_subset(self, active: torch.Tensor) -> float:
        """V(S): accuracy with M \\ S replaced by mean responses.  No retraining."""
        handles = self._mask_hooks(active)
        self.model.eval()
        correct = total = 0
        try:
            with torch.no_grad():
                for x, y in self.eval_batches:
                    logits = _forward(self.model, x, self.task_id)
                    correct += logits.argmax(1).eq(y).sum().item()
                    total += y.numel()
        finally:
            for h in handles:
                h.remove()
        self.num_evaluations += 1
        return correct / total if total else 0.0

    # -- estimation -------------------------------------------------------- #
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
        N = self.num_neurons
        phi = torch.zeros(N, dtype=torch.float64, device=self.device)
        counts = torch.zeros(N, dtype=torch.float64, device=self.device)
        m2 = torch.zeros(N, dtype=torch.float64, device=self.device)
        active = torch.ones(N, dtype=torch.bool, device=self.device)

        empty = torch.zeros(N, dtype=torch.bool, device=self.device)
        v_empty = self.evaluate_subset(empty)

        permutation = 0
        converged = False
        bar = tqdm(total=max_permutations, desc='EstimateSNV', disable=not verbose)

        while permutation < max_permutations:
            perm = torch.randperm(N, device=self.device)
            subset = torch.zeros(N, dtype=torch.bool, device=self.device)
            v_prev = v_empty
            v_prev_valid = True

            for pos in range(N):
                i = int(perm[pos].item())

                if bool(active[i]):
                    if not v_prev_valid:
                        v_prev = self.evaluate_subset(subset)   # true V(S_i^pi)
                        v_prev_valid = True
                    if v_prev > self.tau:                        # (ii) truncation
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

            if permutation < min_permutations:
                continue

            active = self._bandit_active_set(phi, counts, m2, k)
            n_active = int(active.sum().item())
            bar.set_postfix({'active': n_active, 'V(0)': f'{v_empty:.3f}'})
            if n_active == 0:                                    # (iii) A == {}
                converged = True
                break

        bar.close()
        if not converged and verbose:
            print(f"  [EstimateSNV] stopped at the {max_permutations}-permutation cap with "
                  f"{int(active.sum().item())} neurons still unresolved; "
                  f"top-{k} separation is NOT confidence-certified for this task.")

        return {
            'phi': phi.float(),
            'counts': counts.float(),
            'permutations': permutation,
            'evaluations': self.num_evaluations,
            'converged': converged,
            'v_empty': v_empty,
        }

    def _bandit_active_set(self, phi, counts, m2, k) -> torch.Tensor:
        """A <- {i : |phi_i - b| < delta_i},  delta_i = z_alpha sigma_i / sqrt(n_i).

        ``b`` is the *top-k boundary*: the midpoint between the k-th and the
        (k+1)-th largest estimates.  The paper writes the comparison against
        phi^(k) itself, but taken literally that rule never terminates -- the
        neuron holding rank k is at distance exactly 0 from phi^(k), so it
        satisfies |phi_i - phi^(k)| < delta_i for any delta_i > 0 and the loop
        ``while A != {}`` cannot exit.  Placing the boundary between ranks k and
        k+1 is the reading that matches the stated intent, "the Neurons whose
        lower and upper bounds straddle the current top-k position", and it does
        terminate once the two ranks separate by more than their intervals.

        Neurons with fewer than two samples have an undefined sigma and are kept
        active -- the exploration half of the bandit.
        """
        delta = torch.full_like(phi, float('inf'))
        seen = counts > 1
        var = torch.zeros_like(phi)
        var[seen] = m2[seen] / (counts[seen] - 1)
        delta[seen] = self.z_alpha * torch.sqrt(var[seen] / counts[seen])

        n = phi.numel()
        k = max(1, min(k, n))
        top = torch.topk(phi, min(k + 1, n)).values
        boundary = 0.5 * (top[k - 1] + top[k]) if k < n else top[k - 1]
        return (phi - boundary).abs() < delta

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
        return abs(float(phi.sum()) - (self.evaluate_subset(full) - self.evaluate_subset(empty)))


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
        shapley_eval_batches: int = 8,
    ):
        self.model = model.to(device)
        self.device = device
        self.sparsity_ratio = sparsity_ratio
        self.truncation_threshold = truncation_threshold
        self.confidence_level = confidence_level
        self.lr = lr
        self.scenario = scenario
        self.max_permutations = max_permutations
        self.shapley_eval_batches = shapley_eval_batches

        self.mask_manager = NeuronMaskManager(self.model, device)
        self.shapley_values: Dict[int, torch.Tensor] = {}
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
        if hasattr(self.model, 'ensure_head'):
            self.model.ensure_head(task_id)
            self.model.freeze_heads_before(task_id)
        self.model.to(self.device)

        self.mask_manager.snapshot_frozen_state()
        criterion = nn.CrossEntropyLoss()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.lr)

        best_val_loss, best_state, waited = float('inf'), None, 0
        bar = tqdm(range(num_epochs), desc=f'Task {task_id}', disable=not verbose)

        for epoch in bar:
            self.model.train()
            self._set_frozen_norms_eval()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self._train_logits(x, task_id), y)
                loss.backward()
                self.mask_manager.apply_gradient_mask()     # (dL/dtheta . M_{t-1})
                optimizer.step()
                self.mask_manager.restore_frozen_state()    # keeps norm buffers fixed too

            val_loss, val_acc = self._validate(val_loader, task_id, criterion)
            if val_loss < best_val_loss:
                best_val_loss, waited = val_loss, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                waited += 1
                if waited >= patience:
                    if verbose:
                        bar.write(f'  early stop at epoch {epoch + 1}')
                    break
            bar.set_postfix({'val_loss': f'{val_loss:.4f}', 'val_acc': f'{val_acc:.4f}'})
        bar.close()

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.mask_manager.restore_frozen_state()

        drift = self.mask_manager.max_frozen_drift()
        if drift > 1e-6:
            raise RuntimeError(
                f'Frozen state drifted by {drift:.3e} during task {task_id}; '
                'the freezing mask is not doing its job.')

        result = self._value_and_freeze(task_id, val_loader, verbose)
        result['frozen_drift'] = drift
        self.history.append(result)
        return result

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
    def _value_and_freeze(self, task_id: int, val_loader, verbose: bool) -> Dict:
        batches = []
        for i, (x, y) in enumerate(val_loader):
            if i >= self.shapley_eval_batches:
                break
            batches.append((x, y))

        head = None if self.scenario == 'class_il' else task_id

        means = MeanActivationComputer(self.model, self.mask_manager.groups, self.device)
        means.compute(batches, head)

        estimator = ShapleyNeuronEstimator(
            self.model, self.mask_manager.groups, means.mean_activations, self.device,
            self.truncation_threshold, self.confidence_level, task_id=head)
        estimator.set_eval_batches(batches)

        k = int(math.floor(self.sparsity_ratio * self.mask_manager.num_neurons))
        out = estimator.estimate_shapley_values(
            k=max(1, k), max_permutations=self.max_permutations, verbose=verbose)

        phi = out['phi']
        self.shapley_values[task_id] = phi.cpu()
        task_mask = estimator.select_top_k_neurons(phi, self.sparsity_ratio)
        self.mask_manager.update_cumulative_mask(task_id, task_mask)

        stats = self.mask_manager.reuse_stats(task_id)
        if verbose:
            print(f'  Task {task_id}: k={k}  selected={stats["selected"]}  '
                  f'reused={stats["reused"]}  new={stats["newly_frozen"]}  '
                  f'capacity={self.mask_manager.get_capacity_used():.2f}%  '
                  f'permutations={out["permutations"]}  converged={out["converged"]}')

        return {
            'task_id': task_id,
            'shapley_values': phi.cpu(),
            'task_mask': task_mask.cpu(),
            'capacity_used': self.mask_manager.get_capacity_used(),
            'permutations': out['permutations'],
            'evaluations': out['evaluations'],
            'converged': out['converged'],
            **stats,
        }

    # -- evaluation -------------------------------------------------------- #
    def evaluate(self, test_loader, task_id: Optional[int] = None) -> float:
        head = task_id if self.scenario == 'task_il' else None
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = _forward(self.model, x, head)
                correct += logits.argmax(1).eq(y).sum().item()
                total += y.numel()
        return correct / total if total else 0.0

    def evaluate_all_tasks(self, test_loaders, current_task: int) -> np.ndarray:
        """The Evaluate procedure: r_k for k = 1..t."""
        return np.array([self.evaluate(test_loaders[k], k) for k in range(current_task + 1)])
