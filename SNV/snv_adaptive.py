"""SNV-A: SNV with task-local training, routed CIL inference and an adaptive mask.

Motivation (measured on snv_class_il_ht_r13_s42 / r3_s42, D_HT validation):

* Dense CIL training fits the new task against frozen old heads that read
  unmasked features they were never calibrated on.  First-epoch validation
  losses of 33-830 follow, and the only way the optimiser can win is to blow up
  the new head / norm scale: from task 2-3 on, the task subnetworks emit an
  input-independent constant (mean max-logit 200 ... 22 735 for every input)
  and sit at chance even under their own head (TIL 0.18-0.23).
* The concatenated argmax is then owned 100% by whichever dead subnetwork has
  the largest constant, so every older task reads 0.00.

Fixes, each switchable for ablation:

``task_local``  every phase (dense, consolidation, validation, valuation) uses
                the current task's head only -- exactly the TIL objective.  Old
                heads never enter the loss, so nothing pushes the scale up.
``routing``     CIL prediction concatenates per-head log-softmax scores. The
                argmax over that concatenation equals: pick the subnetwork
                whose own head is most confident, then argmax inside it -- the
                task inference used by mask-isolation methods (SupSup, WSN).
``adaptive``    |S_t| is chosen per task instead of a fixed floor(c*N).  After
                phi is estimated, neurons are added in decreasing-phi order
                (Data Shapley's "add high-value points first" curve) and S_t is
                the smallest prefix whose masked value comes within a
                tolerance of V(full).  Neurons with phi <= 0 are never taken
                (Data Shapley: low/negative-value points hurt the model).
                The tolerance is the larger of ``adaptive_tol`` x (V(full) -
                V(empty)) and the bootstrap std of V(full) over the validation
                set -- TMC-Shapley's own performance-tolerance rule.  Already
                frozen neurons are free: only newly frozen neurons count
                against ``adaptive_max`` x N.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from method_loader import load

_core = load('SNV', 'snv_core')      # the shim at source/snv_core.py omits _forward
SNVContinualLearner = _core.SNVContinualLearner
MeanActivationComputer = _core.MeanActivationComputer
install_subnetwork_mask = _core.install_subnetwork_mask
_forward = _core._forward


class _Shifted:
    """Loader view with labels shifted down by ``offset``."""

    def __init__(self, loader, offset):
        self.loader, self.offset = loader, offset
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for x, y in self.loader:
            yield x, y - self.offset


class SNVAdaptive(SNVContinualLearner):

    def __init__(self, *args, task_local: bool = True, routing: str = 'maxprob',
                 adaptive: bool = True, adaptive_tol: float = 0.05,
                 adaptive_min: float = 0.02, adaptive_max: float = 0.2,
                 adaptive_grid: int = 24, bootstrap: int = 200,
                 frozen_norm_eval: bool = True, adaptive_rule: str = 'coverage',
                 adaptive_coverage: float = 0.9, bn_recal: bool = True,
                 recal_samples: int = 2048, bn_recal_mode: str = 'batch',
                 rot_aux: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if adaptive_rule not in ('coverage', 'curve'):
            raise ValueError(f'unknown adaptive_rule {adaptive_rule!r}')
        self.frozen_norm_eval = frozen_norm_eval
        self.adaptive_rule = adaptive_rule
        self.adaptive_coverage = adaptive_coverage
        if routing not in ('concat', 'maxprob', 'entropy', 'negent_z', 'rot_energy_z'):
            raise ValueError(f'unknown routing {routing!r}')
        self.route_stats: Dict[int, Tuple[float, float]] = {}
        # (energy mean, energy sd, rotation-score mean, rotation-score sd) per
        # task, from its own training images through its own subnetwork.
        self.rot_route_stats: Dict[int, Tuple[float, float, float, float]] = {}
        self.task_local = task_local
        self.routing = routing
        self.adaptive = adaptive
        self.adaptive_tol = adaptive_tol
        self.adaptive_min = adaptive_min
        self.adaptive_max = adaptive_max
        self.adaptive_grid = adaptive_grid
        self.bootstrap = bootstrap
        self._current_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._current_task: Optional[int] = None
        self.bn_recal = bn_recal
        self.recal_samples = recal_samples
        self._recal_ctx = None
        self._recal_stats = None
        self._final_recal = None
        self._estimating = False
        # 'eval'  : every consolidation epoch runs all norms in eval mode on
        #           re-estimated masked statistics.  Measured (3 tasks): keeps
        #           every subnetwork input-dependent but hurts optimisation
        #           (ACC 0.298 vs 0.364; T1 masked 0.38 -> 0.30).
        # 'batch' : consolidation trains with ordinary train-mode BN through the
        #           mask (batch statistics of the masked forward); the masked
        #           population statistics are estimated only for validation
        #           and for task_norm_state[t].  Training and inference then see
        #           the same normalisation.
        if bn_recal_mode not in ('eval', 'batch'):
            raise ValueError(f'unknown bn_recal_mode {bn_recal_mode!r}')
        self.bn_recal_mode = bn_recal_mode
        self._batch_consolidation = False
        # Rotation self-supervision for task routing.  Measured on the 10-task
        # batch-mode checkpoint: every subnetwork is healthy (oracle-TIL 0.86)
        # but task identity is recovered for only 33-38% of inputs; softmax and
        # feature-Gaussian scores cannot separate tasks.  A per-task 4-way
        # rotation head on the head-input features, trained with the task
        # (dense and consolidation, hence through the mask), is a known OOD
        # signal (Hendrycks et al. 2019).  One random rotation per sample; BN
        # running statistics are not updated by rotated batches.
        self.rot_aux = rot_aux

    # -- training objective ------------------------------------------------- #
    def _rotation_step(self, x, task_id):
        """Backpropagate rot_aux x CE(rotation head) on one random rotation/sample.

        The parent's loops compute the loss from _train_logits alone, after
        optimizer.zero_grad() and before loss.backward(); gradients of this
        separate graph accumulate first and the same gradient mask (M_{t-1})
        is applied to their sum.
        """
        k = torch.randint(0, 4, (x.shape[0],), device=x.device)
        xr = x.clone()
        for r in (1, 2, 3):
            sel = k == r
            if sel.any():
                xr[sel] = torch.rot90(x[sel], r, dims=(2, 3))
        norms = self._norm_modules()
        momenta = {n: mod.momentum for n, mod in norms.items()}
        feat = {}
        hk = self.model.heads[str(task_id)].register_forward_hook(
            lambda mod, i, o: feat.__setitem__('f', i[0]))
        try:
            for mod in norms.values():
                mod.momentum = 0.0                 # rotated batches leave running stats alone
            _forward(self.model, xr, task_id)
        finally:
            hk.remove()
            for n, mod in norms.items():
                mod.momentum = momenta[n]
        loss = self.rot_aux * F.cross_entropy(self.model.rot_heads[str(task_id)](feat['f']), k)
        loss.backward()

    def _train_logits(self, x, task_id):
        if (self.rot_aux > 0 and self.model.training and torch.is_grad_enabled()
                and not self._estimating):
            self._rotation_step(x, task_id)
        if self.task_local:
            return _forward(self.model, x, task_id)
        return super()._train_logits(x, task_id)

    def _cache_val_batches(self, val_loader):
        self._current_batches = super()._cache_val_batches(val_loader)
        return self._current_batches

    def _value_and_freeze(self, task_id, batches, tracker, verbose, **kw):
        # The valuation, the mean responses and V(S) must score the same
        # objective the task was trained on; the parent picks the head from
        # ``scenario``, so present the task as TIL for the duration.
        self._current_task = task_id
        scenario = self.scenario
        if self.task_local:
            self.scenario = 'task_il'
        self._final_recal = None
        try:
            result = super()._value_and_freeze(task_id, batches, tracker, verbose, **kw)
        finally:
            self.scenario = scenario
        # The parent stored task_norm_state[t] from the live buffers, which the
        # recalibration never writes.  Inference for task t loads this per-task
        # copy only, so the masked statistics go here and nowhere else.
        if self._final_recal is not None and task_id in self.task_norm_state:
            for name, (mean, var) in self._final_recal.items():
                state = self.task_norm_state[task_id].get(name)
                if state is not None:
                    state['running_mean'] = mean.clone()
                    state['running_var'] = var.clone()
            result['bn_recalibrated_layers'] = len(self._final_recal)
            self._final_recal = None
        return result

    # -- BatchNorm recalibration under the task mask ------------------------- #
    # Measured on both 10-task pilots: stored BN statistics describe the DENSE
    # network, but Eq.(1) runs the mean-clamped subnetwork.  Kept channels then
    # see inputs tens to hundreds of sigma off their stats (|BN out| 100-600 by
    # layer3); once early layers keep few channels (task 5+) the input-dependent
    # part is ~3e-4 of that offset and the head reads a constant.  Post-hoc
    # recalibration restores the signal (head-input rel-std 0.000 -> 0.4-1.8)
    # but lowers accuracy because heads were fitted under the stale stats, so
    # it is done here, inside consolidation, and stored per task.
    #
    # The statistics are applied by a forward hook, never written to the live
    # buffers (the buffers are borrowed for the estimate and restored exactly),
    # so the frozen-state snapshot, drift checks and later tasks' dense
    # training are unaffected.  Only task_norm_state[t] -- a per-task copy that
    # inference loads for task t alone -- receives the recalibrated values.
    def _norm_modules(self):
        return {n: mod for n, mod in self.model.named_modules()
                if isinstance(mod, nn.modules.batchnorm._BatchNorm)}

    @torch.no_grad()
    def _estimate_masked_stats(self, task_mask, means, loader):
        norms = self._norm_modules()
        saved = {n: {k: v.clone() for k, v in mod.state_dict().items()} for n, mod in norms.items()}
        modes = {n: (mod.training, mod.momentum) for n, mod in norms.items()}
        was_training = self.model.training
        active = self._recal_stats
        self._recal_stats = None                  # recal hooks off while estimating
        self._estimating = True                   # frozen-BN hooks off too
        handles = install_subnetwork_mask(self.mask_manager.groups, task_mask, means, self.device)
        try:
            self.model.eval()
            for mod in norms.values():
                mod.reset_running_stats(); mod.momentum = None; mod.train()
            seen = 0
            for x, _ in loader:
                _forward(self.model, x.to(self.device), self._current_task)
                seen += len(x)
                if seen >= self.recal_samples:
                    break
            stats = {n: (mod.running_mean.clone(), mod.running_var.clone())
                     for n, mod in norms.items()}
        finally:
            for h in handles:
                h.remove()
            for n, mod in norms.items():
                mod.load_state_dict(saved[n])
                mod.training, mod.momentum = modes[n]
            self.model.train(was_training)
            self._recal_stats = active
            self._estimating = False
        return stats

    def _recal_hooks(self):
        handles = []
        for name, mod in self._norm_modules().items():
            def hook(module, inputs, output, name=name):
                stats = self._recal_stats
                if stats is None or name not in stats:
                    return output
                if self._batch_consolidation and module.training:
                    return output             # batch mode trains on batch stats
                mean, var = stats[name]
                x = inputs[0]
                shape = (1, -1) + (1,) * (x.dim() - 2)
                y = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + module.eps)
                if module.affine:
                    y = y * module.weight.view(shape) + module.bias.view(shape)
                return y
            handles.append(mod.register_forward_hook(hook))
        return handles

    def _set_frozen_norms_eval(self):
        # Called by the parent at the start of every consolidation epoch.  With
        # recalibration on, every norm layer runs on the masked statistics
        # (eval semantics, no running-stat updates) and those are re-estimated
        # for the weights as they stand at this epoch.
        if not (self.bn_recal and self._recal_ctx is not None):
            return super()._set_frozen_norms_eval()
        if self._batch_consolidation:
            # Leave every norm in train mode: batch statistics of the masked
            # forward.  Recal stats are only switched on for validation.
            self._recal_stats = None
            return
        task_mask, means, loader = self._recal_ctx
        self._recal_stats = self._estimate_masked_stats(task_mask, means, loader)
        for mod in self._norm_modules().values():
            mod.eval()

    def _consolidate(self, task_id, task_mask, means, train_loader, val_loader,
                     num_epochs, patience, verbose):
        if not self.bn_recal:
            return super()._consolidate(task_id, task_mask, means, train_loader, val_loader,
                                        num_epochs, patience, verbose)
        self._recal_ctx = (task_mask, means, train_loader)
        self._batch_consolidation = self.bn_recal_mode == 'batch'
        self._recal_stats = self._estimate_masked_stats(task_mask, means, train_loader)
        handles = self._recal_hooks()
        try:
            out = super()._consolidate(task_id, task_mask, means, train_loader, val_loader,
                                       num_epochs, patience, verbose)
            # Final estimate for the weights consolidation returned (best epoch).
            self._final_recal = self._estimate_masked_stats(task_mask, means, train_loader)
            self._recal_stats = self._final_recal
            # The parent removed its mask hooks on return; score the subnetwork,
            # not the dense network under the subnetwork's statistics.
            mask_handles = install_subnetwork_mask(self.mask_manager.groups, task_mask,
                                                   means, self.device)
            try:
                out['masked_val_acc_recal'] = self._validate(val_loader, task_id,
                                                             nn.CrossEntropyLoss())[1]
            finally:
                for h in mask_handles:
                    h.remove()
            return out
        finally:
            for h in handles:
                h.remove()
            self._recal_ctx = None
            self._recal_stats = None
            self._batch_consolidation = False

    def _validate(self, loader, task_id, criterion):
        # Batch-mode consolidation: score (and early-stop) on the masked
        # population statistics that inference will load, re-estimated for the
        # current weights.  The parent's mask hooks are installed at this point.
        if self._batch_consolidation and self._recal_ctx is not None:
            task_mask, means, train_loader = self._recal_ctx
            self._recal_stats = self._estimate_masked_stats(task_mask, means, train_loader)
            try:
                return super()._validate(loader, task_id, criterion)
            finally:
                self._recal_stats = None
        return super()._validate(loader, task_id, criterion)

    # -- adaptive S_t -------------------------------------------------------- #
    @torch.no_grad()
    def _masked_losses(self, active, means, head):
        """Per-sample CE of the task subnetwork ``active`` on the cached batches."""
        handles = install_subnetwork_mask(self.mask_manager.groups, active, means, self.device)
        self.model.eval()
        try:
            out = []
            for x, y in self._current_batches:
                logits = _forward(self.model, x.to(self.device), head)
                out.append(F.cross_entropy(logits.float(), y.to(self.device), reduction='none'))
            return torch.cat(out)
        finally:
            for h in handles:
                h.remove()

    def _select(self, phi_abl, phi_run, k):
        if not self.adaptive or self.selection != 'ablation' or phi_abl is None:
            return super()._select(phi_abl, phi_run, k)
        N = self.mask_manager.num_neurons
        head = None if (self.scenario == 'class_il' and not self.task_local) else self._current_task
        # The same mean responses the parent stores for masked inference: one
        # validation pass on the unmasked model, before anything is frozen.
        means = MeanActivationComputer(self.model, self.mask_manager.groups,
                                       self.device).compute(self._current_batches, head)
        frozen = self.mask_manager.cumulative_mask.to(phi_abl.device)

        full = torch.ones(N, dtype=torch.bool, device=phi_abl.device)
        loss_full = self._masked_losses(full, means, head)
        v_full = -loss_full.mean().item()
        v_empty = -self._masked_losses(~full, means, head).mean().item()
        g = torch.Generator().manual_seed(0)
        n = loss_full.numel()
        boot = torch.stack([loss_full[torch.randint(n, (n,), generator=g).to(loss_full.device)].mean()
                            for _ in range(self.bootstrap)])
        tol = max(self.adaptive_tol * abs(v_full - v_empty), boot.std().item())
        target = v_full - tol

        positive = phi_abl > 0
        order = torch.argsort(phi_abl, descending=True)
        order = order[positive[order]]
        # Cost of a prefix = newly frozen neurons in it.
        new_cum = torch.cumsum((~frozen[order]).long(), 0)
        k_min = max(1, int(self.adaptive_min * N))
        k_max = max(k_min, int(self.adaptive_max * N))
        limit = int((new_cum <= k_max).sum().item())
        limit = max(min(limit, order.numel()), min(k_min, order.numel()))

        if self.adaptive_rule == 'coverage':
            # Efficiency: sum_i phi_i = V(N) - V(empty).  S_t is the smallest
            # top-phi prefix holding ``adaptive_coverage`` of the positive value
            # mass.  No masked evaluation: measured pre-consolidation, masked V
            # FALLS as top-phi neurons are added (task 0: -1.68 at 96 neurons,
            # -3.03 at 960), so a curve criterion never fires and defaults to
            # the budget cap.
            mass = torch.cumsum(phi_abl[order].double(), 0)
            need = self.adaptive_coverage * float(mass[-1]) if mass.numel() else 0.
            size = int((mass < need).sum().item()) + 1
            chosen = min(max(size, min(k_min, order.numel())), limit)
            self._adaptive_log = dict(rule='coverage', coverage=self.adaptive_coverage,
                                      v_full=v_full, v_empty=v_empty,
                                      positive_mass=float(mass[-1]) if mass.numel() else 0.,
                                      coverage_size=size, chosen_size=chosen, fixed_k=k,
                                      limit=limit, positive_neurons=int(positive.sum().item()),
                                      capped=size > limit)
            return self._topk_with_layer_floor(self._prefix_scores(phi_abl, order[:chosen]), chosen)
        sizes = sorted({max(1, int(round(k_min + (limit - k_min) * i / max(1, self.adaptive_grid - 1))))
                        for i in range(self.adaptive_grid)} | {min(k, limit)})
        sizes = [s for s in sizes if 0 < s <= limit] or [limit]

        chosen, curve = sizes[-1], []
        for s in sizes:
            mask = self._topk_with_layer_floor(self._prefix_scores(phi_abl, order[:s]), s)
            v = -self._masked_losses(mask, means, head).mean().item()
            curve.append((s, v))
            if v >= target:
                chosen = s
                break
        self._adaptive_log = dict(v_full=v_full, v_empty=v_empty, tolerance=tol,
                                  bootstrap_std=boot.std().item(), target=target,
                                  chosen_size=chosen, fixed_k=k, limit=limit,
                                  positive_neurons=int(positive.sum().item()),
                                  reached_target=curve[-1][1] >= target, curve=curve)
        return self._topk_with_layer_floor(self._prefix_scores(phi_abl, order[:chosen]), chosen)

    @staticmethod
    def _prefix_scores(phi, prefix):
        """Scores that make ``prefix`` the top-|prefix| (layer floor still applies)."""
        s = torch.full_like(phi, float('-inf'))
        s[prefix] = phi[prefix]
        return s

    def train_task(self, task_id, train_loader, val_loader, num_epochs=200, patience=20,
                   verbose=True):
        self._adaptive_log = None
        if self.task_local and self.scenario == 'class_il':
            # CIL loaders carry global labels (5t+i); the task head is 5-way.
            # Training, consolidation and valuation see local labels; evaluation
            # keeps the global ones because predict() returns global logits.
            offset = task_id * self.model.classes_per_task
            train_loader = _Shifted(train_loader, offset)
            val_loader = _Shifted(val_loader, offset)
        if self.rot_aux > 0:
            # Registered on the model so the parent's optimizers pick it up and
            # checkpoints carry it; 'head' in the name keeps it out of the
            # neuron index (_is_classifier).  Older rotation heads are frozen.
            if not hasattr(self.model, 'rot_heads'):
                self.model.rot_heads = nn.ModuleDict()
            key = str(task_id)
            if key not in self.model.rot_heads:
                self.model.rot_heads[key] = nn.Linear(self.model.feature_dim, 4).to(self.device)
            for name, head in self.model.rot_heads.items():
                for p in head.parameters():
                    p.requires_grad_(name == key)
        handles = self._frozen_bn_hooks() if self.frozen_norm_eval else []
        try:
            result = self._train_task_inner(task_id, train_loader, val_loader, num_epochs,
                                            patience, verbose)
        finally:
            for h in handles:
                h.remove()
        if self.routing == 'rot_energy_z' and self.rot_aux > 0:
            # S_t is frozen now, so these statistics stay valid for the rest of
            # the sequence.  Training images only (no validation leakage).
            es, rs, seen = [], [], 0
            for x, _ in train_loader:
                _, e, r = self._route_scores(task_id, x.to(self.device))
                es.append(e); rs.append(r); seen += len(x)
                if seen >= self.recal_samples:
                    break
            e, r = torch.cat(es), torch.cat(rs)
            self.rot_route_stats[task_id] = (e.mean().item(), e.std().clamp_min(1e-2).item(),
                                             r.mean().item(), r.std().clamp_min(1e-2).item())
            result['rot_route_stats'] = self.rot_route_stats[task_id]
        return result

    def _frozen_bn_hooks(self):
        """Per-channel eval-mode BatchNorm for frozen channels during training.

        The parent's dense loop calls model.train(), so a frozen channel is
        normalised with batch statistics while training but with its restored
        running statistics at evaluation: the frozen function differs between
        the two.  Putting the whole layer in eval mode instead (what
        _set_frozen_norms_eval does) freezes the statistics of the plastic
        channels too and diverged to NaN from task 1.  Here plastic channels keep
        batch statistics and only frozen channels use the stored ones, read
        before this forward pass updates them.
        """
        handles = []
        for g in self.mask_manager.groups:
            bn = g.norm_module
            if not isinstance(bn, nn.modules.batchnorm._BatchNorm):
                continue

            def pre(module, inputs):
                # Off while _estimate_masked_stats resets and re-accumulates the
                # running stats: stashing those would normalise frozen channels
                # with mean 0 / var 1 and poison every downstream estimate.
                if module.training and not self._estimating:
                    module._snv_stats = (module.running_mean.clone(), module.running_var.clone())

            def post(module, inputs, output, g=g):
                stats = getattr(module, '_snv_stats', None)
                if (not module.training or stats is None or self._estimating
                        or self._batch_consolidation):
                    return output
                frozen = self.mask_manager.cumulative_mask[g.slice()]
                if not bool(frozen.any()):
                    return output
                x = inputs[0]
                shape = (1, -1) + (1,) * (x.dim() - 2)
                mean, var = (s.view(shape) for s in stats)
                y = (x - mean) / torch.sqrt(var + module.eps)
                if module.affine:
                    y = y * module.weight.view(shape) + module.bias.view(shape)
                return torch.where(frozen.view(shape), y, output)

            handles += [bn.register_forward_pre_hook(pre), bn.register_forward_hook(post)]
        return handles

    def _train_task_inner(self, task_id, train_loader, val_loader, num_epochs, patience,
                          verbose):
        result = super().train_task(task_id, train_loader, val_loader, num_epochs, patience, verbose)
        # Routing calibration: the task's own subnetwork on its own validation
        # data, recorded once S_t is frozen, so later tasks cannot change it.
        s = self._subnetwork_scores(task_id, self._current_batches)
        # Floor the spread: a near-constant subnetwork (smoke test: sd 1e-6)
        # would otherwise turn every z-score into +/- infinity and own routing.
        self.route_stats[task_id] = (s.mean().item(), s.std().clamp_min(1e-2).item())
        result['route_stats'] = self.route_stats[task_id]
        if self._adaptive_log is not None:
            result['adaptive'] = self._adaptive_log
            self.history[-1] = result
        return result

    # -- inference ------------------------------------------------------------ #
    @torch.no_grad()
    def _subnetwork_scores(self, task_id, batches):
        """Negative entropy of task_id's subnetwork+head on ``batches``."""
        handles, norm_now, norm_modules = self._install_task_subnetwork(task_id)
        self.model.eval()
        try:
            out = []
            for x, _ in batches:
                logp = F.log_softmax(_forward(self.model, x.to(self.device), task_id).float(), 1)
                out.append((logp.exp() * logp).sum(1))
            return torch.cat(out)
        finally:
            self._remove_task_subnetwork(handles, norm_now, norm_modules)

    @torch.no_grad()
    def _route_scores(self, c, x):
        """(logits, energy, rotation score) of subnetwork c on x.

        Rotation score = mean over the four rotations r of log p_rot_c(r | rot_r x):
        a subnetwork predicts rotations of its own task's images better than of
        other tasks'.  Measured on rot_aux=1.0 (10 tasks): energy_z + rot_z
        routing gives CIL 0.400 vs 0.375 energy_z alone and 0.347 without
        rotation training.
        """
        feat = {}
        hk = self.model.heads[str(c)].register_forward_hook(
            lambda mod, i, o: feat.__setitem__('f', i[0].float()))
        handles, norm_now, norm_modules = self._install_task_subnetwork(c)
        self.model.eval()
        try:
            rot = 0.
            for r in range(4):
                lg = _forward(self.model, torch.rot90(x, r, dims=(2, 3)), c).float()
                if r == 0:
                    logits = lg
                rot = rot + F.log_softmax(self.model.rot_heads[str(c)](feat['f']), 1)[:, r] / 4
        finally:
            self._remove_task_subnetwork(handles, norm_now, norm_modules)
            hk.remove()
        return logits, logits.logsumexp(1), rot

    def predict(self, x, task_id=None):
        if self.scenario != 'class_il' or self.routing == 'concat':
            return super().predict(x, task_id)
        if self.routing == 'rot_energy_z':
            x = x.to(self.device)
            outputs = []
            for c in self.model.active_tasks():
                if c not in self.rot_route_stats or str(c) not in getattr(self.model, 'rot_heads', {}):
                    # Future-task heads exist only under full_output_space (FWT
                    # cells).  An untrained task cannot be recognised, so it is
                    # never routed to; its logits keep the global class index.
                    # FWT under this rule is therefore 0 by construction.
                    with torch.no_grad():
                        logp = F.log_softmax(_forward(self.model, x, c).float(), 1)
                    outputs.append(logp - logp.max(1, keepdim=True).values - 1e9)
                    continue
                logits, e, r = self._route_scores(c, x)
                em, esd, rm, rsd = self.rot_route_stats[c]
                z = ((e - em) / esd + (r - rm) / rsd).unsqueeze(1)
                logp = F.log_softmax(logits, 1)
                outputs.append(logp - logp.max(1, keepdim=True).values + 1e4 * z)
            return torch.cat(outputs, 1)
        if self.routing == 'negent_z':
            # Route by negative entropy standardised with each subnetwork's own
            # validation statistics, then argmax inside the chosen head.  The
            # 1e4 scale makes the z-score decide the head; within-head order is
            # kept by the (<= 0) log-prob offsets.
            self.model.eval()
            x = x.to(self.device)
            outputs = []
            for c in self.model.active_tasks():
                handles, norm_now, norm_modules = self._install_task_subnetwork(c)
                try:
                    with torch.no_grad():
                        logp = F.log_softmax(_forward(self.model, x, c).float(), 1)
                finally:
                    self._remove_task_subnetwork(handles, norm_now, norm_modules)
                mu, sd = self.route_stats.get(c, (0.0, 1.0))
                z = ((logp.exp() * logp).sum(1, keepdim=True) - mu) / sd
                outputs.append(logp - logp.max(1, keepdim=True).values + 1e4 * z)
            return torch.cat(outputs, 1)
        self.model.eval()
        x = x.to(self.device)
        outputs = []
        for candidate in self.model.active_tasks():
            handles, norm_now, norm_modules = self._install_task_subnetwork(candidate)
            try:
                with torch.no_grad():
                    logp = F.log_softmax(_forward(self.model, x, candidate).float(), 1)
            finally:
                self._remove_task_subnetwork(handles, norm_now, norm_modules)
            if self.routing == 'entropy':
                # Shift each head by its negative entropy so the argmax picks the
                # least uncertain subnetwork, then the best class inside it.
                ent = -(logp.exp() * logp).sum(1, keepdim=True)
                logp = logp - logp.max(1, keepdim=True).values - ent + 1e3
            outputs.append(logp)
        return torch.cat(outputs, 1)
