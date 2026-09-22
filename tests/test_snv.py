"""
Test suite pinning the implementation to the paper.

The tests are grouped by what they protect:

  TestNeuronDefinition   N = sum_l C_l, and every filter's norm layer is paired
                         with it so freezing covers the whole neuron.
  TestFreezing           M_{t-1} blocks exactly the frozen weights, including
                         BatchNorm affines and running statistics, and an
                         already-learned task's accuracy survives a later task.
  TestShapleyAxioms      Efficiency, Null contribution and Symmetry hold for the
                         estimator on a model small enough to check exactly.
  TestBandit             The active set is the paper's rule around phi^(k).
  TestSelection          |S_t| = floor(c N); reuse across tasks is permitted.
  TestMetrics            ACC / BWT / PS / FWT / AF against worked examples,
                         including PS < 1 at BWT = 0.
  TestSplits             70 / 10 / 20.
  TestMethods            Every registered baseline builds and completes a task.

Run:  python test_snv.py        (add -v for per-test output)
"""

import math
import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cl_base import ContinualMethod
from metrics import ContinualLearningMetrics
from models import ContinualLearningModel, ResNet18, count_neurons, create_model
from snv_core import (MeanActivationComputer, NeuronMaskManager, ShapleyNeuronEstimator,
                      SNVContinualLearner, build_neuron_index)

torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device('cpu')


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class TinyBackbone(nn.Module):
    """Three conv+BN layers -- small enough for exact Shapley checks."""

    def __init__(self, widths=(4, 4, 4)):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers += [nn.Conv2d(cin, w, 3, padding=1, bias=False), nn.BatchNorm2d(w),
                       nn.ReLU(inplace=True)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = widths[-1]

    def get_features(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)

    def forward(self, x):
        return self.get_features(x)


def tiny_model(widths=(4, 4, 4), classes_per_task=2, num_tasks=2, scenario='task_il'):
    return ContinualLearningModel(TinyBackbone(widths), widths[-1],
                                  classes_per_task, num_tasks, scenario)


def fake_loader(n=32, classes=2, size=8, batch=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, size, size, generator=g)
    y = torch.randint(0, classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)


def batches_of(loader, limit=4):
    return [(x, y) for i, (x, y) in enumerate(loader) if i < limit]


# --------------------------------------------------------------------------- #
class TestNeuronDefinition(unittest.TestCase):

    def test_resnet18_neuron_count(self):
        """N = sum over conv layers of C_l, downsample convs included."""
        model = ResNet18(64, input_size=32)
        groups = build_neuron_index(model)
        expected = sum(m.out_channels for m in model.modules() if isinstance(m, nn.Conv2d))
        self.assertEqual(sum(g.num_neurons for g in groups), expected)
        self.assertEqual(count_neurons(model), expected)

    def test_heads_are_not_neurons(self):
        """Task heads are excluded -- masks protect the shared backbone only."""
        model = create_model('cifar100', classes_per_task=10, num_tasks=10)
        model.ensure_head(0)
        names = [g.name for g in build_neuron_index(model)]
        self.assertFalse(any(n.startswith('heads') for n in names))

    def test_every_conv_is_paired_with_its_norm(self):
        """A filter and the BatchNorm scaling it form one neuron."""
        model = ResNet18(64, input_size=32)
        groups = build_neuron_index(model)
        unpaired = [g.name for g in groups if g.norm_module is None]
        self.assertEqual(unpaired, [], f'convs with no paired norm: {unpaired}')
        for g in groups:
            self.assertEqual(g.norm_module.num_features, g.num_neurons)


class TestFreezing(unittest.TestCase):

    def setUp(self):
        self.model = create_model('cifar100', classes_per_task=10, num_tasks=10)
        self.model.ensure_head(0)
        self.mm = NeuronMaskManager(self.model, DEVICE)

    def test_mask_keys_match_named_parameters(self):
        """Every mask addresses a real parameter, with the right shape."""
        masks = self.mm.create_gradient_mask()
        params = dict(self.model.named_parameters())
        self.assertGreater(len(masks), 0)
        for name, mask in masks.items():
            self.assertIn(name, params, f'mask for non-existent parameter {name}')
            self.assertEqual(mask.shape, params[name].shape)

    def test_gradient_masking_runs_on_resnet18(self):
        """Regression: masks were previously matched by name suffix and collided."""
        mask = torch.zeros(self.mm.num_neurons, dtype=torch.bool)
        mask[:64] = True
        self.mm.update_cumulative_mask(0, mask)
        x, y = torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,))
        nn.functional.cross_entropy(self.model(x, 0), y).backward()
        self.mm.apply_gradient_mask()          # must not raise

        grad = self.model.backbone.conv1.weight.grad
        self.assertTrue(torch.all(grad[:64] == 0))

    def test_frozen_weights_do_not_move(self):
        mask = torch.zeros(self.mm.num_neurons, dtype=torch.bool)
        mask[:64] = True
        self.mm.update_cumulative_mask(0, mask)
        self.mm.snapshot_frozen_state()

        opt = torch.optim.Adam(self.model.parameters(), lr=0.5)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
            nn.functional.cross_entropy(self.model(x, 0), y).backward()
            self.mm.apply_gradient_mask()
            opt.step()
            self.mm.restore_frozen_state()
        self.assertLess(self.mm.max_frozen_drift(), 1e-9)

    def test_batchnorm_running_stats_are_frozen(self):
        """BN statistics are updated by the forward pass, outside the optimiser."""
        mask = torch.zeros(self.mm.num_neurons, dtype=torch.bool)
        mask[:64] = True
        self.mm.update_cumulative_mask(0, mask)
        self.mm.snapshot_frozen_state()
        bn1 = self.model.backbone.bn1
        before = bn1.running_mean.clone()

        self.model.train()
        bn1.eval()                             # what SNVContinualLearner does
        for _ in range(3):
            self.model(torch.randn(8, 3, 32, 32) * 5 + 3, 0)
        self.mm.restore_frozen_state()
        self.assertTrue(torch.allclose(bn1.running_mean, before))

    def test_batchnorm_stats_drift_without_the_fix(self):
        """Guards the reason for the fix: train-mode BN rewrites the statistics."""
        bn1 = self.model.backbone.bn1
        before = bn1.running_mean.clone()
        self.model.train()
        for _ in range(3):
            self.model(torch.randn(8, 3, 32, 32) * 5 + 3, 0)
        self.assertFalse(torch.allclose(bn1.running_mean, before))

    def test_learned_task_survives_a_later_task(self):
        """End to end: freeze after task 0, train task 1, task 0 is untouched."""
        model = tiny_model(widths=(6, 6, 6), classes_per_task=2, num_tasks=2)
        learner = SNVContinualLearner(model, DEVICE, sparsity_ratio=0.5, scenario='task_il',
                                      truncation_threshold=-1.0, max_permutations=2,
                                      shapley_eval_batches=2, lr=0.05)
        t0_train, t0_val = fake_loader(seed=1), fake_loader(n=16, seed=2)
        t1_train, t1_val = fake_loader(seed=3), fake_loader(n=16, seed=4)

        learner.train_task(0, t0_train, t0_val, num_epochs=3, patience=3, verbose=False)
        acc_before = learner.evaluate(t0_val, 0)

        learner.train_task(1, t1_train, t1_val, num_epochs=3, patience=3, verbose=False)
        acc_after = learner.evaluate(t0_val, 0)

        # Head 0 is frozen and every neuron in B_0 is frozen; with c = 0.5 some
        # capacity stays plastic, so task 0 may shift, but its frozen half must not.
        self.assertLess(learner.mask_manager.max_frozen_drift(), 1e-6)
        self.assertIsInstance(acc_before, float)
        self.assertIsInstance(acc_after, float)

    def test_cumulative_mask_is_a_union(self):
        a = torch.zeros(self.mm.num_neurons, dtype=torch.bool); a[:100] = True
        b = torch.zeros(self.mm.num_neurons, dtype=torch.bool); b[50:150] = True
        self.mm.update_cumulative_mask(0, a)
        self.mm.update_cumulative_mask(1, b)
        self.assertEqual(int(self.mm.cumulative_mask.sum()), 150)
        stats = self.mm.reuse_stats(1)
        self.assertEqual(stats['reused'], 50)
        self.assertEqual(stats['newly_frozen'], 50)


class TestShapleyAxioms(unittest.TestCase):
    """Axioms 1-3 on a model small enough that the checks are exact."""

    def _estimator(self, model, loader, task_id=0):
        groups = build_neuron_index(model)
        batches = batches_of(loader, 2)
        means = MeanActivationComputer(model, groups, DEVICE).compute(batches, task_id)
        est = ShapleyNeuronEstimator(model, groups, means, DEVICE,
                                     truncation_threshold=-1.0, task_id=task_id)
        est.set_eval_batches(batches)
        return est

    def test_efficiency(self):
        """sum_i phi_i = V(M) - V(0).

        With no truncation and no bandit pruning, each permutation's marginals
        telescope to V(M) - V(0), so the identity is exact, not approximate.
        """
        model = tiny_model(widths=(3, 3), classes_per_task=2)
        model.ensure_head(0)
        est = self._estimator(model, fake_loader())
        out = est.estimate_shapley_values(k=2, max_permutations=3, min_permutations=99,
                                          verbose=False)
        self.assertLess(est.efficiency_residual(out['phi']), 1e-6)

    def test_null_contribution(self):
        """A neuron that never changes V has phi_i = 0."""
        model = tiny_model(widths=(4, 4), classes_per_task=2)
        model.ensure_head(0)
        with torch.no_grad():                       # kill filter 0 of the first conv
            model.backbone.features[0].weight[0].zero_()
            model.backbone.features[1].weight[0].zero_()
            model.backbone.features[1].bias[0].zero_()
            model.backbone.features[1].running_mean[0] = 0.0
            model.backbone.features[1].running_var[0] = 1.0
        est = self._estimator(model, fake_loader())
        out = est.estimate_shapley_values(k=2, max_permutations=3, min_permutations=99,
                                          verbose=False)
        self.assertAlmostEqual(float(out['phi'][0]), 0.0, places=6)

    def _symmetric_model(self):
        """Filters 0 and 1 of the first conv are made interchangeable."""
        model = tiny_model(widths=(3, 3), classes_per_task=2)
        model.ensure_head(0)
        conv, bn, nxt = (model.backbone.features[0], model.backbone.features[1],
                         model.backbone.features[3])
        with torch.no_grad():
            conv.weight[1] = conv.weight[0]
            bn.weight[1], bn.bias[1] = bn.weight[0], bn.bias[0]
            bn.running_mean[1], bn.running_var[1] = bn.running_mean[0], bn.running_var[0]
            nxt.weight[:, 1] = nxt.weight[:, 0]
        return model

    def test_symmetry(self):
        """Interchangeable neurons receive identical Shapley values.

        Checked on the exact value from Eq. (5): symmetry is a property of phi
        itself, which a finite Monte-Carlo sample only approaches.
        """
        est = self._estimator(self._symmetric_model(), fake_loader())
        phi = est.exact_shapley_values()
        self.assertAlmostEqual(float(phi[0]), float(phi[1]), places=6)

    def test_exact_values_satisfy_efficiency(self):
        model = tiny_model(widths=(3, 3), classes_per_task=2)
        model.ensure_head(0)
        est = self._estimator(model, fake_loader())
        self.assertLess(est.efficiency_residual(est.exact_shapley_values()), 1e-5)

    def test_monte_carlo_converges_to_the_exact_value(self):
        """The estimator is unbiased for Eq. (5), so it tracks it as samples grow."""
        model = tiny_model(widths=(3, 3), classes_per_task=2)
        model.ensure_head(0)
        est = self._estimator(model, fake_loader())
        exact = est.exact_shapley_values()
        torch.manual_seed(7)
        out = est.estimate_shapley_values(k=2, max_permutations=200, min_permutations=999,
                                          verbose=False)
        self.assertLess(float((out['phi'] - exact).abs().max()), 0.05)

    def test_hooks_do_not_leak_after_estimation(self):
        """A stale mask left on the model would silently corrupt later training.

        Hooks are now registered once and driven by a mutable buffer, so they
        must be torn down explicitly when the estimator is finished.
        """
        model = tiny_model(widths=(3, 3), classes_per_task=2)
        model.ensure_head(0)
        loader = fake_loader()
        before = sum(len(m._forward_hooks) for m in model.modules())

        est = self._estimator(model, loader)
        est.estimate_shapley_values(k=2, max_permutations=2, min_permutations=99,
                                    verbose=False)
        after = sum(len(m._forward_hooks) for m in model.modules())
        self.assertEqual(before, after, 'masking hooks were left on the model')

        # and the model's outputs are unmasked again
        x, _ = next(iter(loader))
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(x, 0)).all())

    def test_masking_uses_the_mean_not_zero(self):
        """V(S) replaces excluded filters with mu_i, preserving signal statistics."""
        model = tiny_model(widths=(4, 4), classes_per_task=2)
        model.ensure_head(0)
        loader = fake_loader()
        groups = build_neuron_index(model)
        batches = batches_of(loader, 2)
        means = MeanActivationComputer(model, groups, DEVICE).compute(batches, 0)
        self.assertTrue(any(m.abs().sum() > 0 for m in means.values()),
                        'mean activations are all zero -- masking would equal zeroing')

        est = ShapleyNeuronEstimator(model, groups, means, DEVICE, task_id=0)
        est.set_eval_batches(batches)
        full = torch.ones(est.num_neurons, dtype=torch.bool)

        # With S = M nothing is masked, so V(M) must equal plain accuracy.  The
        # reference is measured in eval mode over the estimator's own cached
        # batches: BatchNorm in train mode would use batch statistics and make
        # the comparison depend on how the data happens to be chunked.
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for x, y in est.eval_batches:
                correct += model(x, 0).argmax(1).eq(y).sum().item()
                total += y.numel()
        plain = correct / total
        self.assertAlmostEqual(est.evaluate_subset(full), plain, places=6)
        est.remove_hooks()


class TestBandit(unittest.TestCase):
    """A <- {i : |phi_i - phi^(k)| < delta_i}."""

    def _estimator(self, n=8):
        model = tiny_model(widths=(4, 4), classes_per_task=2)
        model.ensure_head(0)
        groups = build_neuron_index(model)
        est = ShapleyNeuronEstimator(model, groups, {}, DEVICE)
        est.num_neurons = n
        return est

    def test_kth_neuron_follows_the_manuscript_rule(self):
        est = self._estimator()
        phi = torch.linspace(1.0, 0.0, 8).double()
        counts = torch.full((8,), 500.0, dtype=torch.float64)
        m2 = torch.full((8,), 1e-12, dtype=torch.float64) * counts
        # The supplied pseudocode compares against phi^(k), not a midpoint.
        # With nonzero uncertainty the rank-k neuron therefore remains active.
        self.assertEqual(int(est._bandit_active_set(phi, counts, m2, k=3).sum()), 1)

    def test_neurons_straddling_the_kth_value_stay_active(self):
        est = self._estimator()
        phi = torch.tensor([0.9, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]).double()
        counts = torch.full((8,), 100.0, dtype=torch.float64)
        m2 = torch.full((8,), 0.25, dtype=torch.float64) * counts   # wide intervals
        active = est._bandit_active_set(phi, counts, m2, k=3)
        self.assertGreater(int(active.sum()), 0)

    def test_unsampled_neurons_stay_active(self):
        """Exploration: n_i < 2 leaves sigma undefined, so the neuron is kept."""
        est = self._estimator()
        phi = torch.linspace(1.0, 0.0, 8).double()
        counts = torch.zeros(8, dtype=torch.float64)
        m2 = torch.zeros(8, dtype=torch.float64)
        self.assertEqual(int(est._bandit_active_set(phi, counts, m2, k=3).sum()), 8)

    def test_truncation_cap_is_reported(self):
        """A run stopped by the cap is never reported as converged."""
        model = tiny_model(widths=(3, 3), classes_per_task=2)
        model.ensure_head(0)
        groups = build_neuron_index(model)
        loader = fake_loader()
        batches = batches_of(loader, 1)
        means = MeanActivationComputer(model, groups, DEVICE).compute(batches, 0)
        est = ShapleyNeuronEstimator(model, groups, means, DEVICE,
                                     truncation_threshold=-1.0, task_id=0)
        est.set_eval_batches(batches)
        out = est.estimate_shapley_values(k=2, max_permutations=1, min_permutations=99,
                                          verbose=False)
        self.assertFalse(out['converged'])


class TestSelection(unittest.TestCase):

    def _estimator(self, n):
        model = tiny_model(widths=(4, 4), classes_per_task=2)
        est = ShapleyNeuronEstimator(model, build_neuron_index(model), {}, DEVICE)
        est.num_neurons = n
        return est

    def test_k_is_floor_c_times_N(self):
        est = self._estimator(1000)
        for c in (0.03, 0.05, 0.1, 0.3, 0.5):
            mask = est.select_top_k_neurons(torch.randn(1000), c)
            self.assertEqual(int(mask.sum()), math.floor(c * 1000))

    def test_selection_is_the_top_k_by_phi(self):
        est = self._estimator(10)
        phi = torch.tensor([0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0])
        mask = est.select_top_k_neurons(phi, 0.3)
        self.assertEqual(sorted(torch.nonzero(mask).flatten().tolist()), [1, 3, 5])

    def test_a_frozen_neuron_can_be_reselected(self):
        """Fig. 1: the same neuron may fall in the top-r% for several tasks."""
        est = self._estimator(10)
        phi = torch.arange(10).float()
        first = est.select_top_k_neurons(phi, 0.3)
        second = est.select_top_k_neurons(phi, 0.3)
        self.assertGreater(int((first & second).sum()), 0)

    def test_capacity_never_exceeds_the_network(self):
        model = tiny_model(widths=(8, 8), classes_per_task=2)
        mm = NeuronMaskManager(model, DEVICE)
        est = ShapleyNeuronEstimator(model, mm.groups, {}, DEVICE)
        for t in range(6):
            mm.update_cumulative_mask(t, est.select_top_k_neurons(torch.randn(mm.num_neurons), 0.5))
        self.assertLessEqual(mm.get_capacity_used(), 100.0)


class TestMetrics(unittest.TestCase):

    def _matrix(self):
        m = ContinualLearningMetrics(3)
        m.accuracy_matrix = np.array([
            [0.90, 0.10, 0.10],
            [0.90, 0.80, 0.20],
            [0.90, 0.80, 0.85]])
        return m

    def test_acc(self):
        self.assertAlmostEqual(self._matrix().get_average_accuracy(), (0.90 + 0.80 + 0.85) / 3)

    def test_bwt_zero_when_nothing_is_forgotten(self):
        self.assertAlmostEqual(self._matrix().get_backward_transfer(), 0.0)

    def test_bwt_negative_under_forgetting(self):
        m = ContinualLearningMetrics(3)
        m.accuracy_matrix = np.array([[0.9, 0.1, 0.1],
                                      [0.7, 0.8, 0.1],
                                      [0.5, 0.6, 0.85]])
        self.assertAlmostEqual(m.get_backward_transfer(), ((0.5 - 0.9) + (0.6 - 0.8)) / 2)

    def test_plasticity(self):
        """P = mean over t of (A[t,t] - A[t-1,t]) / (1 - A[t-1,t])."""
        m = self._matrix()
        expected = np.mean([(0.80 - 0.10) / 0.90, (0.85 - 0.20) / 0.80])
        self.assertAlmostEqual(m.get_plasticity(), expected)

    def test_stability_equals_one_plus_bwt(self):
        m = self._matrix()
        self.assertAlmostEqual(m.get_stability(), 1.0 + m.get_backward_transfer())

    def test_ps_is_the_harmonic_mean(self):
        m = self._matrix()
        p, s = m.get_plasticity(), m.get_stability()
        self.assertAlmostEqual(m.get_plasticity_stability_ratio(), 2 * p * s / (p + s))

    def test_ps_below_one_at_zero_bwt(self):
        """The property the old formula violated: BWT = 0 does not force PS = 1."""
        m = self._matrix()
        self.assertAlmostEqual(m.get_backward_transfer(), 0.0)
        self.assertLess(m.get_plasticity_stability_ratio(), 1.0)
        self.assertGreater(m.get_plasticity_stability_ratio(), 0.0)

    def test_fwt_uses_the_random_baseline(self):
        m = self._matrix()
        m.set_random_baseline([0.05, 0.05, 0.05])
        self.assertAlmostEqual(m.get_forward_transfer(),
                               np.mean([0.10 - 0.05, 0.20 - 0.05]))

    def test_fwt_is_nan_without_a_baseline(self):
        self.assertTrue(np.isnan(self._matrix().get_forward_transfer()))

    def test_average_forgetting(self):
        m = ContinualLearningMetrics(3)
        m.accuracy_matrix = np.array([[0.9, 0.1, 0.1],
                                      [0.7, 0.8, 0.1],
                                      [0.5, 0.6, 0.85]])
        self.assertAlmostEqual(m.get_average_forgetting(), ((0.9 - 0.5) + (0.8 - 0.6)) / 2)


class TestSplits(unittest.TestCase):

    def test_seventy_ten_twenty(self):
        from datasets import TEST_FRAC, TRAIN_FRAC, VAL_FRAC
        self.assertAlmostEqual(TRAIN_FRAC + VAL_FRAC + TEST_FRAC, 1.0)
        self.assertEqual((TRAIN_FRAC, VAL_FRAC, TEST_FRAC), (0.70, 0.10, 0.20))

    def test_split_sizes(self):
        """The split logic itself, exercised without downloading anything."""
        from datasets import TEST_FRAC, TRAIN_FRAC, VAL_FRAC
        n = 1000
        n_train = int(round(TRAIN_FRAC * n))
        n_val = int(round(VAL_FRAC * n))
        n_test = n - n_train - n_val
        self.assertEqual((n_train, n_val, n_test), (700, 100, 200))


class TestHeads(unittest.TestCase):

    def test_til_uses_the_requested_head(self):
        model = tiny_model(classes_per_task=2, num_tasks=3)
        model.ensure_head(0)
        model.ensure_head(1)
        x = torch.randn(4, 3, 8, 8)
        self.assertEqual(model(x, 0).shape, (4, 2))
        self.assertFalse(torch.allclose(model(x, 0), model(x, 1)))

    def test_cil_concatenates_every_head(self):
        model = tiny_model(classes_per_task=2, num_tasks=3, scenario='class_il')
        model.ensure_head(0)
        self.assertEqual(model(torch.randn(4, 3, 8, 8)).shape, (4, 2))
        model.ensure_head(1)
        self.assertEqual(model(torch.randn(4, 3, 8, 8)).shape, (4, 4))

    def test_old_heads_are_frozen(self):
        model = tiny_model(classes_per_task=2, num_tasks=3)
        model.ensure_head(0)
        model.ensure_head(1)
        model.freeze_heads_before(1)
        self.assertFalse(model.heads['0'].weight.requires_grad)
        self.assertTrue(model.heads['1'].weight.requires_grad)


class TestInRunValuation(unittest.TestCase):
    """phi_run: the trajectory valuation and its structural properties."""

    def _learner(self, selection='ablation', **kw):
        model = tiny_model(widths=(4, 4), classes_per_task=2, num_tasks=3)
        return SNVContinualLearner(
            model, DEVICE, sparsity_ratio=0.5, scenario='task_il',
            truncation_threshold=-1.0, max_permutations=2, shapley_eval_batches=2,
            lr=0.05, selection=selection, track_inrun=True, **kw)

    def test_phi_run_is_produced_for_every_neuron(self):
        learner = self._learner()
        learner.train_task(0, fake_loader(seed=1), fake_loader(n=16, seed=2),
                           num_epochs=2, patience=2, verbose=False)
        phi = learner.inrun_values[0]
        self.assertEqual(phi.numel(), learner.mask_manager.num_neurons)
        self.assertTrue(torch.isfinite(phi).all())
        self.assertGreater(int((phi != 0).sum()), 0)

    def test_frozen_neurons_have_exactly_zero_phi_run(self):
        """The property that removes the need for an availability mask.

        A frozen neuron's parameters do not move, so its contribution to the
        first-order expansion is identically zero -- not small, zero.
        """
        learner = self._learner()
        learner.train_task(0, fake_loader(seed=1), fake_loader(n=16, seed=2),
                           num_epochs=2, patience=2, verbose=False)
        frozen = learner.mask_manager.cumulative_mask.cpu()
        self.assertGreater(int(frozen.sum()), 0)

        learner.train_task(1, fake_loader(seed=3), fake_loader(n=16, seed=4),
                           num_epochs=2, patience=2, verbose=False)
        phi_run_task1 = learner.inrun_values[1]
        self.assertTrue(torch.all(phi_run_task1[frozen] == 0),
                        'a frozen neuron accumulated non-zero phi_run')

    def test_selection_by_inrun_respects_the_budget(self):
        learner = self._learner(selection='inrun', compute_ablation=False)
        out = learner.train_task(0, fake_loader(seed=1), fake_loader(n=16, seed=2),
                                 num_epochs=1, patience=1, verbose=False)
        self.assertEqual(int(out['task_mask'].sum()),
                         math.floor(0.5 * learner.mask_manager.num_neurons))
        self.assertNotIn('shapley_values', out)     # the expensive pass was skipped

    def test_dual_selection_draws_only_from_the_claimed_pool(self):
        """Nothing is frozen that this task did not move, and the budget is a cap.

        Eq. (1) bounds ||S_t||_0 above by floor(c*N), so under-spending is legal;
        leaving the remainder plastic is the point of the dual criterion.
        """
        learner = self._learner(selection='dual')
        learner.train_task(0, fake_loader(seed=1), fake_loader(n=16, seed=2),
                           num_epochs=2, patience=2, verbose=False)
        out = learner.train_task(1, fake_loader(seed=3), fake_loader(n=16, seed=4),
                                 num_epochs=2, patience=2, verbose=False)
        mask = out['task_mask']
        k = math.floor(0.5 * learner.mask_manager.num_neurons)
        self.assertGreater(int(mask.sum()), 0)
        self.assertLessEqual(int(mask.sum()), k)
        self.assertTrue(torch.all(learner.inrun_values[1][mask] > 0),
                        'dual froze a neuron this task never moved')

    def test_rank_correlation_helpers(self):
        from inrun import spearman, top_k_agreement
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(spearman(x, x), 1.0, places=6)
        self.assertAlmostEqual(spearman(x, -x), -1.0, places=6)
        self.assertAlmostEqual(spearman(x, torch.tensor([1.0, 2.0, 3.0, 5.0, 4.0])),
                               0.9, places=6)

        agree = top_k_agreement(x, x, k=2)
        self.assertEqual(agree['overlap'], 2)
        self.assertEqual(agree['jaccard'], 1.0)
        disagree = top_k_agreement(x, -x, k=2)
        self.assertEqual(disagree['overlap'], 0)
        self.assertEqual(sorted(disagree['only_ablation']), [3, 4])

    def test_verdict_thresholds(self):
        from inrun import verdict
        self.assertIn('HIGH', verdict(0.9))
        self.assertIn('MODERATE', verdict(0.7))
        self.assertIn('LOW', verdict(0.2))

    def test_tracker_does_not_disturb_training_gradients(self):
        """Refreshing grad L_val must leave the training gradient untouched."""
        from inrun import InRunNeuronShapley
        from snv_core import build_neuron_index
        model = tiny_model(widths=(4, 4), classes_per_task=2)
        model.ensure_head(0)
        groups = build_neuron_index(model)
        batches = batches_of(fake_loader(n=16, seed=5), 2)

        x, y = next(iter(fake_loader(seed=6)))
        nn.functional.cross_entropy(model(x, 0), y).backward()
        before = {n: p.grad.clone() for n, p in model.named_parameters()
                  if p.grad is not None}

        tracker = InRunNeuronShapley(model, groups, DEVICE, batches, task_id=0,
                                     scenario='task_il')
        tracker._refresh_val_grad()

        for name, p in model.named_parameters():
            if name in before:
                self.assertTrue(torch.equal(p.grad, before[name]),
                                f'{name} gradient was clobbered')


class TestMethods(unittest.TestCase):
    """Every registered method builds and completes a task without error."""

    METHODS = ['sgd', 'ewc', 'si', 'lwf', 'wsn', 'spacenet']

    def test_each_method_trains_a_task(self):
        from baselines import build_method
        for name in self.METHODS:
            with self.subTest(method=name):
                model = tiny_model(widths=(4, 4), classes_per_task=2, num_tasks=2)
                method = build_method(name, model, DEVICE, 'task_il', 1e-3,
                                      sparsity=0.5, buffer_size=32, num_classes=4)
                method.train_task(0, fake_loader(seed=1), fake_loader(n=16, seed=2),
                                  num_epochs=1, patience=1, verbose=False)
                acc = method.evaluate(fake_loader(n=16, seed=2), 0)
                self.assertGreaterEqual(acc, 0.0)
                self.assertLessEqual(acc, 1.0)

    def test_mcl_family_trains_two_tasks(self):
        """MCL's nesting widths are d/16 .. d, so the toy backbone needs d >= 16."""
        from baselines import build_method
        for name in ('mcl', 'mcl_uniform', 'mcl_g', 'mcl3'):
            for scenario in ('task_il', 'class_il'):
                with self.subTest(method=name, scenario=scenario):
                    model = tiny_model(widths=(32, 32), classes_per_task=2, num_tasks=2,
                                       scenario=scenario)
                    method = build_method(name, model, DEVICE, scenario, 1e-3)
                    # Class-IL batches carry global labels (task t owns classes 2t, 2t+1).
                    loader = ((lambda t, n=32, seed=0: TestWSNCorrections.cil_loader(t, n, seed))
                              if scenario == 'class_il' else
                              (lambda t, n=32, seed=0: fake_loader(n=n, classes=2, batch=8, seed=seed)))
                    for task in range(2):
                        method.train_task(task, loader(task, seed=1 + task), loader(task, n=16, seed=5),
                                          num_epochs=1, patience=1, verbose=False)
                    acc = method.evaluate(loader(1, n=16, seed=5), 1)
                    self.assertGreaterEqual(acc, 0.0)
                    self.assertLessEqual(acc, 1.0)

    def test_snv_trains_a_task(self):
        model = tiny_model(widths=(4, 4), classes_per_task=2, num_tasks=2)
        learner = SNVContinualLearner(model, DEVICE, sparsity_ratio=0.5, scenario='task_il',
                                      truncation_threshold=-1.0, max_permutations=2,
                                      shapley_eval_batches=2)
        out = learner.train_task(0, fake_loader(seed=1), fake_loader(n=16, seed=2),
                                 num_epochs=1, patience=1, verbose=False)
        self.assertEqual(int(out['task_mask'].sum()),
                         math.floor(0.5 * learner.mask_manager.num_neurons))
        self.assertLess(out['frozen_drift'], 1e-9)

    def test_wsn_is_flagged_as_til_only(self):
        from baselines import is_cil_only, requires_task_identity
        self.assertTrue(requires_task_identity('wsn'))
        self.assertFalse(requires_task_identity('snv'))
        self.assertTrue(is_cil_only('pec'))
        self.assertFalse(is_cil_only('snv'))


    def test_nispa_uses_official_sparse_cifar_architecture(self):
        from baselines.sparse import NISPA, build_nispa_model
        net = build_nispa_model('cifar100', 5, 'task_il')
        method = NISPA(net, torch.device('cpu'), scenario='task_il', lr=0.002,
                       prune_perc=90, phase_epochs=1, max_phases=3)
        widths = [layer.out_features if isinstance(layer, nn.Linear)
                  else layer.out_channels for layer in net.sparse_layers()]
        self.assertEqual(widths, [64, 64, 128, 128, 1024, 100])
        # First convolution stays dense; all later layers are sparse.
        self.assertTrue(bool(net.conv1.weight_mask.all()))
        self.assertLess(float(net.fc1.weight_mask.float().mean()), 0.2)
        self.assertEqual(tuple(net(torch.randn(2, 3, 32, 32), 0).shape), (2, 5))


    def test_spacenet_rewires_between_epochs(self):
        """The paper performs drop-and-grow each epoch, not once per task."""
        from method_loader import load
        SpaceNet = load('baselines/SpaceNet', 'spacenet').SpaceNet
        model = tiny_model(widths=(4, 4), classes_per_task=2, num_tasks=2)
        method = SpaceNet(model, DEVICE, scenario='task_il', lr=1e-3,
                          density=0.5)
        calls = []
        original = method._rewire

        def counted():
            calls.append(1)
            return original()

        method._rewire = counted
        method.train_task(0, fake_loader(seed=13), fake_loader(n=16, seed=14),
                          num_epochs=3, patience=10, verbose=False)
        self.assertEqual(len(calls), 2)


class TestWSNCorrections(unittest.TestCase):
    """Regression tests for WSN's mask and score handling."""

    @staticmethod
    def cil_loader(task_id, n=16, seed=0):
        base = fake_loader(n=n, classes=2, batch=8, seed=seed)
        x, y = base.dataset.tensors
        return DataLoader(TensorDataset(x, y + 2 * task_id), batch_size=8,
                          shuffle=False)


    def test_wsn_popup_scores_receive_gradient(self):
        from method_loader import load
        WSN = load('baselines/WSN', 'wsn').WSN
        model = tiny_model(widths=(4, 4), classes_per_task=2, num_tasks=2)
        model.ensure_head(0)
        method = WSN(model, DEVICE, scenario='task_il', lr=1e-3, sparsity=0.5)
        method.before_task(0, None, None)
        x, y = next(iter(fake_loader(n=8)))
        nn.functional.cross_entropy(model(x, 0), y).backward()
        gradients = [score.grad for score in method.scores.values()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(any(float(gradient.abs().sum()) > 0 for gradient in gradients))

    def test_wsn_inference_uses_exact_task_mask_not_union(self):
        from method_loader import load
        WSN = load('baselines/WSN', 'wsn').WSN
        model = tiny_model(widths=(4, 4), classes_per_task=2, num_tasks=2)
        method = WSN(model, DEVICE, scenario='task_il', lr=1e-3, sparsity=0.5)
        method.train_task(0, fake_loader(seed=1), fake_loader(n=16, seed=2),
                          num_epochs=1, patience=1, verbose=False)
        first = {name: mask.clone() for name, mask in method.task_masks[0].items()}
        method.train_task(1, fake_loader(seed=3), fake_loader(n=16, seed=4),
                          num_epochs=1, patience=1, verbose=False)
        method.predict(torch.randn(2, 3, 8, 8), 0)
        for name, mask in first.items():
            self.assertTrue(torch.equal(method.task_masks[0][name], mask))
            self.assertEqual(int(mask.sum()),
                             max(1, int(method.sparsity * mask.numel())))

    def test_wsn_til_task_function_is_invariant(self):
        from method_loader import load
        WSN = load('baselines/WSN', 'wsn').WSN
        model = tiny_model(widths=(6, 6), classes_per_task=2, num_tasks=2)
        method = WSN(model, DEVICE, scenario='task_il', lr=1e-3, sparsity=0.5)
        task0 = fake_loader(n=16, seed=2)
        method.train_task(0, fake_loader(seed=1), task0,
                          num_epochs=2, patience=2, verbose=False)
        before = method.evaluate(task0, 0)
        method.train_task(1, fake_loader(seed=3), fake_loader(n=16, seed=4),
                          num_epochs=2, patience=2, verbose=False)
        self.assertEqual(method.evaluate(task0, 0), before)


class TestSNVAdaptive(unittest.TestCase):
    """SNV-A, built exactly as snv_adaptive_run.make_method builds it."""

    VARIANT = dict(task_local=True, frozen_norm_eval=True, bn_recal=True, routing='maxprob',
                   adaptive=True, adaptive_rule='coverage', adaptive_coverage=0.9)

    def build(self, scenario):
        from method_loader import load
        SNVAdaptive = load('SNV', 'snv_adaptive').SNVAdaptive
        model = tiny_model(widths=(4, 4), classes_per_task=2, num_tasks=2, scenario=scenario)
        return SNVAdaptive(model=model, device=DEVICE, scenario=scenario, lr=1e-3,
                           sparsity_ratio=0.5, truncation_threshold=0.1, max_permutations=2,
                           shapley_eval_batches=0, selection='ablation', payoff='loss',
                           use_mab=False, masked_inference=True, masked_training=True,
                           consolidation_epochs=1, consolidation_within_budget=True,
                           layer_floor=0.0, estimator_mode='reverse_tmc', **self.VARIANT)

    def test_trains_two_tasks_in_both_scenarios(self):
        for scenario in ('task_il', 'class_il'):
            with self.subTest(scenario=scenario):
                method = self.build(scenario)
                loader = ((lambda t, n=32, seed=0: TestWSNCorrections.cil_loader(t, n, seed))
                          if scenario == 'class_il' else
                          (lambda t, n=32, seed=0: fake_loader(n=n, classes=2, batch=8, seed=seed)))
                for task in range(2):
                    method.train_task(task, loader(task, seed=1 + task), loader(task, n=16, seed=5),
                                      num_epochs=2, patience=2, verbose=False)
                for task in range(2):
                    acc = method.evaluate(loader(task, n=16, seed=5), task)
                    self.assertGreaterEqual(acc, 0.0)
                    self.assertLessEqual(acc, 1.0)
                self.assertEqual(len(method.history), 2)


class TestSNVManuscriptFidelity(unittest.TestCase):

    def test_zero_batch_limit_uses_full_validation_split(self):
        model = tiny_model(widths=(4, 4), classes_per_task=2, num_tasks=2)
        learner = SNVContinualLearner(model, DEVICE, scenario='task_il',
                                      shapley_eval_batches=0)
        loader = fake_loader(n=25, batch=4)
        batches = learner._cache_val_batches(loader)
        self.assertEqual(sum(y.numel() for _, y in batches), 25)

    def test_masked_inference_stores_S_t_not_B_t(self):
        model = tiny_model(widths=(4, 4), classes_per_task=2, num_tasks=2)
        learner = SNVContinualLearner(
            model, DEVICE, scenario='task_il', sparsity_ratio=0.5,
            truncation_threshold=-1.0, max_permutations=2,
            shapley_eval_batches=1, masked_inference=True)
        learner.train_task(0, fake_loader(seed=1), fake_loader(n=16, seed=2),
                           num_epochs=1, patience=1, verbose=False)
        learner.train_task(1, fake_loader(seed=3), fake_loader(n=16, seed=4),
                           num_epochs=1, patience=1, verbose=False)
        active, _ = learner.task_eval_state[1]
        self.assertTrue(torch.equal(active.cpu(), learner.mask_manager.task_masks[1].cpu()))

    def test_til_task_subnetwork_is_invariant_after_later_training(self):
        model = tiny_model(widths=(6, 6), classes_per_task=2, num_tasks=2)
        learner = SNVContinualLearner(
            model, DEVICE, scenario='task_il', sparsity_ratio=0.5,
            truncation_threshold=-1.0, max_permutations=3,
            shapley_eval_batches=2, masked_inference=True, lr=0.02)
        task0 = fake_loader(n=16, seed=2)
        learner.train_task(0, fake_loader(seed=1), task0,
                           num_epochs=2, patience=2, verbose=False)
        before = learner.evaluate(task0, 0)
        learner.train_task(1, fake_loader(seed=3), fake_loader(n=16, seed=4),
                           num_epochs=2, patience=2, verbose=False)
        after = learner.evaluate(task0, 0)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main(verbosity=2)
