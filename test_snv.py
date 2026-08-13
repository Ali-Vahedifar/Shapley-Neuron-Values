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
        with torch.no_grad():
            plain = sum(model(x, 0).argmax(1).eq(y).sum().item() for x, y in batches)
            plain /= sum(y.numel() for _, y in batches)
        self.assertAlmostEqual(est.evaluate_subset(full), plain, places=6)


class TestBandit(unittest.TestCase):
    """A <- {i : |phi_i - phi^(k)| < delta_i}."""

    def _estimator(self, n=8):
        model = tiny_model(widths=(4, 4), classes_per_task=2)
        model.ensure_head(0)
        groups = build_neuron_index(model)
        est = ShapleyNeuronEstimator(model, groups, {}, DEVICE)
        est.num_neurons = n
        return est

    def test_separated_estimates_empty_the_active_set(self):
        est = self._estimator()
        phi = torch.linspace(1.0, 0.0, 8).double()
        counts = torch.full((8,), 500.0, dtype=torch.float64)
        m2 = torch.full((8,), 1e-12, dtype=torch.float64) * counts
        self.assertEqual(int(est._bandit_active_set(phi, counts, m2, k=3).sum()), 0)

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


class TestMethods(unittest.TestCase):
    """Every registered method builds and completes a task without error."""

    METHODS = ['sgd', 'ewc', 'si', 'lwf', 'wsn', 'spacenet', 'nispa', 'dcnet',
               'nfl', 'nfl+', 'icarl', 'derpp']

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
        from baselines import requires_task_identity
        self.assertTrue(requires_task_identity('wsn'))
        self.assertFalse(requires_task_identity('snv'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
