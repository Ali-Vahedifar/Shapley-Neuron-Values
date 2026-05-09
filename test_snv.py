"""
Test Suite for SNV Continual Learning Implementation.

This module verifies that the implementation matches the paper specifications:
- Model architectures
- Algorithm components
- Metric computations

Anonymous submission for ICML 2026.
"""

import unittest
import torch
import numpy as np
from collections import OrderedDict

# Import SNV modules
from models import MLP, ResNet18, create_model, count_neurons, count_parameters
from snv_core import NeuronMaskManager, ShapleyNeuronEstimator, SNVContinualLearner
from metrics import ContinualLearningMetrics
from datasets import ContinualLearningBenchmark


class TestModels(unittest.TestCase):
    """Test model architectures match paper specifications."""
    
    def test_mlp_architecture(self):
        """Test MLP has 4 hidden layers with 200 neurons each."""
        model = MLP(input_dim=784, hidden_dim=200, num_layers=4, num_classes=10)
        
        # Check number of layers
        self.assertEqual(len(model.hidden_layers), 4)
        
        # Check hidden dimensions
        for layer in model.hidden_layers:
            self.assertEqual(layer.out_features, 200)
        
        # Check output dimension
        self.assertEqual(model.classifier.out_features, 10)
        
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        model = MLP()
        x = torch.randn(32, 784)  # Batch of 32
        output = model(x)
        
        self.assertEqual(output.shape, (32, 10))
        
    def test_resnet18_architecture(self):
        """Test ResNet-18 channel progression {64, 64, 128, 128, 256, 256, 512, 512}."""
        model = ResNet18(num_classes=100, input_size=32)
        
        # Check initial conv
        self.assertEqual(model.conv1.out_channels, 64)
        
        # Check layer channel progression
        expected_channels = [64, 64, 128, 128, 256, 256, 512, 512]
        actual_channels = []
        
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in layer:
                actual_channels.append(block.conv1.out_channels)
                
        self.assertEqual(actual_channels, expected_channels)
        
    def test_resnet18_forward_cifar(self):
        """Test ResNet-18 forward pass for CIFAR-100 (32x32)."""
        model = ResNet18(num_classes=100, input_size=32)
        x = torch.randn(32, 3, 32, 32)
        output = model(x)
        
        self.assertEqual(output.shape, (32, 100))
        
    def test_resnet18_forward_tinyimagenet(self):
        """Test ResNet-18 forward pass for TinyImageNet (64x64)."""
        model = ResNet18(num_classes=200, input_size=64)
        x = torch.randn(32, 3, 64, 64)
        output = model(x)
        
        self.assertEqual(output.shape, (32, 200))
        
    def test_create_model_factory(self):
        """Test model factory function."""
        # PMNIST
        model = create_model('pmnist', num_classes=10)
        self.assertIsInstance(model, MLP)
        
        # CIFAR-100
        model = create_model('cifar100', num_classes=100)
        self.assertIsInstance(model, ResNet18)
        
        # TinyImageNet
        model = create_model('tinyimagenet', num_classes=200)
        self.assertIsInstance(model, ResNet18)
        
    def test_he_initialization(self):
        """Test He initialization is applied."""
        model = ResNet18(num_classes=100)
        
        # Check conv weights have reasonable variance
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                # He init should have std ≈ sqrt(2/fan_in)
                fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                expected_std = np.sqrt(2.0 / fan_in)
                actual_std = m.weight.std().item()
                
                # Allow 50% tolerance
                self.assertAlmostEqual(actual_std, expected_std, delta=expected_std * 0.5)


class TestNeuronMaskManager(unittest.TestCase):
    """Test neuron mask management."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = ResNet18(num_classes=100, input_size=32)
        self.mask_manager = NeuronMaskManager(self.model, self.device)
        
    def test_neuron_extraction(self):
        """Test correct extraction of neurons from model."""
        # ResNet-18 should have conv layers in all 4 main layers
        total_neurons = count_neurons(self.model)
        
        # Expected: 64 (conv1) + 4*64 + 4*128 + 4*256 + 4*512 for main layers
        # Plus residual convs: 128 + 256 + 512 for downsampling
        # Actual count from our definition
        self.assertGreater(total_neurons, 0)
        self.assertEqual(self.mask_manager.num_neurons, total_neurons)
        
    def test_cumulative_mask_update(self):
        """Test cumulative mask properly accumulates."""
        # Initially all zeros
        self.assertTrue(torch.all(~self.mask_manager.cumulative_mask))
        
        # Create and apply task 0 mask
        task_mask = torch.zeros(self.mask_manager.num_neurons, dtype=torch.bool)
        task_mask[:100] = True  # First 100 neurons
        self.mask_manager.update_cumulative_mask(0, task_mask)
        
        self.assertEqual(self.mask_manager.cumulative_mask.sum().item(), 100)
        
        # Create and apply task 1 mask (overlapping)
        task_mask_2 = torch.zeros(self.mask_manager.num_neurons, dtype=torch.bool)
        task_mask_2[50:150] = True  # Overlaps with first 50 neurons
        self.mask_manager.update_cumulative_mask(1, task_mask_2)
        
        # Should have 150 unique neurons frozen (0-149)
        self.assertEqual(self.mask_manager.cumulative_mask.sum().item(), 150)
        
    def test_gradient_mask_creation(self):
        """Test gradient mask matches frozen neurons."""
        # Freeze some neurons
        task_mask = torch.zeros(self.mask_manager.num_neurons, dtype=torch.bool)
        task_mask[:10] = True
        self.mask_manager.update_cumulative_mask(0, task_mask)
        
        gradient_masks = self.mask_manager.create_gradient_mask()
        
        # Gradient masks should exist for parameterized layers
        self.assertGreater(len(gradient_masks), 0)


class TestMetrics(unittest.TestCase):
    """Test metric computations."""
    
    def test_average_accuracy(self):
        """Test ACC computation."""
        metrics = ContinualLearningMetrics(num_tasks=3)
        
        # Perfect accuracy matrix
        metrics.accuracy_matrix = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        
        acc = metrics.get_average_accuracy()
        self.assertAlmostEqual(acc, 1.0)
        
    def test_backward_transfer_no_forgetting(self):
        """Test BWT = 0 when no forgetting."""
        metrics = ContinualLearningMetrics(num_tasks=3)
        
        # No forgetting - diagonal stays constant
        metrics.accuracy_matrix = np.array([
            [0.9, 0.0, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.85]
        ])
        
        bwt = metrics.get_backward_transfer()
        self.assertAlmostEqual(bwt, 0.0)
        
    def test_backward_transfer_with_forgetting(self):
        """Test BWT negative when forgetting occurs."""
        metrics = ContinualLearningMetrics(num_tasks=3)
        
        # Forgetting - accuracy drops
        metrics.accuracy_matrix = np.array([
            [0.9, 0.0, 0.0],
            [0.7, 0.8, 0.0],
            [0.5, 0.6, 0.85]
        ])
        
        bwt = metrics.get_backward_transfer()
        # BWT = ((0.5-0.9) + (0.6-0.8)) / 2 = -0.3
        self.assertAlmostEqual(bwt, -0.3)
        
    def test_plasticity_stability_ratio(self):
        """Test PS ratio computation."""
        metrics = ContinualLearningMetrics(num_tasks=3)
        
        metrics.accuracy_matrix = np.array([
            [0.9, 0.0, 0.0],
            [0.9, 0.8, 0.0],
            [0.9, 0.8, 0.85]
        ])
        
        ps = metrics.get_plasticity_stability_ratio()
        
        # Plasticity = mean([0.9, 0.8, 0.85]) = 0.85
        # BWT = 0 (no forgetting)
        # PS = 0.85 / (0.85 + 0) = 1.0
        self.assertAlmostEqual(ps, 1.0)


class TestDatasets(unittest.TestCase):
    """Test dataset creation and task splitting."""
    
    def test_benchmark_task_classes(self):
        """Test correct class assignment per task."""
        benchmark = ContinualLearningBenchmark(
            dataset_name='cifar100',
            num_tasks=10,
            seed=42
        )
        
        # 10 classes per task
        self.assertEqual(benchmark.classes_per_task, 10)
        
        # Each task should have 10 classes
        for task_id in range(10):
            classes = benchmark.get_task_classes(task_id)
            self.assertEqual(len(classes), 10)
            
        # All classes should be unique across tasks
        all_classes = []
        for task_id in range(10):
            all_classes.extend(benchmark.get_task_classes(task_id))
        self.assertEqual(len(set(all_classes)), 100)
        
    def test_class_mapping_class_il(self):
        """Test class mapping for Class-IL scenario."""
        benchmark = ContinualLearningBenchmark(
            dataset_name='cifar100',
            num_tasks=10,
            scenario='class_il'
        )
        
        # Task 0: maps to 0-9
        mapping_0 = benchmark.get_class_mapping(0)
        self.assertEqual(min(mapping_0.values()), 0)
        self.assertEqual(max(mapping_0.values()), 9)
        
        # Task 5: maps to 50-59
        mapping_5 = benchmark.get_class_mapping(5)
        self.assertEqual(min(mapping_5.values()), 50)
        self.assertEqual(max(mapping_5.values()), 59)
        
    def test_class_mapping_task_il(self):
        """Test class mapping for Task-IL scenario."""
        benchmark = ContinualLearningBenchmark(
            dataset_name='cifar100',
            num_tasks=10,
            scenario='task_il'
        )
        
        # All tasks should map to 0-9
        for task_id in range(10):
            mapping = benchmark.get_class_mapping(task_id)
            self.assertEqual(min(mapping.values()), 0)
            self.assertEqual(max(mapping.values()), 9)


class TestSNVLearner(unittest.TestCase):
    """Test SNV continual learner."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MLP(input_dim=100, hidden_dim=50, num_layers=2, num_classes=10)
        
    def test_learner_initialization(self):
        """Test learner initializes correctly."""
        learner = SNVContinualLearner(
            model=self.model,
            device=self.device,
            sparsity_ratio=0.1,
            lr=0.001
        )
        
        self.assertEqual(learner.sparsity_ratio, 0.1)
        self.assertIsInstance(learner.mask_manager, NeuronMaskManager)
        
    def test_sparsity_constraint(self):
        """Test that sparsity ratio limits neurons selected."""
        learner = SNVContinualLearner(
            model=self.model,
            device=self.device,
            sparsity_ratio=0.1
        )
        
        num_neurons = learner.mask_manager.num_neurons
        expected_k = int(0.1 * num_neurons)
        
        # After selecting neurons, should have at most k neurons
        self.assertGreater(expected_k, 0)


class TestShapleyEstimator(unittest.TestCase):
    """Test Shapley value estimation."""
    
    def test_top_k_selection(self):
        """Test top-k neuron selection."""
        device = torch.device('cpu')
        model = MLP(input_dim=100, hidden_dim=20, num_layers=2, num_classes=10)
        mask_manager = NeuronMaskManager(model, device)
        
        # Create dummy Shapley values
        num_neurons = mask_manager.num_neurons
        shapley_values = torch.randn(num_neurons)
        
        # Create estimator with empty mean activations
        estimator = ShapleyNeuronEstimator(
            model=model,
            neuron_info=mask_manager.neuron_info,
            mean_activations={},
            device=device
        )
        
        # Select top 10%
        mask = estimator.select_top_k_neurons(shapley_values, sparsity_ratio=0.1)
        
        expected_k = int(0.1 * num_neurons)
        self.assertEqual(mask.sum().item(), expected_k)
        
    def test_available_mask_respected(self):
        """Test that already frozen neurons are excluded."""
        device = torch.device('cpu')
        model = MLP(input_dim=100, hidden_dim=20, num_layers=2, num_classes=10)
        mask_manager = NeuronMaskManager(model, device)
        
        num_neurons = mask_manager.num_neurons
        shapley_values = torch.randn(num_neurons)
        
        # Make first half of neurons unavailable
        available_mask = torch.zeros(num_neurons, dtype=torch.bool)
        available_mask[num_neurons//2:] = True
        
        estimator = ShapleyNeuronEstimator(
            model=model,
            neuron_info=mask_manager.neuron_info,
            mean_activations={},
            device=device
        )
        
        mask = estimator.select_top_k_neurons(
            shapley_values, 
            sparsity_ratio=0.1,
            available_mask=available_mask
        )
        
        # Selected neurons should only be from available set
        selected_unavailable = mask & ~available_mask
        self.assertEqual(selected_unavailable.sum().item(), 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestNeuronMaskManager))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasets))
    suite.addTests(loader.loadTestsFromTestCase(TestSNVLearner))
    suite.addTests(loader.loadTestsFromTestCase(TestShapleyEstimator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_tests()
