"""
Shapley Neuron Valuation (SNV) for Continual Learning

A memory-free continual learning framework that leverages Shapley values
from cooperative game theory to identify and protect important neurons.

Anonymous submission for ICML 2026.
"""

from .snv_core import (
    NeuronMaskManager,
    MeanActivationComputer,
    ShapleyNeuronEstimator,
    SNVContinualLearner
)

from .models import (
    MLP,
    ResNet18,
    BasicBlock,
    ContinualLearningModel,
    create_model,
    count_parameters,
    count_neurons
)

from .datasets import (
    PermutedMNIST,
    ContinualLearningDataset,
    TinyImageNet,
    ContinualLearningBenchmark,
    get_cifar100_transforms,
    get_tinyimagenet_transforms
)

from .metrics import (
    ContinualLearningMetrics,
    compute_capacity,
    compute_per_task_accuracies
)

__version__ = '1.0.0'
__author__ = 'Anonymous'

__all__ = [
    # Core SNV
    'NeuronMaskManager',
    'MeanActivationComputer', 
    'ShapleyNeuronEstimator',
    'SNVContinualLearner',
    
    # Models
    'MLP',
    'ResNet18',
    'BasicBlock',
    'ContinualLearningModel',
    'create_model',
    'count_parameters',
    'count_neurons',
    
    # Datasets
    'PermutedMNIST',
    'ContinualLearningDataset',
    'TinyImageNet',
    'ContinualLearningBenchmark',
    'get_cifar100_transforms',
    'get_tinyimagenet_transforms',
    
    # Metrics
    'ContinualLearningMetrics',
    'compute_capacity',
    'compute_per_task_accuracies'
]
