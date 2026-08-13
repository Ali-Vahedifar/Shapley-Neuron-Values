"""
Shapley Neuron Valuation (SNV) for Continual Learning.

Reference implementation for
"Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?"

The modules are written to run as scripts (``python train.py ...``) as well as
to be imported as a package, so the package directory is placed on ``sys.path``
before the submodules are loaded and every internal import is absolute.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snv_core import (                                              # noqa: E402
    MeanActivationComputer,
    NeuronGroup,
    NeuronMaskManager,
    SNVContinualLearner,
    ShapleyNeuronEstimator,
    build_neuron_index,
)
from models import (                                                # noqa: E402
    MLP,
    BasicBlock,
    ContinualLearningModel,
    ResNet18,
    count_neurons,
    count_parameters,
    create_backbone,
    create_model,
)
from datasets import (                                              # noqa: E402
    ContinualLearningBenchmark,
    TinyImageNet,
    build_transforms,
)
from metrics import (                                               # noqa: E402
    ContinualLearningMetrics,
    compute_capacity,
    compute_per_task_accuracies,
)
from cl_base import ContinualMethod, Reservoir                      # noqa: E402
from baselines import (                                             # noqa: E402
    ALL_METHODS,
    BUFFER_FREE,
    MEMORY_BASED,
    build_method,
    default_buffer_size,
    requires_task_identity,
)

__version__ = '2.0.0'

__all__ = [
    # core
    'SNVContinualLearner', 'ShapleyNeuronEstimator', 'NeuronMaskManager',
    'MeanActivationComputer', 'NeuronGroup', 'build_neuron_index',
    # models
    'MLP', 'ResNet18', 'BasicBlock', 'ContinualLearningModel',
    'create_model', 'create_backbone', 'count_parameters', 'count_neurons',
    # data
    'ContinualLearningBenchmark', 'TinyImageNet', 'build_transforms',
    # metrics
    'ContinualLearningMetrics', 'compute_capacity', 'compute_per_task_accuracies',
    # methods
    'ContinualMethod', 'Reservoir', 'build_method', 'default_buffer_size',
    'requires_task_identity', 'ALL_METHODS', 'BUFFER_FREE', 'MEMORY_BASED',
]
