"""SNV core -- implementation moved to SNV/snv_core.py.

Re-exported here so ``from snv_core import ...`` keeps working for train.py,
baselines/__init__.py, models.py and tests/test_snv.py.
"""

from method_loader import load

_snv = load("SNV", "snv_core")
NeuronGroup = _snv.NeuronGroup
build_neuron_index = _snv.build_neuron_index
NeuronMaskManager = _snv.NeuronMaskManager
MeanActivationComputer = _snv.MeanActivationComputer
ShapleyNeuronEstimator = _snv.ShapleyNeuronEstimator
install_subnetwork_mask = _snv.install_subnetwork_mask
SNVContinualLearner = _snv.SNVContinualLearner
subnetwork_weight_fraction = _snv.subnetwork_weight_fraction
_owned_weights = _snv._owned_weights
_total_weights = _snv._total_weights

__all__ = ['NeuronGroup', 'build_neuron_index', 'NeuronMaskManager', 'MeanActivationComputer', 'ShapleyNeuronEstimator', 'install_subnetwork_mask', 'SNVContinualLearner', 'subnetwork_weight_fraction']
