"""Sparse / architecture baselines.

The implementations live one per method folder; this module re-exports them so
the registry and the audited GTEP worker share a single import path.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method_loader import load

_wsn = load('baselines/WSN', 'wsn')
_maskable_modules = _wsn._maskable_modules
_WeightMask = _wsn._WeightMask
WSN = _wsn.WSN

_spacenet = load('baselines/SpaceNet', 'spacenet')
SpaceNet = _spacenet.SpaceNet

_nispa = load('baselines/NISPA', 'nispa')
NISPA = _nispa.NISPA
NISPANet = _nispa.NISPANet
build_nispa_model = _nispa.build_nispa_model

_pec = load('baselines/PEC', 'pec')
PEC = _pec.PEC

__all__ = ['_maskable_modules', '_WeightMask', 'WSN', 'SpaceNet',
           'NISPA', 'NISPANet', 'build_nispa_model', 'PEC']
