"""Sparse / architecture baselines.

The implementations live one per method folder; this module re-exports them so
the registry and the audited GTEP worker share a single import path.
"""

from method_loader import load

_wsn = load('WSN', 'wsn')
_maskable_modules = _wsn._maskable_modules
_WeightMask = _wsn._WeightMask
WSN = _wsn.WSN

_spacenet = load('SpaceNet', 'spacenet')
SpaceNet = _spacenet.SpaceNet

_nispa = load('NISPA', 'nispa')
NISPA = _nispa.NISPA
NISPANet = _nispa.NISPANet
build_nispa_model = _nispa.build_nispa_model

_pec = load('PEC', 'pec')
PEC = _pec.PEC

__all__ = ['_maskable_modules', '_WeightMask', 'WSN', 'SpaceNet',
           'NISPA', 'NISPANet', 'build_nispa_model', 'PEC']
