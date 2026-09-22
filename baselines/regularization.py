"""Regularisation baselines.

The implementations now live one per method folder; this module re-exports
them so existing imports keep working.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method_loader import load

_sgd = load('baselines/SGD', 'sgd')
SGDBaseline = _sgd.SGDBaseline

_ewc = load('baselines/EWC', 'ewc')
EWC = _ewc.EWC

_si = load('baselines/SI', 'si')
SI = _si.SI

_lwf = load('baselines/LwF', 'lwf')
LwF = _lwf.LwF

__all__ = ['SGDBaseline', 'EWC', 'SI', 'LwF']
