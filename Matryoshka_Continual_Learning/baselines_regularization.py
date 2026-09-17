"""Regularisation baselines.

The implementations now live one per method folder; this module re-exports
them so existing imports keep working.
"""

from method_loader import load

_sgd = load('SGD', 'sgd')
SGDBaseline = _sgd.SGDBaseline

_ewc = load('EWC', 'ewc')
EWC = _ewc.EWC

_si = load('SI', 'si')
SI = _si.SI

_lwf = load('LwF', 'lwf')
LwF = _lwf.LwF

__all__ = ['SGDBaseline', 'EWC', 'SI', 'LwF']
