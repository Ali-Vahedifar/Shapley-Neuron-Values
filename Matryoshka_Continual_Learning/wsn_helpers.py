"""Shared helpers for the sparse methods (moved out of baselines_sparse.py)."""
from importlib.util import module_from_spec, spec_from_file_location
import os
_p = os.path.join(os.path.dirname(__file__), "WSN", "wsn.py")
_s = spec_from_file_location("_wsn_impl", _p)
_m = module_from_spec(_s); _s.loader.exec_module(_m)
_maskable_modules = _m._maskable_modules
_WeightMask = _m._WeightMask
