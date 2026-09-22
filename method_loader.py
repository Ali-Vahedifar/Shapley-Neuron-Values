"""Import a method implementation from its own folder.

Each method lives in ``<Folder>/<module>.py``.  Method folders are not Python
packages, so modules are loaded by
file path rather than by the normal import machinery.
"""

import os
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def load(folder: str, module: str):
    """Load ``<folder>/<module>.py`` once and cache it."""
    key = (folder, module)
    if key in _CACHE:
        return _CACHE[key]
    path = os.path.join(_ROOT, folder, module + '.py')
    if not os.path.exists(path):
        raise ImportError(f'no implementation at {path}')
    # ``torch.save`` verifies that a module-valued object's class can be
    # imported by its fully-qualified name.  Registering only
    # ``methods.<module>`` in sys.modules is insufficient because importing it
    # first resolves the parent package.  The method folders are intentionally
    # not Python packages , so provide one
    # stable synthetic namespace for all file-loaded implementations.
    package = sys.modules.get('methods')
    if package is None:
        package = types.ModuleType('methods')
        package.__path__ = [_ROOT]
        package.__package__ = 'methods'
        sys.modules['methods'] = package

    spec = spec_from_file_location(f'methods.{module}', path)
    mod = module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    setattr(package, module, mod)
    _CACHE[key] = mod
    return mod
