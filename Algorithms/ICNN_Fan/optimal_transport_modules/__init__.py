# Auto-expose all submodules in this package (lazy-loaded)

import importlib
import pkgutil

__all__ = [m.name for m in pkgutil.iter_modules(__path__)]

def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
