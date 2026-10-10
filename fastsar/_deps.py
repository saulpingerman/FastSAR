"""Optional dependencies: an ImportError that names the extra installing the missing package."""
import importlib

EXTRA = {'sarpy': 'io', 'rasterio': 'geo', 'cupy': 'cuda'}


def require(module):
    """Import the optional dependency `module` (its top-level name) or raise an ImportError saying which extra of
    fastsar installs it."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        extra = EXTRA.get(module, module)
        raise ImportError(f'{module} is not installed; it is an optional dependency of fastsar: '
                          f'pip install "fastsar[{extra}]"') from e
