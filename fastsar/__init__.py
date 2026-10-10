"""SAR image formation on x86 CPUs, Nvidia GPUs and Cloud TPUs: factorized backprojection with a kernel for each
(C++/OpenMP, CUDA, Pallas) for spotlight, stripmap, sliding spotlight and burst collections, exact backprojection,
polar format, and the chain from a CPHD file to geolocated products. Start with form_cphd or form_image."""
__version__ = '0.1.2'

import os as _os

# JAX takes 75% of a GPU's memory when it first runs there unless told otherwise; FastSAR's CUDA kernels (CuPy) and
# its JAX programs share the GPU, so JAX allocates as it goes (a setting the user made before importing JAX stands);
# JAX keeps what it has allocated, and the CuPy formers shrink their working sets to what is left
_os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from .api import form_image, available_backends, ImageFormer  # noqa: F401
from . import io, autofocus, stripmap, burst, patches, quality, products, sim  # noqa: F401
from .bp import backproject, plane_points  # noqa: F401
from .exact import ExactFormer  # noqa: F401
from .cphd import form_cphd  # noqa: F401
from .memory import MemoryWarning  # noqa: F401

__all__ = ['form_image', 'available_backends', 'ImageFormer', 'ExactFormer', 'form_cphd', 'backproject', 'plane_points',
           'MemoryWarning', 'io', 'autofocus', 'stripmap', 'burst', 'patches', 'quality', 'products', 'sim']
