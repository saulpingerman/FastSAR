"""Fast spotlight SAR image formation: factorized backprojection with kernels for Cloud TPUs (Pallas), Nvidia GPUs
(CUDA) and x86 CPUs (C++/OpenMP), and polar format with its geometric resampling.  See fastsar.api.form_image."""
import os as _os

# JAX takes 75% of a GPU's memory when it first runs there unless told otherwise; FastSAR's CUDA kernels (CuPy) and
# its JAX programs share the GPU, so JAX allocates as it goes (a setting the user made before importing JAX stands);
# JAX keeps what it has allocated, and the CuPy formers shrink their working sets to what is left
_os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from .api import form_image, available_backends, ImageFormer  # noqa: F401
from . import io, autofocus, stripmap, burst, patches, quality, products  # noqa: F401
from .bp import backproject, plane_points  # noqa: F401
from .exact import ExactFormer  # noqa: F401
from .cphd import form_cphd  # noqa: F401
from .memory import MemoryWarning  # noqa: F401

__version__ = '0.1.1'
