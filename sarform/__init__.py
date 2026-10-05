"""Fast spotlight SAR image formation: factorized backprojection with kernels for Cloud TPUs (Pallas), Nvidia GPUs
(CUDA) and x86 CPUs (C++/OpenMP), and polar format with its geometric resampling.  See sarform.api.form_image."""
from .api import form_image, available_backends  # noqa: F401
from . import io  # noqa: F401

__version__ = '0.1.0'
