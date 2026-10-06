"""Fast spotlight SAR image formation: factorized backprojection with kernels for Cloud TPUs (Pallas), Nvidia GPUs
(CUDA) and x86 CPUs (C++/OpenMP), and polar format with its geometric resampling.  See fastsar.api.form_image."""
from .api import form_image, available_backends, ImageFormer  # noqa: F401
from . import io, autofocus, stripmap, burst, patches, quality, products  # noqa: F401
from .bp import backproject, plane_points  # noqa: F401

__version__ = '0.1.0'
