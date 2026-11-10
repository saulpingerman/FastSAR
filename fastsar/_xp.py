"""Where an array lives. The stages that follow image formation (planes, autofocus, weighting, remap) take the
image as a NumPy array (host), a CuPy array (an Nvidia GPU) or a JAX array (a TPU, or whatever device JAX runs on)
and return their result in the same place, so that a chain of them runs on the device that formed the image and the
image crosses to the host once, at the end.

    xp = _xp.module(img)                 # numpy, cupy or jax.numpy
    img = _xp.device(img, 'cuda')        # to the device of a backend ('cpu', 'cuda', 'jax', 'tpu'; 'auto' and
    out = _xp.host(img)                  # 'metal' keep these stages on the host), and back
"""
import numpy as np

__all__ = ['kind', 'module', 'fft', 'host', 'device']


def kind(a):
    """'numpy', 'cupy' or 'jax': the library that holds array a (anything else counts as NumPy's)."""
    mod = type(a).__module__
    if mod.startswith('cupy'):
        return 'cupy'
    if mod.startswith('jax'):
        return 'jax'
    return 'numpy'


def module(a):
    """The array namespace of a: numpy, cupy or jax.numpy."""
    k = kind(a)
    if k == 'cupy':
        import cupy
        return cupy
    if k == 'jax':
        import jax.numpy as jnp
        return jnp
    return np


def fft(a):
    """The FFT module for a, with scipy.fft's functions: scipy.fft (give workers=-1 for all cores), cupyx.scipy.fft
    or jax.numpy.fft."""
    k = kind(a)
    if k == 'cupy':
        import cupyx.scipy.fft as f
        return f
    if k == 'jax':
        import jax.numpy.fft as f
        return f
    import scipy.fft as f
    return f


def host(a):
    """a as a NumPy array (a copy from the device, or a itself)."""
    if kind(a) == 'cupy':
        return a.get()
    return np.asarray(a)


def device(a, backend):
    """a where the stages after formation run for a backend: a CuPy array for 'cuda', a JAX array for 'jax' and
    'tpu', a NumPy array for the others ('cpu'; 'metal', whose stages after formation run on the host)."""
    if backend == 'cuda':
        import cupy
        return cupy.asarray(a)
    if backend in ('jax', 'tpu'):
        import jax.numpy as jnp
        return jnp.asarray(a)
    return host(a)
