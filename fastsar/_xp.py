"""Where an array lives. The stages that follow image formation (planes, autofocus, weighting, remap) take the
image as a NumPy array (host), a CuPy array (an Nvidia GPU) or a JAX array (a TPU, or whatever device JAX runs on)
and return their result in the same place, so that a chain of them runs on the device that formed the image and the
image crosses to the host once, at the end.

    xp = _xp.module(img)                 # numpy, cupy or jax.numpy
    img = _xp.device(img, 'cuda')        # to the device of a backend ('cpu', 'cuda', 'jax', 'tpu'; 'auto' and
    out = _xp.host(img)                  # 'metal' keep these stages on the host), and back
"""
import numpy as np

__all__ = ['kind', 'module', 'fft', 'host', 'device', 'jax_block']


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


def jax_block():
    """Bytes of temporary arrays a stage after formation may hold on JAX's device at a time, beside the image and
    its result: FASTSAR_JAX_BLOCK_MB if set, else a sixteenth of the device's memory (1 GB where the device does
    not report it; a 16 GB TPU reports it), at most 2 GB. The stages cut the image into blocks of rows, columns or
    pixels that fit, each through one compiled program per block shape."""
    import os
    mb = os.environ.get('FASTSAR_JAX_BLOCK_MB', '').strip()
    if mb:
        if not float(mb) > 0:
            raise ValueError(f'FASTSAR_JAX_BLOCK_MB must be positive, got {mb!r}')
        return int(float(mb) * (1 << 20))
    import jax
    try:
        limit = (jax.devices()[0].memory_stats() or {}).get('bytes_limit')
    except Exception:                                              # a device without statistics
        limit = None
    return int(min(2 << 30, limit // 16)) if limit else 1 << 30


def host(a):
    """a as a NumPy array (a copy from the device, or a itself). A complex array on a TPU crosses as its two real
    planes, in blocks of rows of at most jax_block() bytes: a TPU hands a complex array to the host some thirty
    times slower than a real one of the same size."""
    k = kind(a)
    if k == 'cupy':
        return a.get()
    if k == 'jax' and a.ndim and 'complex' in str(a.dtype) and _on_tpu(a):
        import jax.numpy as jnp
        out = np.empty(a.shape, a.dtype)
        v = out.view(out.real.dtype).reshape(a.shape + (2,))
        step = int(max(1, min(a.shape[0], jax_block() // max(a[:1].nbytes, 1))))
        for i in range(0, a.shape[0], step):
            v[i:i + step, ..., 0], v[i:i + step, ..., 1] = np.asarray(jnp.real(a[i:i + step])), np.asarray(jnp.imag(a[i:i + step]))
        return out
    return np.asarray(a)


def _on_tpu(a):
    """Whether the JAX array a lives on a TPU."""
    try:
        return next(iter(a.devices())).platform == 'tpu'
    except (AttributeError, StopIteration):                        # a tracer
        return False


def device(a, backend):
    """a where the stages after formation run for a backend: a CuPy array for 'cuda', a JAX array for 'jax' and
    'tpu', a NumPy array for the others ('cpu'; 'metal', whose stages after formation run on the host)."""
    if backend == 'cuda':
        import cupy
        return cupy.asarray(a)
    if backend in ('jax', 'tpu'):
        import jax.numpy as jnp
        # a private copy of a NumPy array: on its CPU device JAX may keep the host buffer, and the caller may go on
        # to write it
        return jnp.asarray(np.array(a) if kind(a) == 'numpy' else a)
    return host(a)
