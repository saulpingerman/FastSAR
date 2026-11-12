"""Where an array lives. The stages that follow image formation (planes, autofocus, weighting, remap) take the
image as a NumPy array (host), a CuPy array (an Nvidia GPU) or a JAX array (a TPU, or whatever device JAX runs on)
and return their result in the same place, so that a chain of them runs on the device that formed the image and the
image crosses to the host once, at the end.

    xp = _xp.module(img)                 # numpy, cupy or jax.numpy
    img = _xp.device(img, 'cuda')        # to the device of a backend ('cpu', 'cuda', 'jax', 'tpu'; 'auto' and
    out = _xp.host(img)                  # 'metal' keep these stages on the host), and back
"""
import numpy as np

__all__ = ['kind', 'module', 'fft', 'host', 'device', 'jax_block', 'jax_budget', 'cuda_budget', 'cuda_block']


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


def _jax_env():
    """FASTSAR_JAX_BLOCK_MB in bytes, or None when it is not set."""
    import os
    mb = os.environ.get('FASTSAR_JAX_BLOCK_MB', '').strip()
    if not mb:
        return None
    try:
        if not float(mb) > 0:
            raise ValueError
    except ValueError:
        raise ValueError(f'FASTSAR_JAX_BLOCK_MB must be a positive number of megabytes, got {mb!r}') from None
    return int(float(mb) * (1 << 20))


def _jax_stats():
    import jax
    try:
        return jax.devices()[0].memory_stats() or {}
    except Exception:                                              # a device without statistics
        return {}


def jax_budget():
    """Bytes of JAX's device new arrays may take now, or None where the device does not say (JAX on a CPU):
    FASTSAR_JAX_BLOCK_MB megabytes if set (for a test, or to hold FastSAR to part of a device), else nine tenths of
    what is not in use on the device (a TPU v5e has about 16 GB, a v6e about 31 GB). The stages after formation size
    their blocks from it, and process_cphd decides with it which images of a product stay on the host
    (memory.cuda_plan)."""
    env = _jax_env()
    if env is not None:
        return env
    st = _jax_stats()
    return int(0.9 * (st['bytes_limit'] - st.get('bytes_in_use', 0))) if st.get('bytes_limit') else None


def jax_block():
    """Bytes of temporary arrays a stage after formation may hold on JAX's device at a time, beside its input and
    its result: a quarter of FASTSAR_JAX_BLOCK_MB if set, else a sixteenth of the device's memory (1 GB where the
    device does not report it), at most 2 GB. The stages cut the image into blocks of rows, columns or pixels that
    fit, each through one compiled program per block shape."""
    env = _jax_env()
    if env is not None:
        return int(min(2 << 30, max(env // 4, 1)))
    limit = _jax_stats().get('bytes_limit')
    return int(min(2 << 30, limit // 16)) if limit else 1 << 30


def jax_rows(n, row_bytes, what):
    """Rows (of n) a block may hold at row_bytes of device memory a row, within jax_block(). MemoryError when one
    row does not fit in jax_budget(); what: the stage, for the message."""
    nb = jax_block() // max(int(row_bytes), 1)
    if nb < 1:
        free = jax_budget()
        if free is not None and row_bytes > free:
            raise MemoryError(f'{what}: one line takes {row_bytes / 1e6:.1f} MB of device memory and '
                              f'{free / 1e6:.1f} MB are available (FASTSAR_JAX_BLOCK_MB, or the device itself)')
        nb = 1
    return int(min(n, nb))


def to_host(out, index, block):
    """A JAX block written into the NumPy array out at index (a tuple of slices): a complex block as its two real
    planes, which a TPU hands over some thirty times faster than the complex array."""
    if 'complex' in str(block.dtype):
        import jax.numpy as jnp
        out.real[index], out.imag[index] = np.asarray(jnp.real(block)), np.asarray(jnp.imag(block))
    else:
        out[index] = np.asarray(block)


def cuda_budget():
    """Bytes of the CUDA device new arrays may take now: FASTSAR_CUDA_BLOCK_MB megabytes if set (for a test, or to
    hold FastSAR to part of a shared device), else nine tenths of what is free on the device, counting the blocks
    CuPy's pool holds unused (memory.cuda_free). The stages after formation size their blocks from it, and
    process_cphd decides with it which images of a product stay on the host (memory.cuda_plan)."""
    import os
    mb = os.environ.get('FASTSAR_CUDA_BLOCK_MB', '').strip()
    if mb:
        try:
            if not float(mb) > 0:
                raise ValueError
        except ValueError:
            raise ValueError(f'FASTSAR_CUDA_BLOCK_MB must be a positive number of megabytes, got {mb!r}') from None
        return int(float(mb) * (1 << 20))
    from .memory import cuda_free
    return int(0.9 * cuda_free())


def cuda_block():
    """Bytes of temporary arrays a stage after formation may hold on the CUDA device at a time, beside its input
    and its result: a quarter of cuda_budget(), at most 2 GB."""
    return int(min(2 << 30, cuda_budget() // 4))


def cuda_rows(n, row_bytes, what):
    """Rows (of n) a block may hold at row_bytes of device memory a row, within cuda_block(). MemoryError when
    one row does not fit in cuda_budget(); what: the stage, for the message."""
    nb = cuda_block() // max(int(row_bytes), 1)
    if nb < 1:
        free = cuda_budget()
        if row_bytes > free:
            raise MemoryError(f'{what}: one line takes {row_bytes / 1e6:.1f} MB of device memory and '
                              f'{free / 1e6:.1f} MB are free (FASTSAR_CUDA_BLOCK_MB, or the device itself)')
        nb = 1
    return int(min(n, nb))


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
            blk = a[i:i + step]
            v[i:i + step, ..., 0], v[i:i + step, ..., 1] = np.asarray(jnp.real(blk)), np.asarray(jnp.imag(blk))
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
        if kind(a) != 'numpy':
            return cupy.asarray(a)
        # a copy that is complete when this returns. cupy.asarray reads a page-locked source (the image a cuda former
        # returns) after it returns, on the current stream, and the caller may go on to write that image. The stream
        # is waited for first: the new array may be a block of the pool that a queued kernel still writes, and a
        # synchronous copy does not queue behind the work of a stream other than the default one
        cupy.cuda.get_current_stream().synchronize()
        return cupy.asarray(a, blocking=True)
    if backend in ('jax', 'tpu'):
        import jax.numpy as jnp
        # a private copy of a NumPy array: on its CPU device JAX may keep the host buffer, and the caller may go on
        # to write it
        return jnp.asarray(np.array(a) if kind(a) == 'numpy' else a)
    return host(a)
