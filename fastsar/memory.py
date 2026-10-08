"""Memory planning: what the full-speed configuration of a former needs, what is free, and the warning raised when
FastSAR falls back to a slower path for lack of memory.

    former = fastsar.ImageFormer(...)
    former.memory()          # dict(backend, needed, available, full_speed, parts)

Full speed keeps the phase history on the device (two planes of float32, or float16 on CUDA; the CPU reads complex64
in place) and forms the first level in groups of 8 children (4 on a TPU), each holding all pulses at the decimated
range length. With less memory the CPU uses smaller groups or pulse blocks, the GPU and TPU use smaller groups, CUDA
streams the history from host memory through the first level, and a CUDA mosaic keeps a window of its shared range
profiles on the device. Each fallback that FastSAR chooses raises a MemoryWarning naming it; filter it with
warnings.simplefilter('ignore', fastsar.MemoryWarning)."""
import os
import warnings


class MemoryWarning(UserWarning):
    """FastSAR fell back to a slower path because the device or host lacked memory."""


def _gb(n):
    """A byte count for messages: GB, or MB below one GB."""
    return f'{n / 1e9:.1f} GB' if n >= 1e9 else f'{n / 1e6:.0f} MB'


def _nchildren(n):
    return f'{n} child' if n == 1 else f'{n} children'


def _warn(msg):
    warnings.warn(msg, MemoryWarning, stacklevel=3)


def _env_number(name, least, kind=int):
    """The environment variable `name` as a number of `kind` at least `least`, or None when it is unset or empty."""
    v = os.environ.get(name)
    if not v:
        return None
    try:
        x = kind(v)
    except ValueError:
        raise ValueError(f'{name}={v!r}: expected {"an integer" if kind is int else "a number"}') from None
    if not x >= least:
        raise ValueError(f'{name}={v!r}: must be at least {least}')
    return x


def host_available():
    """Bytes of host memory available (MemAvailable on Linux, else total physical memory)."""
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        return 64e9


def cuda_free():
    """Free bytes on the current CUDA device, counting the blocks CuPy's memory pool holds cached but unused (an
    earlier call's arrays, freed to the pool and not to the driver)."""
    import cupy as cp
    return cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()


def _jax_stats():
    import jax
    try:
        return jax.devices()[0].memory_stats()
    except Exception:
        return None


def device_available(backend):
    """Free bytes on the backend's device (host memory for 'cpu'; 16 GB on a TPU that reports no statistics)."""
    if backend == 'cuda':
        return float(cuda_free())
    if backend in ('tpu', 'jax'):
        st = _jax_stats()
        if st and 'bytes_limit' in st:
            return float(st['bytes_limit'] - st.get('bytes_in_use', 0))
        return float(host_available()) if backend == 'jax' else 16e9
    return float(host_available())


def tpu_capacity():
    """Bytes a TPU program may use (the device's limit, whatever other arrays hold now; 16 GB without statistics)."""
    st = _jax_stats()
    return float(st['bytes_limit']) if st and 'bytes_limit' in st else 16e9


# device bytes of a CUDA first-level child over its planes (all pulses at the decimated range length), measured on
# the L4 (docs/performance.md)
CUDA_CHILD = 1.8
CUDA_GROUP = 8             # first-level children per group at full speed on CUDA (and on the CPU)
STREAM_PULSES = 4096       # pulses per uploaded block when the CUDA first level streams a host phase history
STREAM_FRACTION = 0.5      # CUDA streams when the history's device planes exceed this fraction of the free memory

# first-level children per group at full speed on a TPU (api._jax_group), and XLA's temporaries of the TPU program
# besides the first-level children
TPU_GROUP = 4
TPU_FIXED = 4e9


def tpu_history_bytes(plan, K):
    """Device bytes of the phase history as the TPU program holds it: two float32 planes padded for the first-level
    kernel and its plan arrays (about 1.2 times the bare planes)."""
    return 1.2 * 8.0 * plan['levels'][0]['P'] * K


def child_bytes(plan, backend):
    """Device bytes of one first-level child: all pulses and its output pulses at the decimated range length. On a
    TPU, XLA's temporaries per child as compiled for the Capella spotlights on a v6e (about 12 bytes per output
    range sample and pulse), beside TPU_FIXED for the later levels."""
    lv = plan['levels'][0]
    per = 12.0 if backend == 'tpu' else 8.0
    return per * lv['Ko'] * (lv['P'] + lv['Po'])


def cuda_child(lv0, isz, stream):
    """Device bytes of one CUDA first-level child (level lv0, isz bytes per complex sample in storage): CUDA_CHILD
    times its planes of all pulses, or when streaming its output planes and CUDA_CHILD times a block's planes."""
    if stream:
        return isz * lv0['Ko'] * (lv0['Po'] + CUDA_CHILD * STREAM_PULSES)
    return CUDA_CHILD * isz * lv0['Ko'] * lv0['P']


def cuda_streams(planes, free):
    """Whether a host history of `planes` device bytes streams through the CUDA first level, given `free` bytes."""
    return planes > STREAM_FRACTION * free


def cuda_group(lv0, K, isz, free, stream):
    """First-level children per CUDA group: CUDA_GROUP, halved until the group fits in `free` bytes (the memory
    left after the image and the uploaded history; when streaming, less the block in flight), at most lv0's C."""
    per = cuda_child(lv0, isz, stream)
    if stream:
        free -= (8 + isz) * STREAM_PULSES * K          # the uploaded complex64 block and its planes
    ng = CUDA_GROUP
    while ng > 1 and min(ng, lv0['C']) * per > free:
        ng //= 2
    return min(ng, lv0['C'])


def full_speed(plan, K, backend, isz=8):
    """Device memory the full-speed configuration requires, with the headroom FastSAR's sizing keeps: (total, parts
    dict of what full speed holds). isz: bytes per complex sample of the CUDA storage (4 for float16).
    CPU: groups of 8 children within a quarter of the available memory. CUDA: the image, the history uploaded (its
    planes within STREAM_FRACTION of the memory left after the image) and groups of 8 children in what remains, as
    ffbp_cuda sizes them. TPU: the history and groups of TPU_GROUP children within 95% of the device memory."""
    lv = plan['levels'][0]
    image = 8.0 * plan['Nx'] * plan['Ny']
    if backend == 'tpu':
        hist, groups = tpu_history_bytes(plan, K) + TPU_FIXED, min(TPU_GROUP, lv['C']) * child_bytes(plan, backend)
        return (hist + groups) / 0.95, dict(history=hist, first_level=groups, image=image)
    if backend == 'cuda':
        hist, groups = float(isz) * lv['P'] * K, min(CUDA_GROUP, lv['C']) * cuda_child(lv, isz, False)
        return image + max(hist / STREAM_FRACTION, hist + groups), dict(history=hist, first_level=groups, image=image)
    groups = 8 * child_bytes(plan, backend)
    return 4 * groups + image, dict(history=0.0, first_level=groups, image=image)
