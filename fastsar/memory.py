"""Memory planning: what the full-speed configuration of a former needs, what is free, and the warning raised when
FastSAR falls back to a slower path for lack of memory.

    former = fastsar.ImageFormer(...)
    former.memory()          # dict(backend, needed, available, full_speed, note)

Full speed keeps the phase history on the device (two float32 planes; the CPU reads complex64 in place) and forms
the first level in groups of 8 children (4 on a TPU), each holding all pulses at the decimated range length. With
less memory the CPU and GPU use smaller groups (the history is read more often), the GPU streams the history from
host memory through the first level, and a mosaic keeps its shared range profiles on the host. Each fallback raises
a MemoryWarning that names it and the memory full speed needs; filter it with
warnings.simplefilter('ignore', fastsar.MemoryWarning)."""
import os
import warnings

import numpy as np


class MemoryWarning(UserWarning):
    """FastSAR fell back to a slower path because the device or host lacked memory."""


def gb(n):
    """A byte count for messages: GB, or MB below one GB."""
    return f'{n / 1e9:.1f} GB' if n >= 1e9 else f'{n / 1e6:.0f} MB'


def children(n):
    return f'{n} child' if n == 1 else f'{n} children'


def warn(msg):
    warnings.warn(msg, MemoryWarning, stacklevel=3)


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


def device_available(backend):
    """Free bytes on the backend's device (host memory for 'cpu')."""
    if backend == 'cuda':
        import cupy as cp
        return float(cp.cuda.Device().mem_info[0])
    if backend in ('tpu', 'jax'):
        import jax
        try:
            st = jax.devices()[0].memory_stats()
            return float(st['bytes_limit'] - st.get('bytes_in_use', 0))
        except Exception:
            return host_available() if backend == 'jax' else 16e9
    return float(host_available())


# device bytes of a CUDA first-level child over its planes (all pulses at the decimated range length), measured on
# the L4: the 2025 Capella spotlight peaked at 22.7 GB in groups of 8, the 2024 one needed 25.4 GB
CUDA_CHILD = 1.8

# first-level children per group at full speed on a TPU (api._jax_group: groups of 4 and 8 are equally fast), and
# XLA's temporaries of the TPU program besides the first-level children (5.7 GB with one child at 74,203 pulses)
TPU_GROUP = 4
TPU_FIXED = 4e9


def tpu_history_bytes(plan, K):
    """Device bytes of the phase history as the TPU program holds it: two float32 planes padded for the first-level
    kernel and its plan arrays (12.2 GB for the 74,203 x 17,282 Capella spotlight, 1.18 times the bare planes)."""
    return 1.2 * 8.0 * plan['levels'][0]['P'] * K


def child_bytes(plan, backend):
    """Device bytes of one first-level child: all pulses and its output pulses at the decimated range length. On a
    TPU, XLA's temporaries per child as compiled for the Capella spotlights on a v6e (2.4 GB per child at 74,203
    pulses, about 12 bytes per output range sample and pulse), beside TPU_FIXED for the later levels."""
    lv = plan['levels'][0]
    per = 12.0 if backend == 'tpu' else 8.0
    return per * lv['Ko'] * (lv['P'] + lv['Po'])


def full_speed(plan, K, backend):
    """Device memory the full-speed configuration requires, with the headroom FastSAR's sizing keeps: (total, parts
    dict of what full speed holds). CPU and CUDA give the first-level groups a quarter of the free memory; the TPU
    holds the history and groups of TPU_GROUP children within 95% of the device memory."""
    lv = plan['levels'][0]
    if backend == 'tpu':
        hist, groups = tpu_history_bytes(plan, K) + TPU_FIXED, TPU_GROUP * child_bytes(plan, backend)
        return (hist + groups) / 0.95, dict(history=hist, first_level=groups, image=8.0 * plan['Nx'] * plan['Ny'])
    if backend == 'cuda':
        # as ffbp_cuda sizes its groups: 8 children of CUDA_CHILD times their planes, within 95% of the device
        hist, groups = 8.0 * lv['P'] * K, 8 * CUDA_CHILD * 8.0 * lv['Ko'] * lv['P']
        return (hist + groups) / 0.95, dict(history=hist, first_level=groups, image=8.0 * plan['Nx'] * plan['Ny'])
    hist = 0.0
    groups = 8 * child_bytes(plan, backend)
    image = 8.0 * plan['Nx'] * plan['Ny']
    parts = dict(history=hist, first_level=groups, image=image)
    return hist + 4 * groups + image, parts
