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


def _env_choice(name, default, allowed):
    """The environment variable `name` (default when unset or empty), which must be one of `allowed` (strings)."""
    v = os.environ.get(name) or default
    if v not in allowed:
        raise ValueError(f'{name}={v!r}: expected one of {", ".join(sorted(allowed))}')
    return v


def _pinned_size(n):
    """n bytes rounded up to a size class of CuPy's pinned pool: at most 1/8 more, with 8 to 16 classes per factor of
    two. The pool reuses a block only at the same size, and patches of a mosaic ask for sizes a few percent apart,
    which without classes left tens of blocks in it."""
    n = max(int(n), 1)
    g = 1 << max(0, n.bit_length() - 4)
    return -(-n // g) * g


def _pinned(n):
    """A block of at least n bytes from CuPy's pinned pool, its size rounded to a class (_pinned_size)."""
    import cupy as cp
    return cp.cuda.alloc_pinned_memory(_pinned_size(n))


def _cgroup_file_cache(stat):
    """Bytes of page cache in a control group's usage, from its memory.stat (the file lists; of a v1 hierarchy the
    totals), or 0 where it cannot be read."""
    try:
        with open(stat) as fh:
            kv = dict(line.split()[:2] for line in fh if len(line.split()) >= 2)
    except (OSError, ValueError):
        return 0
    try:
        if 'total_inactive_file' in kv or 'total_active_file' in kv:
            return int(kv.get('total_inactive_file', 0)) + int(kv.get('total_active_file', 0))
        return int(kv.get('inactive_file', 0)) + int(kv.get('active_file', 0))
    except ValueError:
        return 0


def _cgroup_available():
    """Bytes this process may still use under its control group's memory limit (a container, a systemd scope, a
    batch job), or None where there is no limit: the least of limit minus usage over the group and its ancestors
    (cgroup v2, else v1). The usage is taken without the group's page cache: the kernel charges a group for the
    cache of the files it reads (a phase history just read counts twice, once in the process and once as cache)
    and gives that cache back before it enforces the limit, as MemAvailable counts it for the host."""
    try:
        with open('/proc/self/cgroup') as fh:
            lines = fh.read().split()
    except OSError:
        return None
    best = None
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        if parts[0] == '0' and parts[1] == '':                      # v2: one hierarchy
            base, lim, use = '/sys/fs/cgroup', 'memory.max', 'memory.current'
        elif 'memory' in parts[1].split(','):                       # v1: the memory controller's hierarchy
            base, lim, use = '/sys/fs/cgroup/memory', 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        d = parts[2].rstrip('/')
        while True:
            try:
                with open(f'{base}{d}/{lim}') as fh:
                    v = fh.read().strip()
                if v != 'max' and int(v) < 1 << 60:                 # (v1 writes a huge number for no limit)
                    with open(f'{base}{d}/{use}') as fh:
                        used = int(fh.read().strip())
                    left = int(v) - max(used - _cgroup_file_cache(f'{base}{d}/memory.stat'), 0)
                    best = left if best is None else min(best, left)
            except (OSError, ValueError):
                pass
            if not d:
                break
            d = d.rsplit('/', 1)[0]
    return best


def host_available():
    """Bytes of host memory available: MemAvailable on Linux, less where the process runs under a control group's
    memory limit; else total physical memory."""
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    free = int(line.split()[1]) * 1024
                    cg = _cgroup_available()
                    return free if cg is None else max(min(free, cg), 0)
    except OSError:
        pass
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        return 64e9


def cuda_release():
    """Return CuPy's cached blocks to the device, and with them the work areas of cupy.fft's plan cache, which keeps a
    plan and its work area for the process (7 GB after a wide-band ExactFormer call): called where an allocation on
    the device has failed, before the retry."""
    import cupy as cp
    try:
        cp.fft.config.get_plan_cache().clear()
    except Exception:          # no cache on this CuPy
        pass
    cp.get_default_memory_pool().free_all_blocks()


def cuda_free():
    """Free bytes on the current CUDA device, counting the blocks CuPy's memory pool holds cached but unused (an
    earlier call's arrays, freed to the pool and not to the driver); under a limit of the pool, no more than the
    limit leaves."""
    import cupy as cp
    pool = cp.get_default_memory_pool()
    free = cp.cuda.Device().mem_info[0] + pool.free_bytes()
    limit = pool.get_limit()
    return min(free, limit - pool.used_bytes()) if limit else free


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


def cuda_rest(plan, isz):
    """Device bytes of one first-level child carried through the later levels at the largest of them: per child its
    input planes, the range-decimated intermediate of all its input pulses and the output planes; the deeper levels
    multiply the children by the earlier levels' C. isz: bytes per complex sample in storage."""
    n, worst = 1, 1
    for lv in plan['levels'][1:]:
        worst = max(worst, n * isz * (lv['P'] * lv['K'] + lv['C'] * (lv['P'] + lv['Po']) * lv['Ko']))
        n *= lv['C']
    return worst


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
    planes within STREAM_FRACTION of the memory left after the image), groups of 8 children and one child carried
    through the later levels in what remains, as ffbp_cuda sizes them. TPU: the history and groups of TPU_GROUP children within 95% of the device memory."""
    lv = plan['levels'][0]
    image = 8.0 * plan['Nx'] * plan['Ny']
    if backend == 'tpu':
        hist, groups = tpu_history_bytes(plan, K) + TPU_FIXED, min(TPU_GROUP, lv['C']) * child_bytes(plan, backend)
        return (hist + groups) / 0.95, dict(history=hist, first_level=groups, image=image)
    if backend == 'cuda':
        hist, groups = float(isz) * lv['P'] * K, min(CUDA_GROUP, lv['C']) * cuda_child(lv, isz, False)
        rest = float(cuda_rest(plan, isz)) if len(plan['levels']) > 1 else 0.0
        return (image + max(hist / STREAM_FRACTION, hist + groups + rest),
                dict(history=hist, first_level=groups, later_levels=rest, image=image))
    groups = 8 * child_bytes(plan, backend)
    return 4 * groups + image, dict(history=0.0, first_level=groups, image=image)


def cuda_plan(sizes, budget, *, autofocus=False, sva=False, joined=False):
    """Where the images of a process_cphd product live while the stages after formation run on a CUDA device.

    sizes: bytes of the arrays the product goes through, by name, those it has among 'ground' (the image as
    formed), 'mid' and 'slant' (to_slant's intermediate and its result), 'detected', 'ground_mid' and 'product'
    (the detected slant image, to_ground's intermediate and its result), and 'valid' (a mask of the ground image),
    'display' (the uint8 image of a SIDD, remapped from the last array). budget: device bytes the arrays and the
    stages' temporaries may take (_xp.cuda_budget()). autofocus, sva: whether those stages run.

    A stage needs the arrays it reads and writes that are on the device, and an eighth of the budget for its own
    blocks; autofocus, which writes a new image through blocks of up to 1 GiB, runs on the device only when the
    ground image is there. Of the placements whose every stage fits, the one that moves the fewest bytes between
    host and device is taken (an array on the host crosses once for every stage that reads or writes it).

    joined: the JAX path's accounting (a TPU). An array written on the device is made of blocks that are joined at
    the end, so it is there twice for a moment, and the join of a complex one takes 0.4 of it more (measured on a
    TPU v6e, which keeps complex numbers as two real arrays); autofocus and the display run in blocks whether the
    image is on the device or on the host, the correction writing a new image. Of two placements that move the
    same bytes the one that takes less of the device is taken: the last image of a complex product then goes to
    the host block by block instead of being joined on the device and copied.

    -> dict(place: {name: 'device' or 'host'}, fits: everything is on the device, need: the bytes that takes,
    budget)."""
    import itertools
    names = [k for k in ('ground', 'mid', 'slant', 'detected', 'ground_mid', 'product') if k in sizes]
    size = {k: float(sizes[k]) for k in names}
    valid, display = float(sizes.get('valid', 0)), float(sizes.get('display', 0))
    reserve = budget / 8
    # (arrays read, arrays written, device bytes beside them, runs only where its input is)
    stages = [(('ground',), ('ground',), 0.0, False)]                                      # baseband
    if autofocus and joined:                                         # the image is kept for the guard, with its planes
        stages.append((('ground',), ('ground',), size['ground'], False))
    elif autofocus:
        stages.append((('ground',), (), size['ground'] + 4 * min(size['ground'], float(1 << 30)), True))
    if 'mid' in size:
        stages += [(('ground',), ('mid',), 0.0, False), (('mid',), ('slant',), valid, False)]
        if sva:
            stages.append((('slant',), ('slant',), 0.0, False))
    elif valid:
        stages.append((('ground',), (), valid, True))
    if 'detected' in size:
        stages += [(('slant',), ('detected',), 0.0, False), (('detected',), ('ground_mid',), 0.0, False),
                   (('ground_mid',), ('product',), 0.0, False)]
    if display and not joined:
        stages.append(((names[-1],), (), 3 * display, True))

    def need(dev):
        worst = 0.0
        for read, written, extra, local in stages:
            if local and read[0] not in dev:
                continue
            if joined:                                              # the array read, the blocks written and their join
                made = sum((2.4 if k in ('ground', 'mid', 'slant') else 2.0) * size[k] for k in written if k in dev)
                worst = max(worst, sum(size[k] for k in read if k in dev) + made + extra)
            else:
                worst = max(worst, sum(size[k] for k in set(read + written) if k in dev) + extra)
        if joined and display and names[-1] in dev:
            worst = max(worst, size[names[-1]] + 2 * display)
        return worst + reserve

    def moved(dev):
        n = size['ground'] if 'ground' in dev else 0.0
        if not display and names[-1] in dev:
            n += size[names[-1]]                                    # a complex product comes to the host at the end
        for read, written, _, local in stages:
            if not local:
                n += sum(size[k] for k in read + written if k not in dev)
        return n
    best = None
    # megabytes moved (two sums of the same sizes in another order differ in their last digit), then the need
    cost = (lambda dev: (round((moved(dev) + (size[names[-1]] if display and names[-1] not in dev else 0.0)) / 2 ** 20),
                         need(dev))) if joined else moved
    for r in range(len(names), -1, -1):
        for dev in itertools.combinations(names, r):
            if need(dev) <= budget and (best is None or cost(dev) < cost(best)):
                best = dev
    best = () if best is None else best
    return dict(place={k: 'device' if k in best else 'host' for k in names},
                fits=need(tuple(names)) <= budget if joined else len(best) == len(names),
                need=need(tuple(names)), budget=float(budget))
