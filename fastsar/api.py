"""One call for spotlight image formation on any of the supported devices.

    import fastsar
    img = fastsar.form_image(S, ant, fmin, df, nx, ny, spx, spy, e1, e2)            # factorized backprojection
    img = fastsar.form_image(..., algorithm='pfa')                                  # polar format
    img = fastsar.form_image(..., algorithm='bp')                                   # exact backprojection

S is the phase history [pulses, samples] (complex, frequency domain, motion compensated to the scene reference
point), ant the antenna phase centers [pulses, 3] in a frame whose origin is the scene reference point, fmin and df
the first frequency and the sample spacing (Hz), and nx, ny, spx, spy, e1, e2 the output grid: pixel counts and
spacings (m) along the unit vectors e1 (azimuth) and e2 (range) of the image plane. The result is a complex64 image
[nx, ny] on that grid.

Backends for factorized backprojection (the same plan, filters and float64 geometry on every device):

    'tpu'   Pallas kernels (JAX on a Cloud TPU)
    'cuda'  CUDA kernels through CuPy (Nvidia GPU)
    'cpu'   C++ kernels with OpenMP (x86-64; compiled with g++ on first use)
    'jax'   the plain JAX program, on whatever device JAX has
    'auto' tpu if JAX reports a TPU, else cuda if CuPy reports a GPU, else cpu

ref [P]: the range (one way) each pulse's samples are referenced to, when it is not |ant| (a bistatic collection's
half path, a vendor's reference point); both algorithms honor it (polar format re-references the samples to |ant|).

pfa_support: 'union' (default) keeps every pulse's band on the polar-format raster, the trapezoidal support of
backprojection; 'inscribed' keeps the rectangle common to all pulses, a clean window at the cost of the bandwidth
outside it (half the range bandwidth in the ground plane of a squinted orbital spotlight, little in the slant plane).
pfa_guard: margin (m) around the scene that polar format keeps free of wrap-around; 300 m suits orbital scenes of a
few kilometers and must be smaller for small simulated scenes.

Precision: 'float32' (default), 'float16' (CUDA: float16 storage throughout with float32 accumulation; JAX: float16
throughout), 'single-pass' (TPU: one bfloat16 pass per product, what the matrix unit does by itself), 'three-pass'
(TPU: float32-class accuracy). On the TPU 'float32' means three-pass.

Tile size: T='auto' (default) picks the largest final tile (32 or 16 pixels) whose predicted error against exact
backprojection meets target_db (-40 dB by default). The error of the final stage's plane-wave model grows as the
square of the tile size and falls with range, so orbital collections keep T=32 and short-range (airborne) ones
drop to 16. The prediction is kept as ImageFormer.predicted_error_db.
"""
import os
import platform
import threading

import numpy as np

from . import memory as _mem
from .sim import Collect


def _window(P, K, sll=35.0, nbar=4):
    from scipy.signal.windows import taylor
    return taylor(P, nbar=nbar, sll=sll, norm=False).astype(np.float32), taylor(K, nbar=nbar, sll=sll, norm=False).astype(np.float32)


_PROBE_ERRORS = {}          # why a device backend is absent (available_backends), for _backend's message


def available_backends():
    """The backends this machine can run, in the order 'auto' tries them. When CuPy or JAX is installed but its
    device probe fails, the error is kept and reported by a request for that backend."""
    out = []
    try:
        import jax
        if any(d.platform == 'tpu' for d in jax.devices()):
            out.append('tpu')
    except ImportError:
        pass
    except Exception as e:
        _PROBE_ERRORS['tpu'] = f'{type(e).__name__}: {e}'
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            out.append('cuda')
    except ImportError:
        pass
    except Exception as e:
        _PROBE_ERRORS['cuda'] = f'{type(e).__name__}: {e}'
    out.append('cpu')
    return out


_JAX_PROGRAMS = {}         # compiled JAX/TPU programs by plan signature (ImageFormer)
_JAX_LOCK = threading.Lock()   # mosaic workers build formers concurrently: one build per signature, no torn eviction
_WIN_CAPS = {}             # TPU level-kernel pulse-block caps by plan signature, found when a kernel ran out of vector memory

BACKENDS = ('auto', 'tpu', 'cuda', 'cpu', 'jax', 'metal')     # the values form_image, ImageFormer, ExactFormer and form_cphd accept


def _backend(backend):
    """backend checked against BACKENDS and this machine, 'auto' resolved."""
    if backend not in BACKENDS:
        raise ValueError(f'unknown backend {backend!r}: one of {", ".join(map(repr, BACKENDS))}')
    if backend in ('cpu', 'jax'):            # no device probe (a mosaic builds a former per patch)
        return backend
    have = available_backends()
    if backend == 'auto':
        return have[0]
    if backend == 'cuda' and 'cuda' not in have:
        raise ValueError("backend 'cuda' needs CuPy and an Nvidia GPU; this machine has " + ', '.join(have + ['jax'])
                         + (f" (the CuPy probe failed: {_PROBE_ERRORS['cuda']})" if 'cuda' in _PROBE_ERRORS else ''))
    if backend == 'metal':
        from . import metal
        if not metal.available():
            raise ValueError("backend 'metal' needs macOS on Apple silicon (ExactFormer only); this machine has " + ', '.join(have + ['jax']))
        return backend
    if backend == 'tpu' and 'tpu' not in have and not os.environ.get('FFBP_FORCE_TPU_KERNELS'):
        raise ValueError("backend 'tpu' needs JAX on a Cloud TPU; this machine has " + ', '.join(have + ['jax'])
                         + (f" (the JAX probe failed: {_PROBE_ERRORS['tpu']})" if 'tpu' in _PROBE_ERRORS else ''))
    return backend


def _check_history(S, P=None, name='phase history', finite=True):
    """S [P, K] complex (numpy or cupy) with at least 2 pulses and 2 samples, all finite (finite=False: the scan
    for NaN and inf is left to the caller); -> S unchanged. Real or integer samples are refused (an I/Q pair
    interleaved along the last axis is S[..., 0] + 1j * S[..., 1])."""
    if not hasattr(S, 'ndim'):
        S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError(f'{name} must be 2-D [pulses, samples], got shape {tuple(S.shape)}')
    if S.dtype.kind != 'c':
        raise TypeError(f'{name} must be complex (complex64 or complex128), got {S.dtype}')
    if P is not None and S.shape[0] != P:
        raise ValueError(f'{name} has {S.shape[0]} pulses but there are {P} antenna positions')
    if S.shape[0] < 2 or S.shape[1] < 2:
        raise ValueError(f'{name} needs at least 2 pulses and 2 samples, got shape {tuple(S.shape)}')
    # a sum per block of rows: one pass, no full-size temporary
    for i in range(0, S.shape[0] if finite else 0, 4096):
        if not np.isfinite(complex(S[i:i + 4096].sum())):
            first = i + int(np.argmax((~np.isfinite(S[i:i + 4096])).any(1)))
            raise ValueError(f'{name} has non-finite samples (NaN or inf), the first in pulse {first}; '
                             'zero them (S[~np.isfinite(S)] = 0) or drop the pulses')
    return S


def _check_positions(a, name='antenna positions', P=None):
    """a [P, 3] finite float64, at least 2 rows (P of them when given)."""
    a = np.asarray(a, np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f'{name} must be [pulses, 3], got shape {a.shape}')
    if P is not None and len(a) != P:
        raise ValueError(f'{name}: {len(a)} rows for {P} pulses')
    if len(a) < 2:
        raise ValueError(f'{name}: at least 2 pulses are needed, got {len(a)}')
    if not np.isfinite(a).all():
        raise ValueError(f'{name}: non-finite values (NaN or inf)')
    return a


def _check_grid(nx, ny, spx, spy, e1, e2):
    """Pixel counts (positive integers), spacings (positive) and the axes (orthonormal 3-vectors) -> nx, ny (int),
    e1, e2 (float64)."""
    for n, v in (('nx', nx), ('ny', ny)):
        if np.ndim(v) != 0 or not np.isfinite(v) or v < 1 or int(v) != v:
            raise ValueError(f'{n} must be a positive integer, got {v!r}')
    for n, v in (('spx', spx), ('spy', spy)):
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f'{n} must be a positive spacing in meters, got {v!r}')
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    if e1.shape != (3,) or e2.shape != (3,):
        raise ValueError(f'e1 and e2 must be 3-vectors, got shapes {e1.shape} and {e2.shape}')
    if abs(e1 @ e1 - 1) > 1e-6 or abs(e2 @ e2 - 1) > 1e-6 or abs(e1 @ e2) > 1e-6:
        raise ValueError(f'e1 and e2 must be orthonormal (unit length, perpendicular), got {e1.tolist()} and {e2.tolist()}')
    return int(nx), int(ny), e1, e2


def _jax_group(plan, K):
    """First-level children per group on a TPU: memory.TPU_GROUP (groups of 4 are as fast as 8; docs/performance.md),
    halved until memory.full_speed's model (the padded history planes and XLA's temporaries per child) fits in 95% of
    the device's memory limit. The limit, not the memory free at this moment, so that patches of a mosaic with equal
    plans get equal groups (and share a program) however many histories the workers have staged. FASTSAR_TPU_GROUP
    overrides it; ImageFormer halves it again if the device still runs out of memory."""
    forced = _mem._env_number('FASTSAR_TPU_GROUP', 1)
    if forced:
        return forced
    hbm = _mem.tpu_capacity()
    hist, per = _mem.tpu_history_bytes(plan, K) + _mem.TPU_FIXED, _mem.child_bytes(plan, 'tpu')
    ng = _mem.TPU_GROUP
    while ng > 1 and hist + ng * per > 0.95 * hbm:
        ng //= 2
    return ng


def final_weights(plan, weight, P, grad=False, points=None, nodes=3):
    """Weight of each final subaperture at each final tile [ntiles, Pf] (float32), for ImageFormer's aperture_weight:
    weight(points [m, 3], pulses [n] int) -> [n, m], the per-pulse weights at the tile centers, averaged over the
    pulses that each final subaperture spans (its center on the input pulse axis composed through the levels'
    decimations, width the product of the decimation factors) by 3-point Gauss-Legendre quadrature (nodes=1: the
    center pulse alone). grad=True also returns the gradient along e1 and e2 (per metre) at the tile centers:
    [3, ntiles, Pf], by central differences of the center-pulse weights at four points half a tile away, the four
    point sets in one call of weight. The quadrature sum is written out rather than a BLAS product: a mosaic calls
    this on several prefetch threads at once, and each BLAS call there started a thread pool the size of the host."""
    if nodes not in (1, 3):
        raise ValueError(f'final_weights: nodes must be 1 or 3, got {nodes!r}')
    cen = np.asarray(plan['final']['cen'], np.float64)
    if grad:
        h1, h2 = 0.5 * plan['T'] * plan['spx'], 0.5 * plan['T'] * plan['spy']
        e1, e2 = np.asarray(plan['e1'], np.float64), np.asarray(plan['e2'], np.float64)
        w0 = final_weights(plan, weight, P, points=cen)
        m = len(cen)
        wq = final_weights(plan, weight, P, points=np.concatenate([cen + h1 * e1, cen - h1 * e1, cen + h2 * e2, cen - h2 * e2]), nodes=1)
        return np.stack([w0, (wq[:m] - wq[m:2 * m]) / (2 * h1), (wq[2 * m:3 * m] - wq[3 * m:]) / (2 * h2)]).astype(np.float32)
    q = cen if points is None else points
    idx, D = np.arange(plan['final']['P'], dtype=np.float64), 1.0
    for lv in reversed(plan['levels']):
        idx = lv['Dp'] * idx + float(lv['pidx'][0])
        D *= lv['Dp']
    # pulse p covers [p - 1/2, p + 1/2); a subaperture spans [idx - D/2, idx + D/2] within the collection
    a = np.clip(idx - D / 2, -0.5, P - 0.5); b = np.clip(idx + D / 2, -0.5, P - 0.5)
    if nodes == 1:
        t, gw = np.array([0.0]), np.array([1.0])
    else:
        t, gw = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5.0, 8.0, 5.0]) / 18.0
    nodes_ = np.clip(np.rint(0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * t[None, :]), 0, P - 1).astype(np.int64)
    U, inv = np.unique(nodes_.ravel(), return_inverse=True)
    W = np.asarray(weight(q, U), np.float64)[inv].reshape(len(idx), len(t), -1)    # [Pf, nodes, npoints]
    # a subaperture centered beyond the collection holds the filter tails of the edge pulses (the decimators run m
    # outputs past each end): it takes the weight of the nearest pulse (its nodes clip to it), not zero
    m = gw[0] * W[:, 0]
    for k in range(1, len(t)):
        m = m + gw[k] * W[:, k]                                                     # [Pf, npoints]
    return np.ascontiguousarray(m.T, np.float32)


def _jax_runtime_error():
    """The exception type JAX raises for a failed computation (JaxRuntimeError; XlaRuntimeError in older releases)."""
    import jax
    err = getattr(jax.errors, 'JaxRuntimeError', None)
    if err is None:
        from jaxlib.xla_extension import XlaRuntimeError as err
    return err


def _to_host(dev):
    """A complex64 host copy of a device image through pinned memory, several times faster than a copy into pageable
    memory (0.03 s against 0.2 s for an 860 MB image). The block comes from CuPy's pinned pool, which keeps it for
    the next image once this one is released (the returned image is itself page-locked memory while it is held);
    FASTSAR_PINNED=0 restores the plain copy."""
    import cupy as cp
    if _mem._env_choice('FASTSAR_PINNED', '1', {'0', '1'}) == '0':
        return cp.asnumpy(dev).astype(np.complex64, copy=False)
    dev = cp.ascontiguousarray(dev.astype(cp.complex64, copy=False))
    mem = _mem._pinned(dev.nbytes)
    host = np.frombuffer(mem, np.complex64, dev.size).reshape(dev.shape)
    dev.get(out=host)
    return host


def _gate_plan(ant, ref, fmin, df, K, nx, ny, spx, spy, e1, e2):
    """The range gate of an nx x ny image centered on the origin, as the mosaic's patches are gated
    (patches.patch_history): the spectrum is padded by a guard of an eighth of the band on either side, each
    pulse's inverse transform (range bins of c / (2 Kp df), about the pulse's reference range) is cut to the N bins
    the image's points span over the aperture with a margin, and the transform back is kept over the band and the
    guards, tapered to 0 across them. -> dict(npl: samples padded below, Kp: padded length, N: bins kept, k0, k1:
    the samples of the gated spectrum kept, taper [k1 - k0] float32 (with the scale Kp / N), fmin, df: of the
    gated history)."""
    from scipy.fft import next_fast_len
    from .patches import _taper
    C = 299792458.0
    band = (fmin, fmin + (K - 1) * df)
    guard = 0.125 * (band[1] - band[0])
    npl = int(np.ceil(guard / df))
    Kp = next_fast_len(K + 2 * npl)
    idx = np.unique(np.linspace(0, len(ant) - 1, min(len(ant), 257)).astype(np.int64))
    a = ant[idx]
    r0 = np.linalg.norm(a, axis=1) if ref is None else np.asarray(ref, np.float64)[idx]
    t = np.linspace(-0.5, 0.5, 5)
    pts = ((t * nx * spx)[:, None, None] * e1 + (t * ny * spy)[None, :, None] * e2).reshape(-1, 3)
    half = np.abs(np.linalg.norm(pts[None] - a[:, None], axis=2) - r0[:, None]).max() + 6 * C / (2 * guard)
    N = next_fast_len(2 * int(np.ceil(half * 2 * Kp * df / C)))
    N += N % 2
    df2, f0 = Kp * df / N, fmin - npl * df
    T = _taper(f0 + df2 * np.arange(N), band, guard)
    keep = np.nonzero(T > 0)[0]
    k0, k1 = int(keep[0]), int(keep[-1]) + 1
    return dict(npl=npl, Kp=int(Kp), N=int(N), k0=k0, k1=k1, taper=(T[k0:k1] * (Kp / N)).astype(np.float32),
                fmin=float(f0 + k0 * df2), df=float(df2))


class _Staged:
    """A phase history staged for a JAX or TPU former (ImageFormer.stage): the host array, its device planes and
    scale, and the program they were padded for."""
    __slots__ = ('S', 'hre', 'him', 'scale', 'fn')

    def __init__(self, S, hre, him, scale, fn):
        self.S, self.hre, self.him, self.scale, self.fn = S, hre, him, scale, fn


class ImageFormer:
    """Factorized backprojection set up once for a collection geometry and output grid, then called on phase
    histories:  former = ImageFormer(ant, fmin, df, K, nx, ny, spx, spy, e1, e2); img = former(S).

    Building plans the tiles and filters, computes the float64 geometry and compiles the kernels; each call then
    performs only the image formation. Reuse one former for repeated images of the same geometry (or for timing);
    a different antenna path needs a new former. Arguments as for form_image (those after e2 by keyword only), and
    aperture_weight(points [m, 3], pulses [n]) -> W [n, m]: a per-pixel weight of pulses (int indices) (a stripmap aperture window), applied
    in the final stage as the mean weight of each final subaperture's pulses at each final tile's center, with its
    first-order variation across the tile (final_weights), and
    ref [P]: the range (one way) each pulse is referenced to, when it is not |ant| (the distance to the origin):
    a bistatic collection's half path |tx| / 2 + |rcv| / 2, or a reference point other than the origin.
    range_gate: the phase history gated in range to the image's swath before formation (each pulse's inverse
    transform, the range bins about the reference range that the image and a margin span, transformed back: fewer
    samples over the same band, and a first level that decimates less). 'auto': where the cuda backend's first
    level could not otherwise hold its filter window in the GPU's shared memory (an image narrow in range from a
    wide-band collection); True: wherever the gate keeps less than half of the samples; False: never. The image
    equals the ungated one to the former's accuracy. On the cuda backend the gate runs on the device in blocks
    sized to the memory that is free, and on the host on all cores when other work has filled the device (the
    note then says so). former.range_gate: None, or dict(K, kept, note)."""

    def __init__(self, ant, fmin, df, K, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), *, backend='auto',
                 precision='float32', window=True, T='auto', levels=None, pmax=0.4, target_db=-40.0, aperture_weight=None,
                 ref=None, range_gate='auto'):
        from . import ffbp2
        import warnings
        self.ant = _check_positions(ant)
        self.P, self.K = self.ant.shape[0], int(K)
        if self.K < 2:
            raise ValueError(f'K must be at least 2 frequency samples, got {K}')
        if not (np.isfinite(fmin) and np.isfinite(df) and fmin > 0 and df > 0):
            raise ValueError(f'fmin and df must be positive frequencies in Hz, got {fmin!r} and {df!r}')
        self.window = window
        col = Collect(fmin=float(fmin), df=float(df), K=self.K, ant=self.ant, res=0.5)
        nx, ny, e1, e2 = _check_grid(nx, ny, spx, spy, e1, e2)
        b = _backend(backend)
        self.backend = 'cpu' if b == 'metal' else b          # no Metal kernels for the factorized former
        self.precision = precision
        self.predicted_error_db = None
        if T == 'auto':
            fmax = float(fmin) + self.K * float(df)
            if self.backend == 'cuda' and precision == 'float16':
                T, err = 32, ffbp2.final_phase_error(self.ant, fmax, nx, ny, spx, spy, e1, e2, 32)
                if err > target_db:
                    warnings.warn(f'cuda float16 needs T=32, predicted error {err:.1f} dB misses target {target_db:.1f} dB; '
                                  "use precision='float32' for this geometry")
            else:
                T, err = ffbp2.choose_T(self.ant, fmax, nx, ny, spx, spy, e1, e2, target_db)
                if err > target_db:
                    warnings.warn(f'predicted error {err:.1f} dB at T={T} misses target {target_db:.1f} dB '
                                  '(short range for this pixel size)')
                if self.backend == 'cpu' and T == 32 and platform.machine() in ('arm64', 'aarch64'):
                    # NEON's four lanes make the 32-pixel tile's final stage the slower one; 16 is also more accurate
                    T, err = 16, ffbp2.final_phase_error(self.ant, fmax, nx, ny, spx, spy, e1, e2, 16)
            self.predicted_error_db = float(err)
        self.T = T
        # levels=None: three, or more where three cannot split a long side into final tiles (factors up to 8)
        nlev = 3 if levels is None else levels
        if ref is not None and np.shape(ref) != (self.P,):
            raise ValueError(f'ref must hold one range per pulse ({self.P}), got shape {np.shape(ref)}')
        if ref is not None and not np.all(np.isfinite(ref)):
            raise ValueError('ref has non-finite values')
        if range_gate not in ('auto', True, False):
            raise ValueError(f"range_gate must be 'auto', True or False, got {range_gate!r}")

        def planned(col, nlev):
            while True:
                try:
                    return ffbp2.make_plan(col, nx, ny, spx, spy, T=T, nlev=nlev, pmax=pmax, e1=e1, e2=e2), nlev
                except ValueError as e:
                    if levels is not None or 'cannot split' not in str(e) or nlev >= 8:
                        raise
                    nlev += 1
        plan, nlev = planned(col, nlev)
        self.range_gate = None
        why = None
        if range_gate == 'auto' and self.backend == 'cuda' and aperture_weight is None:
            from . import ffbp_cuda
            if not ffbp_cuda.first_level_fits(plan):
                why = ("the cuda backend's first level could not hold the range filter of a decimation by "
                       f"{plan['levels'][0]['Dk']} in the GPU's shared memory")
        if why or range_gate is True:
            g = _gate_plan(self.ant, ref, float(fmin), float(df), self.K, nx, ny, spx, spy, e1, e2)
            kept = g['k1'] - g['k0']
            if g['N'] < g['Kp'] and kept < (self.K if why else self.K // 2):
                col = Collect(fmin=g['fmin'], df=g['df'], K=kept, ant=self.ant, res=0.5)
                plan, nlev = planned(col, 3 if levels is None else levels)
                self.range_gate = dict(g, K=self.K, kept=kept, note=(
                    f'range gate: each pulse cut to the {g["N"]} range bins about the image before formation, '
                    f'{kept} samples for {self.K}' + (f' ({why})' if why else '')))
        self._Kf = plan['levels'][0]['K']                     # samples per pulse as the former takes them
        self.levels = nlev
        self._plan = plan
        coll = ffbp2.collection_arrays(plan, self.ant, ref)
        wf = None if aperture_weight is None else final_weights(plan, aperture_weight, self.P, grad=os.environ.get('FASTSAR_WEIGHT_GRAD', '1') == '1')
        if window:
            self.wp, self.wk = _window(self.P, self.K)
        if self.backend == 'cuda':
            from . import ffbp_cuda
            if precision not in ('float32', 'float16'):
                raise ValueError("cuda precision: 'float32' or 'float16'")
            if precision == 'float16' and T != 32:
                raise ValueError('cuda float16 uses the tensor-core final stage, which is built for T=32')
            self._form = ffbp_cuda.make_ffbp_cuda(plan, coll, final_mode='f16tc' if precision == 'float16' else 'fp32',
                                                  store='f16' if precision == 'float16' else 'fp32', wf=wf)
        elif self.backend == 'cpu':
            from . import ffbp_cpu
            if T not in (16, 32):
                raise ValueError(f'the cpu backend supports T=16 or T=32, not {T}')
            if precision != 'float32':
                raise ValueError("cpu precision: 'float32'")
            self._form = ffbp_cpu.make_ffbp_cpu(plan, coll, wf=wf)
        elif self.backend in ('tpu', 'jax'):
            pol = {'float32': 'fp32_high' if self.backend == 'tpu' else 'fp32', 'three-pass': 'fp32_high',
                   'single-pass': 'fp32_fast', 'float16': 'f16'}.get(precision)
            if pol is None:
                raise ValueError(f'unknown precision {precision!r}')
            filt = 'pallas2' if self.backend == 'tpu' else 'conv'
            self._pol = pol
            # one compiled program per plan signature: patches of a mosaic with equal shapes and filters share it
            # (with the device arrays that depend on the plan alone: filters, tile geometry)
            self._filt = filt
            ng = _jax_group(plan, self._Kf) if filt == 'pallas2' else 1
            if filt == 'pallas2' and ng < min(_mem.TPU_GROUP, plan['levels'][0]['C']) and not os.environ.get('FASTSAR_TPU_GROUP'):
                _mem._warn(f'tpu: first-level groups of {_mem._nchildren(ng)} instead of {_mem.TPU_GROUP} for lack of device '
                           f'memory, which is slower; full speed needs about {_mem._gb(_mem.full_speed(plan, self._Kf, "tpu")[0])} '
                           f'of TPU memory, the device has {_mem._gb(_mem.tpu_capacity())}')
            self._fn, static = self._program(ng)
            self._arrs = ffbp2.device_arrays(pol, plan, coll, static)
            if wf is not None:
                import jax.numpy as jnp
                self._arrs['final']['w'] = jnp.asarray(wf if wf.ndim == 3 else np.stack([wf, 0 * wf, 0 * wf]))
        else:
            raise ValueError(f'unknown backend {backend!r}')

    def stage(self, S):
        """The work of a call that precedes formation, done ahead of it: on the JAX and TPU backends the checks of S,
        its scaling to float32 planes and their upload to the device; former(former.stage(S)) equals former(S).
        A mosaic stages the next patches on its worker threads while one forms. Other backends return S."""
        if self.backend not in ('jax', 'tpu'):
            return S
        from . import ffbp2
        S = self._host(S) if self.range_gate is None else self._gate(S)
        hre, him, scale = self._upload(S)
        return _Staged(S, hre, him, scale, self._fn)

    def _upload(self, S):
        """The host history scaled, uploaded and padded for this former's program: (hre, him, scale). A history whose
        planes do not fit in device memory raises MemoryError."""
        from . import ffbp2
        try:
            hre, him, scale = ffbp2.prepare(self._pol, S, getattr(self._fn, 'layout', None))
            hre, him = self._fn.pad(hre, him)
        except Exception as e:
            # a device out of memory, whatever class the runtime raised it as (JaxRuntimeError for a failed
            # computation; a buffer allocation that fails on a TPU has surfaced as a ValueError)
            if 'RESOURCE_EXHAUSTED' not in str(e):
                raise
            raise MemoryError(f'{self.backend}: the phase history ({_mem._gb(8.0 * self.P * self._Kf)} as float32 planes, '
                              f'more once padded for the first-level kernel) does not fit in device memory '
                              f'({_mem._gb(_mem.tpu_capacity())}). Use the cuda or cpu backend (both stream or read the '
                              'history in place), a device with more memory, or fewer pulses.') from e
        return hre, him, scale

    def _host(self, S):
        if not (self.backend == 'cuda' and type(S).__module__.startswith('cupy')):
            S = np.asarray(S)
        if S.shape != (self.P, self.K):
            raise ValueError(f'phase history must be {(self.P, self.K)}, got {S.shape}')
        _check_history(S)
        if self.window:
            wp, wk = self.wp, self.wk
            if not isinstance(S, np.ndarray):
                import cupy as cp
                S = (S * cp.asarray(wp)[:, None] * cp.asarray(wk)[None, :]).astype(np.complex64)
            else:
                # one complex64 copy, windowed in row blocks: whole-array products would hold two more copies of a
                # history that is tens of GB for a long spotlight (320,360 pulses by 20,996 samples: 54 GB)
                out = np.empty(S.shape, np.complex64)
                for p0 in range(0, self.P, 4096):
                    sl = slice(p0, min(self.P, p0 + 4096))
                    np.multiply(S[sl], (wp[sl, None] * wk[None, :]).astype(np.float32), out=out[sl])
                S = out
        else:
            S = S.astype(np.complex64, copy=False)
        if not S.flags.c_contiguous:              # a transposed or strided view (the kernels read rows in place)
            S = S.copy()
        return S

    def _gate(self, S):
        """S [P, K] windowed, checked for non-finite samples and gated in range (_gate_plan) -> [P, K'] complex64
        on the gated frequencies: on the device in pulse blocks for the cuda backend (a CuPy array), in row blocks
        on all cores otherwise (NumPy). On a CUDA device without room for the gate (ffbp_cuda.range_gate: its
        result and one pulse's block within fastsar._xp.cuda_budget) the gate runs on the host as well, the result
        is a NumPy array, and the gate's note says so."""
        K, g = self.K, self.range_gate
        if not type(S).__module__.startswith('cupy'):
            S = np.asarray(S)
        if S.shape != (self.P, K):
            raise ValueError(f'phase history must be {(self.P, K)}, got {S.shape}')
        win = (self.wp, self.wk) if self.window else None
        if self.backend == 'cuda':
            from . import ffbp_cuda
            _check_history(S, finite=False)
            out = ffbp_cuda.range_gate(S, g, win)
            if out is not None:
                return out
            # a device that other work has filled: the gate on the host, on all cores; the former then takes the
            # gated history from the host as it takes any (in blocks, or streamed)
            if 'on the host' not in g['note']:
                g['note'] += (f'; gated on the host: the GPU had {_mem._gb(_mem.cuda_free())} free, less than the '
                              'gate takes there')
            if type(S).__module__.startswith('cupy'):
                S = S.get()
        import scipy.fft as sf
        from .patches import _rows
        _check_history(S)
        out = np.empty((self.P, g['kept']), np.complex64)
        npl, Kp, N, k0, k1, T = g['npl'], g['Kp'], g['N'], g['k0'], g['k1'], g['taper']
        h = N // 2

        def blk(sl):
            z = np.zeros((sl.stop - sl.start, Kp), np.complex64)
            z[:, npl:npl + K] = S[sl] if win is None else S[sl] * (win[0][sl, None] * win[1][None, :])
            q = sf.ifft(z, axis=1, workers=1)
            u = sf.fft(np.concatenate([q[:, :N - h], q[:, Kp - h:]], axis=1), axis=1, workers=1)
            np.multiply(u[:, k0:k1], T, out=out[sl])
        _rows(blk, self.P, chunk=int(max(1, min(64, (1 << 22) // Kp))))
        return out

    def __call__(self, S):
        """S [P, K] complex phase history (numpy, or cupy on the cuda backend), or former.stage(S) -> complex64
        image [nx, ny]."""
        staged = S if isinstance(S, _Staged) else None
        if staged is None and self.range_gate is not None:
            S = self._gate(S)                                      # windowed and checked there; on the device for cuda
            if self.backend == 'cuda':
                return _to_host(self._form(S))
            staged = _Staged(S, None, None, None, None)            # the host history as the former takes it
        if self.backend == 'cuda' and staged is None and not type(S).__module__.startswith('cupy'):
            # a host history is windowed and checked on the device as its row blocks are uploaded (two passes over
            # 1.75 GB on the host cost a 4-vCPU instance more than the whole formation); one that streams is prepared
            # on the host
            S = np.asarray(S)
            if S.shape != (self.P, self.K):
                raise ValueError(f'phase history must be {(self.P, self.K)}, got {S.shape}')
            _check_history(S, finite=False)
            return _to_host(self._form(S, window=(self.wp, self.wk) if self.window else None, check=True))
        S = staged.S if staged is not None else self._host(S)
        if self.backend == 'cuda':
            return _to_host(self._form(S))                                        # a host S may stream (ffbp_cuda)
        if self.backend == 'cpu':
            return self._form(S).astype(np.complex64, copy=False)
        import jax
        from . import ffbp2
        runtime_error = _jax_runtime_error()
        while True:
            if staged is not None and staged.fn is self._fn:
                hre, him, scale = staged.hre, staged.him, staged.scale
                staged = None
            else:
                staged = None
                hre, him, scale = self._upload(S)
            try:
                # dispatch is asynchronous: a device that runs out of memory may say so only when the result is read
                re, im = jax.block_until_ready(self._fn(hre, him, self._arrs))
                break
            except Exception as e:          # device memory: fewer first-level children per group, down to one
                if 'RESOURCE_EXHAUSTED' not in str(e) or not isinstance(e, (runtime_error, ValueError, RuntimeError)):
                    raise
                if 'vmem' in str(e).lower():
                    # a kernel's on-chip scratch (VMEM) is sized at compile time and does not depend on the group
                    # size: smaller pulse blocks in the level kernels (long range lines), until the smallest
                    cap = self._next_win_cap()
                    if cap is None:
                        raise
                    del hre, him
                    _WIN_CAPS[ffbp2.plan_signature(self._plan)] = cap
                    self._fn = self._program(self._fn.stages['ng'])[0]
                    _mem._warn(f'{self.backend}: a level kernel ran out of vector memory; retrying with pulse blocks of '
                               f'at most {cap} samples')
                    continue
                ng = self._fn.stages['ng']          # the group the program runs (a divisor of the children)
                if ng <= 1:
                    need = _mem.tpu_history_bytes(self._plan, self._Kf) + _mem.child_bytes(self._plan, 'tpu')
                    raise MemoryError(f'{self.backend}: the phase history ({_mem._gb(8.0 * self.P * self._Kf)} as float32 '
                                      f'planes) and one first-level child do not fit in device memory; this needs about '
                                      f'{_mem._gb(need)}. Use the cuda or cpu backend (both stream or read the history in '
                                      'place), a device with more memory, or fewer pulses.') from e
                del hre, him
                self._fn = self._program(ng // 2)[0]
                _mem._warn(f'{self.backend}: out of device memory; retrying with first-level groups of '
                           f'{_mem._nchildren(self._fn.stages["ng"])} (slower); full speed needs about '
                           f'{_mem._gb(_mem.full_speed(self._plan, self._Kf, "tpu")[0])}')
        return ((np.asarray(re) + 1j * np.asarray(im)) * scale).astype(np.complex64)

    def memory(self):
        """What full speed needs on this former's device and what is free: dict(backend, needed, available,
        full_speed, parts) in bytes (fastsar.memory)."""
        isz = 4 if self.backend == 'cuda' and self.precision == 'float16' else 8
        need, parts = _mem.full_speed(self._plan, self._Kf, self.backend, isz)
        avail = _mem.device_available(self.backend)
        return dict(backend=self.backend, needed=need, available=avail, full_speed=need <= avail, parts=parts)

    def _next_win_cap(self):
        """The next, halved cap on the samples in a level kernel's pulse block (pulses by band window), starting from
        256 pulses of the widest window; None when blocks of 8 pulses are already reached."""
        bands = self._fn.stages.get('bands') or {}
        if self._filt != 'pallas2' or not bands:
            return None
        wmax = max(b.get('seg', b['win']) for b in bands.values())
        cap = (self._fn.stages.get('win_cap') or 256 * wmax) // 2
        return cap if cap >= 8 * wmax else None

    def _program(self, ng):
        """The compiled JAX/TPU program for ng first-level children per group, with the plan's device arrays: one per
        plan signature, so that patches of a mosaic with equal shapes and filters share it."""
        from . import ffbp2
        sig = ffbp2.plan_signature(self._plan)
        cap = _WIN_CAPS.get(sig)
        key = (self._pol, self._filt, ng, sig, cap)
        with _JAX_LOCK:
            hit = _JAX_PROGRAMS.get(key)
            if hit is None:
                hit = (ffbp2.make_ffbp(self._pol, self._plan, self._filt, 1 << 26, 'direct', pallas_pb=256, pallas_nc=8,
                                       pallas_ng=ng, pallas_final=2, pallas_gen=3, pallas_win_cap=cap),
                       ffbp2.static_arrays(self._pol, self._plan, self._filt))
                if len(_JAX_PROGRAMS) >= 32:
                    _JAX_PROGRAMS.pop(next(iter(_JAX_PROGRAMS)))
                _JAX_PROGRAMS[key] = hit
        return hit


def form_image(S, ant, fmin, df, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), *, algorithm='ffbp',
               backend='auto', precision='float32', window=True, T='auto', levels=None, pmax=0.4, pfa_guard=300.0, pfa_support='union', target_db=-40.0,
               ref=None, interp=None, upsample=None, center=None, range_gate='auto'):
    """Form the complex image [nx, ny] (complex64). See the module docstring for the arguments. precision, T, levels,
    pmax, target_db and range_gate (ImageFormer's) apply to factorized backprojection, pfa_guard and pfa_support to polar format, and interp ('cubic' or
    'linear'), upsample and center to exact backprojection (algorithm='bp'; see ExactFormer). For more than one
    image of the same geometry, build an ImageFormer (or ExactFormer) once and call it; this function sets one up
    on every call."""
    if algorithm not in ('ffbp', 'pfa', 'bp'):
        raise ValueError("algorithm must be 'ffbp', 'pfa' or 'bp'")
    if algorithm != 'bp':
        for n, v in (('interp', interp), ('upsample', upsample), ('center', center)):
            if v is not None:
                raise ValueError(f"{n} applies to exact backprojection (algorithm='bp'), not to algorithm={algorithm!r}")
    # the formers scan S for NaN and inf themselves; polar format is checked here
    S = _check_history(np.asarray(S), len(_check_positions(ant)), finite=algorithm == 'pfa')
    if algorithm == 'bp':
        from .exact import ExactFormer
        return ExactFormer(ant, fmin, df, S.shape[1], nx, ny, spx, spy, e1, e2, backend=backend, window=window,
                           interp='cubic' if interp is None else interp, upsample=upsample, ref=ref, center=center)(S)
    if algorithm == 'pfa':
        nx, ny, e1, e2 = _check_grid(nx, ny, spx, spy, e1, e2)
        P, K = S.shape
        col = Collect(fmin=float(fmin), df=float(df), K=K, ant=np.asarray(ant, np.float64), res=0.5)
        if ref is not None:               # polar format needs the samples referenced to |ant|: move them there
            from .io import rereference
            if np.shape(ref) != (P,):
                raise ValueError(f'ref must hold one range per pulse ({P}), got shape {np.shape(ref)}')
            S = rereference(S, float(fmin), float(df), np.asarray(ref, np.float64) - np.linalg.norm(col.ant, axis=1))
        return _pfa(S, col, nx, ny, spx, spy, np.asarray(e1, np.float64), np.asarray(e2, np.float64), pfa_guard, window, pfa_support)
    return ImageFormer(ant, fmin, df, S.shape[1], nx, ny, spx, spy, e1, e2, backend=backend, precision=precision,
                       window=window, T=T, levels=levels, pmax=pmax, target_db=target_db, ref=ref,
                       range_gate=range_gate)(S)


_PFA_PROGRAMS = {}
_PFA_LOCK = threading.Lock()


def _pfa(S, col, nx, ny, spx, spy, e1, e2, guard=300.0, window=True, support='union'):
    """Polar format with its final resampling (removes the planar-wavefront displacement), pulse resampling in
    gather form. The frequency samples are weighted by f_c / f_k so that polar format applies the same spectral
    weighting as backprojection (the polar Jacobian). The compiled program and its geometry arrays are kept per
    collection geometry (the last four), so repeated images of one geometry compile once; the window, the weighting
    and the scaling run on the device."""
    import hashlib, sys
    from . import pfa2
    import jax
    import jax.numpy as jnp
    P, K = S.shape
    key = (P, K, float(col.fmin), float(col.df), nx, ny, float(spx), float(spy), tuple(e1), tuple(e2), float(guard), bool(window), support,
           hashlib.sha1(np.ascontiguousarray(col.ant).tobytes()).hexdigest())
    with _PFA_LOCK:
        prog = _PFA_PROGRAMS.get(key)
    if prog is None:
        geo = pfa2.geometry(col, nx, ny, spx, spy, e1=e1, e2=e2, guard=guard, support=support)
        dist = pfa2.distortion(col, nx, ny, spx, spy, e1, e2)
        fn = pfa2.make_pfa(geo, nx, ny, spx, spy, 'taps', None, jax.lax.Precision.HIGHEST, dist=dist)
        arrs = pfa2.arrays(geo, P, 'taps')
        f_k = col.fmin + np.arange(K) * col.df
        wp, wk = _window(P, K) if window else (np.ones(P), np.ones(K))
        wkf = jnp.asarray((wk * (col.fmin + (K // 2) * col.df) / f_k).astype(np.float32))
        wpd = jnp.asarray(np.asarray(wp, np.float32))

        @jax.jit
        def prep(Sd):
            x = Sd * wpd[:, None] * wkf[None, :]
            scale = jnp.max(jnp.abs(x))
            scale = jnp.where(scale > 0, scale, 1.0)          # an all-zero history gives a zero image, not 0/0
            return jnp.real(x) / scale, jnp.imag(x) / scale, scale

        prog = (fn, arrs, prep)
        with _PFA_LOCK:
            if len(_PFA_PROGRAMS) >= 4:
                _PFA_PROGRAMS.pop(next(iter(_PFA_PROGRAMS)))
            _PFA_PROGRAMS[key] = prog
    fn, (W, alpha, eps_r, shift, eps_a), prep = prog
    if 'cupy' in sys.modules and jax.default_backend() == 'gpu':
        _mem.cuda_release()                   # blocks and FFT plans CuPy cached are not JAX's to use
    S = np.asarray(S, np.complex64)
    if jax.default_backend() == 'gpu':
        import cupy as cp                                                # upload in row blocks through two pinned
        dev = cp.empty(S.shape, cp.complex64)                            # staging buffers (several times faster than
        rows = max(1, (256 << 20) // max(S.shape[1] * 8, 1))             # JAX's copy from pageable memory, and
        bufs = [_mem._pinned(rows * S.shape[1] * 8) for _ in range(2)]   # without a second host copy of the history)
        stream = cp.cuda.Stream(non_blocking=True)
        events = [None, None]
        for i, p0 in enumerate(range(0, S.shape[0], rows)):
            p1 = min(S.shape[0], p0 + rows); b = i % 2
            if events[b] is not None:
                events[b].synchronize()
            hb = np.frombuffer(bufs[b], np.complex64, rows * S.shape[1]).reshape(rows, S.shape[1])[:p1 - p0]
            np.copyto(hb, S[p0:p1])
            dev[p0:p1].set(hb, stream=stream)
            events[b] = cp.cuda.Event(); events[b].record(stream)
        stream.synchronize()
        del bufs
        Sd = jax.dlpack.from_dlpack(dev)
    else:
        Sd = jax.device_put(S)
    re, im, scale = prep(Sd)
    out = fn(re, im, W, alpha, eps_r, shift, eps_a, scale)               # scaled on the device
    if type(out).__module__.startswith('cupy'):                          # the CUDA gather wrote a CuPy image
        return _to_host(out)                                             # through pinned memory, as ImageFormer returns
    if jax.default_backend() == 'gpu':
        import cupy as cp
        return _to_host(cp.from_dlpack(out))
    return np.asarray(out)
