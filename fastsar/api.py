"""One call for spotlight image formation on any of the supported devices.

    import fastsar
    img = fastsar.form_image(S, ant, fmin, df, nx, ny, spx, spy, e1, e2)            # factorized backprojection
    img = fastsar.form_image(..., algorithm='pfa')                                  # polar format

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
    'auto'  tpu if JAX sees a TPU, else cuda if CuPy sees a GPU, else cpu

pfa_guard: margin (m) around the scene that polar format keeps free of wrap-around; 300 m suits orbital scenes of a
few kilometers and must be smaller for small simulated scenes.

Precision: 'float32' (default), 'float16' (CUDA: float16 storage throughout with float32 accumulation),
'single-pass' (TPU: one bfloat16 pass per product, the device default), 'three-pass' (TPU: float32-class accuracy).
On the TPU 'float32' means three-pass.

Tile size: T='auto' (default) picks the largest final tile (32 or 16 pixels) whose predicted error against exact
backprojection meets target_db (-40 dB by default). The error of the final stage's plane-wave model grows as the
square of the tile size and falls with range, so orbital collections keep T=32 and short-range (airborne) ones
drop to 16. The prediction is kept as ImageFormer.predicted_error_db.
"""
import os
import numpy as np

from .sim import Collect


def _window(P, K, sll=35.0, nbar=4):
    from scipy.signal.windows import taylor
    return taylor(P, nbar=nbar, sll=sll, norm=False).astype(np.float32), taylor(K, nbar=nbar, sll=sll, norm=False).astype(np.float32)


def available_backends():
    """The backends this machine can run, in the order 'auto' tries them."""
    out = []
    try:
        import jax
        if any(d.platform == 'tpu' for d in jax.devices()):
            out.append('tpu')
    except Exception:
        pass
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            out.append('cuda')
    except Exception:
        pass
    out.append('cpu')
    return out


_JAX_PROGRAMS = {}         # compiled JAX/TPU programs by plan signature (ImageFormer)

BACKENDS = ('auto', 'tpu', 'cuda', 'cpu', 'jax')


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
        raise ValueError("backend 'cuda' needs CuPy and an Nvidia GPU; this machine has " + ', '.join(have + ['jax']))
    if backend == 'tpu' and 'tpu' not in have and not os.environ.get('FFBP_FORCE_TPU_KERNELS'):
        raise ValueError("backend 'tpu' needs JAX on a Cloud TPU; this machine has " + ', '.join(have + ['jax']))
    return backend


def _check_history(S, P=None, name='phase history'):
    """S [P, K] complex (numpy or cupy) with at least 2 pulses and 2 samples, all finite; -> S unchanged. Real or
    integer samples are refused (an I/Q pair interleaved along the last axis is S[..., 0] + 1j * S[..., 1])."""
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
    for i in range(0, S.shape[0], 4096):
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
        if not np.isscalar(v) or not np.isfinite(v) or v < 1 or int(v) != v:
            raise ValueError(f'{n} must be a positive integer, got {v!r}')
    for n, v in (('spx', spx), ('spy', spy)):
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f'{n} must be a positive spacing in metres, got {v!r}')
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    if e1.shape != (3,) or e2.shape != (3,):
        raise ValueError(f'e1 and e2 must be 3-vectors, got shapes {e1.shape} and {e2.shape}')
    if abs(e1 @ e1 - 1) > 1e-6 or abs(e2 @ e2 - 1) > 1e-6 or abs(e1 @ e2) > 1e-6:
        raise ValueError(f'e1 and e2 must be orthonormal (unit length, perpendicular), got {e1.tolist()} and {e2.tolist()}')
    return int(nx), int(ny), e1, e2


def _jax_group(plan, K):
    """First-level children per group on a TPU: 4, fewer when they do not fit (FASTSAR_TPU_GROUP overrides;
    ImageFormer halves it again if the device still runs out of memory). On a v6e the 2025 Capella spotlight formed
    in 9.9 s with groups of 4 and of 8, 1% slower with 2 and 5% slower with 1, and 16 did not fit. The fit is
    memory.full_speed's model (the padded history planes and XLA's temporaries per child) within 95% of the
    device memory: groups of 2 for the 74,203-pulse 2024 spotlight on a 32 GB v6e, 4 for the 2025 one."""
    if os.environ.get('FASTSAR_TPU_GROUP'):
        return int(os.environ['FASTSAR_TPU_GROUP'])
    from .memory import device_available, child_bytes, TPU_GROUP, tpu_history_bytes
    hbm = device_available('tpu')
    hist, per = tpu_history_bytes(plan, K), child_bytes(plan, 'tpu')
    ng = TPU_GROUP
    while ng > 1 and hist + ng * per > 0.95 * hbm:
        ng //= 2
    return ng


def final_weights(plan, weight, P, grad=False, points=None):
    """Weight of each final subaperture at each final tile [ntiles, Pf] (float32), for ImageFormer's aperture_weight:
    weight(points [m, 3], pulses [n] int) -> [n, m], the per-pulse weights at the tile centers, averaged over the
    pulses that each final subaperture spans (its center on the input pulse axis composed through the levels'
    decimations, width the product of the decimation factors) by 3-point Gauss-Legendre quadrature. grad=True
    also returns the gradient along e1 and e2 (per metre) at the tile centers: [3, ntiles, Pf]."""
    cen = np.asarray(plan['final']['cen'], np.float64)
    if grad:
        h1, h2 = 0.5 * plan['T'] * plan['spx'], 0.5 * plan['T'] * plan['spy']
        e1, e2 = np.asarray(plan['e1'], np.float64), np.asarray(plan['e2'], np.float64)
        w = [final_weights(plan, weight, P, points=q) for q in (cen, cen + h1 * e1, cen - h1 * e1, cen + h2 * e2, cen - h2 * e2)]
        return np.stack([w[0], (w[1] - w[2]) / (2 * h1), (w[3] - w[4]) / (2 * h2)]).astype(np.float32)
    q = cen if points is None else points
    idx, D = np.arange(plan['final']['P'], dtype=np.float64), 1.0
    for lv in reversed(plan['levels']):
        idx = lv['Dp'] * idx + float(lv['pidx'][0])
        D *= lv['Dp']
    # pulse p covers [p - 1/2, p + 1/2); a subaperture spans [idx - D/2, idx + D/2] within the collection
    a = np.clip(idx - D / 2, -0.5, P - 0.5); b = np.clip(idx + D / 2, -0.5, P - 0.5)
    t, gw = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5.0, 8.0, 5.0]) / 18.0
    nodes = np.clip(np.rint(0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * t[None, :]), 0, P - 1).astype(np.int64)
    U, inv = np.unique(nodes.ravel(), return_inverse=True)
    W = np.asarray(weight(q, U), np.float64)[inv].reshape(len(idx), 3, -1)    # [Pf, 3, ntiles]
    # a subaperture centred beyond the collection holds the filter tails of the edge pulses (the decimators run m
    # outputs past each end): it takes the weight of the nearest pulse (its nodes clip to it), not zero
    m = np.tensordot(gw, W, axes=(0, 1))                                          # [Pf, ntiles]
    return np.ascontiguousarray(m.T, np.float32)


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
    pays only the image formation. Reuse one former for repeated images of the same geometry (or for timing);
    a different antenna path needs a new former. Arguments as for form_image, and
    aperture_weight(points [m, 3], pulses [n]) -> W [n, m]: a per-pixel weight of pulses (int indices) (a stripmap aperture window), applied
    in the final stage as the mean weight of each final subaperture's pulses at each final tile's center, with its
    first-order variation across the tile (final_weights), and
    ref [P]: the range (one way) each pulse is referenced to, when it is not |ant| (the distance to the origin):
    a bistatic collection's half path |tx| / 2 + |rcv| / 2, or a reference point other than the origin."""

    def __init__(self, ant, fmin, df, K, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), backend='auto',
                 precision='float32', window=True, T='auto', levels=3, pmax=0.4, target_db=-40.0, aperture_weight=None,
                 ref=None):
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
        self.backend = _backend(backend)
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
            self.predicted_error_db = float(err)
        self.T = T
        plan = ffbp2.make_plan(col, nx, ny, spx, spy, T=T, nlev=levels, pmax=pmax, e1=e1, e2=e2)
        self._plan, self._grid, self._levels, self._e1, self._e2 = plan, (nx, ny, spx, spy), levels, e1, e2
        if ref is not None and np.shape(ref) != (self.P,):
            raise ValueError(f'ref must hold one range per pulse ({self.P}), got shape {np.shape(ref)}')
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
            self._plan, self._filt = plan, filt
            self._ng = _jax_group(plan, self.K) if filt == 'pallas2' else 1
            from .memory import TPU_GROUP
            if filt == 'pallas2' and self._ng < min(TPU_GROUP, plan['levels'][0]['C']) and not os.environ.get('FASTSAR_TPU_GROUP'):
                from .memory import warn, gb, full_speed, device_available, children
                warn(f'tpu: first-level groups of {children(self._ng)} instead of {TPU_GROUP} for lack of device memory, which is '
                     f'slower; full speed needs about {gb(full_speed(plan, self.K, "tpu")[0])} of TPU memory, '
                     f'{gb(device_available("tpu"))} is free')
            self._fn, static = self._program(self._ng)
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
        S = self._host(S)
        hre, him, scale = ffbp2.prepare(self._pol, S)
        hre, him = self._fn.pad(hre, him)
        return _Staged(S, hre, him, scale, self._fn)

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
                wp, wk = cp.asarray(wp), cp.asarray(wk)
            S = (S * wp[:, None] * wk[None, :]).astype(np.complex64)
        else:
            S = S.astype(np.complex64, copy=False)
        if not S.flags.c_contiguous:              # a transposed or strided view (the kernels read rows in place)
            S = S.copy()
        return S

    def __call__(self, S):
        """S [P, K] complex phase history (numpy, or cupy on the cuda backend), or former.stage(S) -> complex64
        image [nx, ny]."""
        staged = S if isinstance(S, _Staged) else None
        S = staged.S if staged is not None else self._host(S)
        if self.backend == 'cuda':
            import cupy as cp
            return cp.asnumpy(self._form(S)).astype(np.complex64, copy=False)       # a host S may stream (ffbp_cuda)
        if self.backend == 'cpu':
            return self._form(S).astype(np.complex64, copy=False)
        from . import ffbp2
        while True:
            if staged is not None and staged.fn is self._fn:
                hre, him, scale = staged.hre, staged.him, staged.scale
                staged = None
            else:
                staged = None
                hre, him, scale = ffbp2.prepare(self._pol, S)
                hre, him = self._fn.pad(hre, him)
            try:
                re, im = self._fn(hre, him, self._arrs)
                break
            except Exception as e:        # device memory: fewer first-level children per group, down to one
                # a kernel's on-chip scratch (VMEM) is sized at compile time and does not depend on the group size
                if not any(m in str(e) for m in ('RESOURCE_EXHAUSTED', 'Ran out of memory', 'OOM')) or 'Vmem' in str(e):
                    raise
                from .memory import warn, gb, full_speed, child_bytes, children, tpu_history_bytes
                if self._ng <= 1:
                    need = tpu_history_bytes(self._plan, self.K) + child_bytes(self._plan, 'tpu')
                    raise MemoryError(f'{self.backend}: the phase history ({gb(8.0 * self.P * self.K)} as float32 planes) and '
                                      f'one first-level child do not fit in device memory; this needs about {gb(need)}. '
                                      'Use the cuda or cpu backend (both stream or read the history in place), a device '
                                      'with more memory, or fewer pulses.') from e
                self._ng //= 2
                warn(f'{self.backend}: out of device memory; retrying with first-level groups of {children(self._ng)} '
                     f'(slower); full speed needs about {gb(full_speed(self._plan, self.K, "tpu")[0])}')
                del hre, him
                self._fn = self._program(self._ng)[0]
        return ((np.asarray(re) + 1j * np.asarray(im)) * scale).astype(np.complex64)

    def memory(self):
        """What full speed needs on this former's device and what is free: dict(backend, needed, available,
        full_speed, parts) in bytes (fastsar.memory)."""
        from . import memory as mem
        from . import ffbp2
        plan = getattr(self, '_plan', None) or ffbp2.make_plan(Collect(fmin=1.0, df=1.0, K=self.K, ant=self.ant, res=0.5),
                                                                 *self._grid, T=self.T, nlev=self._levels, e1=self._e1, e2=self._e2)
        need, parts = mem.full_speed(plan, self.K, self.backend)
        avail = mem.device_available(self.backend)
        return dict(backend=self.backend, needed=need, available=avail, full_speed=need <= avail, parts=parts)

    def _program(self, ng):
        """The compiled JAX/TPU program for ng first-level children per group, with the plan's device arrays: one per
        plan signature, so that patches of a mosaic with equal shapes and filters share it."""
        from . import ffbp2
        key = (self._pol, self._filt, ng, ffbp2.plan_signature(self._plan))
        hit = _JAX_PROGRAMS.get(key)
        if hit is None:
            hit = (ffbp2.make_ffbp(self._pol, self._plan, self._filt, 1 << 26, 'direct', pallas_pb=256, pallas_nc=8,
                                   pallas_ng=ng, pallas_final=2, pallas_gen=3), ffbp2.static_arrays(self._pol, self._plan, self._filt))
            if len(_JAX_PROGRAMS) >= 32:
                _JAX_PROGRAMS.pop(next(iter(_JAX_PROGRAMS)))
            _JAX_PROGRAMS[key] = hit
        return hit


def form_image(S, ant, fmin, df, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), algorithm='ffbp',
               backend='auto', precision='float32', window=True, T='auto', levels=3, pmax=0.4, pfa_guard=300.0, target_db=-40.0,
               ref=None):
    """Form the complex image [nx, ny] (complex64). See the module docstring for the arguments. For more than one
    image of the same geometry, build an ImageFormer once and call it; this function sets one up on every call."""
    if algorithm not in ('ffbp', 'pfa'):
        raise ValueError("algorithm must be 'ffbp' or 'pfa'")
    S = _check_history(np.asarray(S), len(_check_positions(ant)))
    if algorithm == 'pfa':
        nx, ny, e1, e2 = _check_grid(nx, ny, spx, spy, e1, e2)
        P, K = S.shape
        col = Collect(fmin=float(fmin), df=float(df), K=K, ant=np.asarray(ant, np.float64), res=0.5)
        if window:
            wp, wk = _window(P, K)
            S = (S * wp[:, None] * wk[None, :]).astype(np.complex64)
        return _pfa(S.astype(np.complex64), col, nx, ny, spx, spy, np.asarray(e1, np.float64), np.asarray(e2, np.float64), pfa_guard)
    return ImageFormer(ant, fmin, df, S.shape[1], nx, ny, spx, spy, e1, e2, backend, precision, window, T, levels, pmax, target_db,
                       ref=ref)(S)


def _pfa(S, col, nx, ny, spx, spy, e1, e2, guard=300.0):
    """Polar format with its final resampling (removes the planar-wavefront displacement), pulse resampling in
    gather form. The frequency samples are weighted by f_c / f_k so that polar format applies the same spectral
    weighting as backprojection (the polar Jacobian)."""
    from . import pfa2
    import jax
    import jax.numpy as jnp
    K = S.shape[1]
    f_k = col.fmin + np.arange(K) * col.df
    S = (S * ((col.fmin + (K // 2) * col.df) / f_k)[None, :]).astype(np.complex64)
    geo = pfa2.geometry(col, nx, ny, spx, spy, e1=e1, e2=e2, guard=guard)
    dist = pfa2.distortion(col, nx, ny, spx, spy, e1, e2)
    fn = pfa2.make_pfa(geo, nx, ny, spx, spy, 'taps', None, jax.lax.Precision.HIGHEST, dist=dist)
    W, alpha, eps_r, shift, eps_a = pfa2.arrays(geo, S.shape[0], 'taps')
    scale = float(np.abs(S).max()) or 1.0          # an all-zero history gives a zero image, not 0/0
    out = fn(jnp.asarray((S.real / scale).astype(np.float32)), jnp.asarray((S.imag / scale).astype(np.float32)), W, alpha, eps_r, shift, eps_a)
    return (np.asarray(out) * scale).astype(np.complex64)
