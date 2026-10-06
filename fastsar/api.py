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


class ImageFormer:
    """Factorized backprojection set up once for a collection geometry and output grid, then called on phase
    histories:  former = ImageFormer(ant, fmin, df, K, nx, ny, spx, spy, e1, e2); img = former(S).

    Building plans the tiles and filters, computes the float64 geometry and compiles the kernels; each call then
    pays only the image formation. Reuse one former for repeated images of the same geometry (or for timing);
    a different antenna path needs a new former. Arguments as for form_image."""

    def __init__(self, ant, fmin, df, K, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), backend='auto',
                 precision='float32', window=True, T='auto', levels=3, pmax=0.4, target_db=-40.0):
        from . import ffbp2
        import warnings
        self.ant = np.asarray(ant, np.float64)
        self.P, self.K = self.ant.shape[0], int(K)
        self.window = window
        col = Collect(fmin=float(fmin), df=float(df), K=self.K, ant=self.ant, res=0.5)
        e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
        self.backend = available_backends()[0] if backend == 'auto' else backend
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
        coll = ffbp2.collection_arrays(plan, self.ant)
        if window:
            self.wp, self.wk = _window(self.P, self.K)
        if self.backend == 'cuda':
            from . import ffbp_cuda
            if precision not in ('float32', 'float16'):
                raise ValueError("cuda precision: 'float32' or 'float16'")
            if precision == 'float16' and T != 32:
                raise ValueError('cuda float16 uses the tensor-core final stage, which is built for T=32')
            self._form = ffbp_cuda.make_ffbp_cuda(plan, coll, final_mode='f16tc' if precision == 'float16' else 'fp32',
                                                  store='f16' if precision == 'float16' else 'fp32')
        elif self.backend == 'cpu':
            from . import ffbp_cpu
            if precision != 'float32':
                raise ValueError("cpu precision: 'float32'")
            self._form = ffbp_cpu.make_ffbp_cpu(plan, coll)
        elif self.backend in ('tpu', 'jax'):
            pol = {'float32': 'fp32_high' if self.backend == 'tpu' else 'fp32', 'three-pass': 'fp32_high',
                   'single-pass': 'fp32_fast', 'float16': 'f16'}.get(precision)
            if pol is None:
                raise ValueError(f'unknown precision {precision!r}')
            filt = 'pallas2' if self.backend == 'tpu' else 'conv'
            self._pol = pol
            self._fn = ffbp2.make_ffbp(pol, plan, filt, 1 << 26, 'direct', pallas_pb=256, pallas_nc=8, pallas_ng=16,
                                       pallas_final=2, pallas_gen=3)
            self._arrs = ffbp2.device_arrays(pol, plan, coll, ffbp2.static_arrays(pol, plan, filt))
        else:
            raise ValueError(f'unknown backend {backend!r}')

    def __call__(self, S):
        """S [P, K] complex phase history -> complex64 image [nx, ny]."""
        S = np.asarray(S)
        if S.shape != (self.P, self.K):
            raise ValueError(f'phase history must be {(self.P, self.K)}, got {S.shape}')
        S = (S * self.wp[:, None] * self.wk[None, :]).astype(np.complex64) if self.window else S.astype(np.complex64)
        if self.backend == 'cuda':
            import cupy as cp
            return cp.asnumpy(self._form(cp.asarray(S), ng=8)).astype(np.complex64)
        if self.backend == 'cpu':
            return self._form(S, ng=8).astype(np.complex64)
        from . import ffbp2
        hre, him, scale = ffbp2.prepare(self._pol, S)
        re, im = self._fn(hre, him, self._arrs)
        return ((np.asarray(re) + 1j * np.asarray(im)) * scale).astype(np.complex64)


def form_image(S, ant, fmin, df, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), algorithm='ffbp',
               backend='auto', precision='float32', window=True, T='auto', levels=3, pmax=0.4, pfa_guard=300.0, target_db=-40.0):
    """Form the complex image [nx, ny] (complex64). See the module docstring for the arguments. For more than one
    image of the same geometry, build an ImageFormer once and call it; this function sets one up on every call."""
    S = np.asarray(S)
    if algorithm == 'pfa':
        P, K = S.shape
        col = Collect(fmin=float(fmin), df=float(df), K=K, ant=np.asarray(ant, np.float64), res=0.5)
        if window:
            wp, wk = _window(P, K)
            S = (S * wp[:, None] * wk[None, :]).astype(np.complex64)
        return _pfa(S.astype(np.complex64), col, nx, ny, spx, spy, np.asarray(e1, np.float64), np.asarray(e2, np.float64), pfa_guard)
    if algorithm != 'ffbp':
        raise ValueError("algorithm must be 'ffbp' or 'pfa'")
    return ImageFormer(ant, fmin, df, S.shape[1], nx, ny, spx, spy, e1, e2, backend, precision, window, T, levels, pmax, target_db)(S)


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
    scale = float(np.abs(S).max())
    out = fn(jnp.asarray((S.real / scale).astype(np.float32)), jnp.asarray((S.imag / scale).astype(np.float32)), W, alpha, eps_r, shift, eps_a)
    return (np.asarray(out) * scale).astype(np.complex64)
