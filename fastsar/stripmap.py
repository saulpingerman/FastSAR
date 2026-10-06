"""Stripmap SAR on a straight, constant-velocity track: a point-target simulator of raw echoes, range compression,
the range-Doppler algorithm (RDA), omega-k with exact Stolt interpolation, and time-domain backprojection as the
float64 reference.

    from fastsar import stripmap as sm
    p = sm.make_params(squint_deg=5.0)
    raw = sm.simulate(p, targets)                        # [pulses, fast-time samples] complex128
    img, r, x = sm.focus_stripmap(raw, p, algorithm='omegak', awin='taylor', rwin='taylor')

Geometry. The platform moves along x at speed v; pulse p leaves at slow time eta_p = eta0 + p/prf from x = v eta_p. A
target is given by its zero-Doppler coordinates (x_n, r_n), the along-track position and slant range at closest
approach, so R_n(eta) = sqrt(r_n^2 + (v eta - x_n)^2). The look angle theta is measured from zero Doppler, positive
forward: sin(theta) = (x_n - v eta)/R, and the Doppler frequency is f = 2 v sin(theta)/lambda, so forward squint gives
a positive Doppler centroid f_dc = 2 v sin(theta_s)/lambda.

Signal. The transmitted pulse is the up-chirp exp(+j pi Kr t^2), |t| <= Tp/2, Kr = B/Tp > 0, and the echo is
demodulated with exp(-j 2 pi fc t), so fast-time sample tau_k = t0 + k/fs of pulse p holds
    sum_n amp_n G(u) exp(j pi Kr (tau_k - 2R/c)^2) exp(-j 4 pi R/lambda)    for |tau_k - 2R/c| <= Tp/2,
with t0 the near-range start time and G the two-way amplitude pattern of an antenna of length La steered to the
squint: u = La (sin(theta) - sin(theta_s))/lambda, G = sinc(u)^2 ('sinc2') or exp(-ln 2 (u/0.443)^2) ('gauss'), both
with the one-way 3 dB beamwidth 0.886 lambda/La. Since f - f_dc = 2 v u/La exactly, u is also the normalized Doppler.
FFTs use exp(-j 2 pi f t).

Images are on the zero-Doppler grid: column k at slant range r_k = c t0/2 + k c/(2 fs), row i at zero-Doppler time
eta0 + r_ref tan(theta_s)/v + i/prf (x = v eta). The azimuth processing is circular over the pulses (no padding), so
rows within half an aperture of either end are incomplete. Every algorithm images a target at (x_n, r_n) with phase
arg(amp_n) and the complex scale of backprojection,
    I(x, r) = sum_p W_a(u_p/umax) s_rc(2 R_p/c, eta_p) exp(+j 4 pi R_p/lambda),
where s_rc is the range-compressed echo (matched filter normalized to unit peak gain, range window W_r over |f| <= B/2)
and W_a the azimuth window over the processed band |u| <= umax (default 1: the main lobe of the sinc^2 pattern). The
frequency-domain algorithms apply W_a in the two-dimensional spectrum as a function of the look angle,
sin(theta) = c f_eta/(2 v (fc + f_tau)), which is the stationary-phase counterpart of backprojection's per-pulse
weight, together with the stationary-phase factor prf sqrt(c r/(2 (fc + f_tau) v^2 cos^3(theta))) exp(j pi/4), so
the three can be compared sample by sample. What remains between them comes mostly from the band edge, cut on pulses
in backprojection and on Doppler bins in the frequency domain: -60 to -65 dB where the weighting falls to zero there
(sinc^2 pattern, umax = 1) and -44 dB for a Gaussian pattern cut at -31 dB.

Precision. Phases (4 pi r fc/c is about 10^6 rad) and interpolation positions are computed in float64 with numpy;
transforms, products and the sinc interpolation run in JAX in float32 (default) or float64.
"""
import contextlib
from dataclasses import dataclass

import numpy as np

C = 299792458.0
_U3 = 0.4429                    # sinc(u)^2 = 1/2: one-way 3 dB point of a uniform aperture


@dataclass
class StripParams:
    fc: float           # carrier (Hz)
    B: float            # chirp bandwidth (Hz)
    Tp: float           # pulse length (s)
    fs: float           # complex sampling rate (Hz)
    prf: float          # Hz
    v: float            # platform speed (m/s)
    r0: float           # zero-Doppler slant range of the swath center (m)
    La: float           # antenna length (m)
    squint: float       # rad, positive forward
    na: int             # pulses
    nr: int             # fast-time samples
    t0: float           # fast time of sample 0 (s)
    eta0: float         # slow time of pulse 0 (s)
    pattern: str = 'sinc2'
    extent: float = 1.0  # simulated |u|

    @property
    def lam(self):
        return C / self.fc

    @property
    def Kr(self):
        return self.B / self.Tp

    @property
    def beamwidth(self):
        """One-way 3 dB azimuth beamwidth (rad)."""
        return 2 * _U3 * self.lam / self.La

    @property
    def fdc(self):
        return 2 * self.v * np.sin(self.squint) / self.lam

    @property
    def eta(self):
        return self.eta0 + np.arange(self.na) / self.prf

    @property
    def tau(self):
        return self.t0 + np.arange(self.nr) / self.fs


def make_params(fc=9.6e9, B=100e6, Tp=2e-6, fs=125e6, prf=650.0, v=200.0, r0=5e3, La=1.5, beamwidth_deg=None,
                squint_deg=0.0, swath=200.0, na=1024, pattern='sinc2', extent=None):
    """Airborne X-band defaults: nominal resolution c/(2B) = 1.5 m in range and La/2 = 0.75 m in azimuth; the PRF
    exceeds the Doppler bandwidth of the pattern's main lobe, 4 v/La = 533 Hz. The fast-time window covers zero-Doppler
    ranges r0 +- swath/2 over the simulated beam, pulse na/2 sees the swath center at beam center, and beamwidth_deg
    (one-way 3 dB), when given, sets La = 0.886 lambda/beamwidth. extent: simulated |u|, by default the main lobe of
    the sinc^2 pattern (up to its first null, so the cut is smooth) or 1.6 for the Gaussian one (-69 dB)."""
    lam = C / fc
    if beamwidth_deg is not None:
        La = 2 * _U3 * lam / np.deg2rad(beamwidth_deg)
    if extent is None:
        extent = 1.0 if pattern == 'sinc2' else 1.6
    sq = np.deg2rad(squint_deg)
    s = np.sin(sq) + np.array([-1.0, 1.0]) * extent * lam / La
    cmin = np.sqrt(1 - s ** 2).min()
    cmax = 1.0 if s[0] < 0 < s[1] else np.sqrt(1 - s ** 2).max()
    t0 = 2 * (r0 - swath / 2) / (C * cmax) - Tp / 2
    t1 = 2 * (r0 + swath / 2) / (C * cmin) + Tp / 2
    nr = 32 * int(np.ceil((t1 - t0) * fs / 32))
    eta0 = -r0 * np.tan(sq) / v - (na // 2) / prf
    return StripParams(fc, B, Tp, fs, prf, v, r0, La, sq, na, nr, t0, eta0, pattern, extent)


def pattern(p, u):
    """Two-way amplitude pattern at normalized Doppler u."""
    u = np.asarray(u, np.float64)
    if p.pattern == 'sinc2':
        return np.sinc(u) ** 2
    if p.pattern == 'gauss':
        return np.exp(-np.log(2) * (u / _U3) ** 2)
    raise ValueError(f'unknown pattern {p.pattern!r}')


def simulate(p, targets, amp=None):
    """Raw echoes [na, nr] complex128 of point targets [n, 2] = (x_n, r_n), computed per target in the time domain."""
    targets = np.atleast_2d(np.asarray(targets, np.float64))
    amp = np.ones(len(targets), np.complex128) if amp is None else np.asarray(amp, np.complex128)
    eta, tau = p.eta, p.tau
    out = np.zeros((p.na, p.nr), np.complex128)
    for (x, r), a in zip(targets, amp):
        dx = x - p.v * eta
        R = np.sqrt(r * r + dx * dx)
        u = p.La * (dx / R - np.sin(p.squint)) / p.lam
        on = np.abs(u) <= p.extent
        d = tau[None, :] - 2 * R[on, None] / C
        out[on] += np.where(np.abs(d) <= p.Tp / 2, (a * pattern(p, u[on]))[:, None]
                            * np.exp(1j * (np.pi * p.Kr * d * d - 4 * np.pi * R[on, None] / p.lam)), 0)
    return out


# ------------------------------------------------------------------ windows and filters

def _taylor_fn(sll=35.0, nbar=4):
    A = np.arccosh(10 ** (sll / 20)) / np.pi
    s2 = nbar ** 2 / (A ** 2 + (nbar - 0.5) ** 2)
    ma = np.arange(1, nbar)
    F = np.array([(-1) ** (m + 1) * np.prod(1 - m * m / s2 / (A ** 2 + (ma - 0.5) ** 2))
                  / (2 * np.prod([1 - m * m / (n * n) for n in ma if n != m])) for m in ma])

    def w(t):
        return 1 + 2 * np.sum(F[:, None] * np.cos(np.pi * ma[:, None] * np.ravel(t)[None]), 0).reshape(np.shape(t))
    return lambda t: w(t) / w(0.0)


def window(spec):
    """Window as a function of t in [-1, 1] (zero outside): None, 'taylor' (35 dB, nbar 4), ('taylor', sll, nbar),
    'kaiser' (beta 2.5), ('kaiser', beta), or a callable."""
    if spec is None or spec == 'none':
        fn = lambda t: np.ones_like(t)
    elif callable(spec):
        fn = spec
    else:
        name, *arg = (spec,) if isinstance(spec, str) else spec
        if name == 'taylor':
            fn = _taylor_fn(*arg)
        elif name == 'kaiser':
            beta = arg[0] if arg else 2.5
            fn = lambda t: np.i0(beta * np.sqrt(np.clip(1 - t * t, 0, 1))) / np.i0(beta)
        else:
            raise ValueError(f'unknown window {spec!r}')
    return lambda t: np.where(np.abs(t) <= 1, fn(np.clip(np.asarray(t, np.float64), -1, 1)), 0.0)


def _ntp(p):
    return int(np.floor(p.Tp / 2 * p.fs)) * 2 + 1


def _fast(n):
    from scipy.fft import next_fast_len
    return next_fast_len(int(n))


def matched_filter(p, n):
    """Range matched filter on the n-point FFT grid: conj(FFT(replica centered at t = 0)), unit peak gain."""
    m = np.arange(_ntp(p)) - _ntp(p) // 2
    t = m / p.fs
    rep = np.zeros(n, np.complex128)
    rep[m % n] = np.exp(1j * np.pi * p.Kr * t * t)
    return np.conj(np.fft.fft(rep)) / _ntp(p)


def range_compress(raw, p, rwin=None, dtype='float32'):
    """Matched filter and range window W_r(f/(B/2)) in the frequency domain (JAX); same fast-time grid as raw."""
    import jax.numpy as jnp
    cdt = np.complex128 if dtype == 'float64' else np.complex64
    n = _fast(raw.shape[1] + _ntp(p))
    f = np.fft.fftfreq(n, 1 / p.fs)
    H = matched_filter(p, n) * window(rwin)(f / (p.B / 2))
    with _x64(dtype):
        X = jnp.fft.fft(jnp.asarray(np.asarray(raw).astype(cdt)), n, axis=1) * jnp.asarray(H.astype(cdt))
        return np.asarray(jnp.fft.ifft(X, axis=1)[:, :raw.shape[1]])


def doppler(p, n):
    """Unwrapped Doppler frequencies of an n-point azimuth FFT: within prf/2 of the centroid."""
    f = np.fft.fftfreq(n, 1 / p.prf)
    return p.fdc + np.mod(f - p.fdc + p.prf / 2, p.prf) - p.prf / 2


def axes(p, r_ref=None, nr=None, na=None):
    """Zero-Doppler slant range of each column and along-track position of each row (x = v eta)."""
    r_ref = p.r0 if r_ref is None else r_ref
    r = C * p.t0 / 2 + np.arange(p.nr if nr is None else nr) * C / (2 * p.fs)
    x = p.v * (p.eta0 + r_ref * np.tan(p.squint) / p.v) + np.arange(p.na if na is None else na) * p.v / p.prf
    return r, x


def _sp_amp(p, f, s):
    """Stationary-phase factor that gives the frequency-domain image the scale and phase of backprojection (times
    sqrt(r)); exp(+j pi/4) undoes the phase of the azimuth chirp's spectrum, whose FM rate is negative."""
    return p.prf * np.sqrt(C / (2 * (p.fc + f) * p.v ** 2 * (1 - s * s) ** 1.5)) * np.exp(1j * np.pi / 4)


# ------------------------------------------------------------------ JAX pieces

def _x64(dtype):
    if dtype == 'float64':
        import jax
        if hasattr(jax, 'enable_x64'):
            return jax.enable_x64(True)
        from jax.experimental import enable_x64
        return enable_x64()
    if dtype != 'float32':
        raise ValueError("dtype: 'float32' or 'float64'")
    return contextlib.nullcontext()


def _interp(x, pos, taps=16, beta=None):
    """Kaiser-windowed sinc interpolation of the rows of x [m, n] (circular) at fractional indices pos [m, k]
    (float64, host); taps samples per output."""
    import jax.numpy as jnp
    h = taps // 2
    beta = 2.2 * h if beta is None else beta
    i0 = np.floor(pos)
    fr = jnp.asarray(pos - i0, x.real.dtype)
    i0 = i0.astype(np.int64)
    n = x.shape[1]
    out = 0
    for k in range(-h + 1, h + 1):
        d = fr - k
        w = jnp.sinc(d) * jnp.i0(beta * jnp.sqrt(jnp.clip(1 - (d / h) ** 2, 0, 1))) / np.i0(beta)
        out = out + jnp.take_along_axis(x, jnp.asarray((i0 + k) % n, jnp.int32), axis=1) * w
    return out


def _spectrum(data, p, n, cdt):
    import jax.numpy as jnp
    return jnp.fft.fft2(jnp.asarray(np.asarray(data).astype(cdt)), (data.shape[0], n))


# ------------------------------------------------------------------ algorithms

def omegak(data, p, compressed=False, rwin=None, awin=None, umax=1.0, r_ref=None, dtype='float32', taps=16):
    """Omega-k: 2-D FFT, matched filter and bulk compression at r_ref with the exact 2-D phase
    4 pi r_ref sqrt((fc + f_tau)^2 - (c f_eta/2v)^2)/c, Stolt interpolation along range frequency onto
    fc + f' = sqrt((fc + f_tau)^2 - (c f_eta/2v)^2) (Kaiser-windowed sinc on a twice-oversampled spectrum, the
    output band of each Doppler row demodulated by fc cos(theta)), then the windows and the Jacobian at the mapped
    frequency, and the 2-D inverse FFT. From raw echoes the range window is applied after the mapping, on a smooth
    spectrum; from range-compressed input the window's band edge is interpolated as it stands.
    -> (image [na, nr], r, x)."""
    import jax.numpy as jnp
    cdt = np.complex128 if dtype == 'float64' else np.complex64
    na, nr = data.shape
    r_ref = p.r0 if r_ref is None else r_ref
    n = _fast(2 * (nr + _ntp(p)))
    f = np.fft.fftfreq(n, 1 / p.fs)[None]
    fe = doppler(p, na)[:, None]
    a = C * fe / (2 * p.v)
    D = np.sqrt(1 - (a / p.fc) ** 2)
    r, x = axes(p, r_ref, nr, na)
    rs = C * p.t0 / 2
    H = np.exp(1j * (4 * np.pi * r_ref / C * np.sqrt((p.fc + f) ** 2 - a * a) - 2 * np.pi * f * p.t0
                     + 2 * np.pi * fe * r_ref * np.tan(p.squint) / p.v))
    if not compressed:
        H = H * matched_filter(p, n)[None]
    fin = f * (2 * p.fc * D + f) / (np.sqrt((p.fc * D + f) ** 2 + a * a) + p.fc)      # f_tau at the output grid
    s = a / (p.fc + fin)
    W = (window(rwin if not compressed else None)(fin / (p.B / 2)) * window(awin)(p.La * (s - np.sin(p.squint)) / p.lam / umax)
         * (p.fc * D + f) / (p.fc + fin) * _sp_amp(p, fin, s) * np.exp(1j * 4 * np.pi * f * (rs - r_ref) / C))
    post = np.exp(1j * 4 * np.pi * p.fc * D * (r[None] - r_ref) / C) * np.sqrt(r)[None]
    with _x64(dtype):
        X = _spectrum(data, p, n, cdt) * jnp.asarray(H.astype(cdt))
        Y = _interp(X, fin * n / p.fs, taps) * jnp.asarray(W.astype(cdt))
        Y = jnp.fft.ifft(Y, axis=1)[:, :nr] * jnp.asarray(post.astype(cdt))
        img = np.asarray(jnp.fft.ifft(Y, axis=0))
    return img, r, x


def rda(data, p, compressed=False, rwin=None, awin=None, umax=1.0, r_ref=None, src=True, dtype='float32', taps=16):
    """Range-Doppler algorithm: 2-D FFT, matched filter, windows and stationary-phase amplitude (as in omega-k),
    optional secondary range compression exp(-j pi f_tau^2/Ksrc) with Ksrc = 2 v^2 fc^3 D^3/(c r_ref f_eta^2) and
    D = cos(theta) = sqrt(1 - (c f_eta/2 v fc)^2), range IFFT to twice the sampling rate, RCMC by Kaiser-windowed
    sinc interpolation of each Doppler row at the exact migration r/D (r the output column's range), azimuth matched
    filter exp(+j 4 pi fc D r/c), azimuth IFFT. -> (image [na, nr], r, x)."""
    import jax.numpy as jnp
    cdt = np.complex128 if dtype == 'float64' else np.complex64
    na, nr = data.shape
    r_ref = p.r0 if r_ref is None else r_ref
    n = _fast(nr + _ntp(p))
    f = np.fft.fftfreq(n, 1 / p.fs)[None]
    fe = doppler(p, na)[:, None]
    a = C * fe / (2 * p.v)
    D = np.sqrt(1 - (a / p.fc) ** 2)
    s = a / (p.fc + f)
    H = (window(rwin if not compressed else None)(f / (p.B / 2)) * window(awin)(p.La * (s - np.sin(p.squint)) / p.lam / umax)
         * _sp_amp(p, f, s) * np.exp(2j * np.pi * fe * r_ref * np.tan(p.squint) / p.v))
    if not compressed:
        H = H * matched_filter(p, n)[None]
    if src:
        H = H * np.exp(-1j * np.pi * f * f * r_ref * C * fe * fe / (2 * p.v ** 2 * p.fc ** 3 * D ** 3))
    r, x = axes(p, r_ref, nr, na)
    pos = (2 * r[None] / (C * D) - p.t0) * 2 * p.fs
    post = np.exp(1j * 4 * np.pi * p.fc * D * r[None] / C) * np.sqrt(r)[None]
    with _x64(dtype):
        X = _spectrum(data, p, n, cdt) * jnp.asarray(H.astype(cdt))
        X = jnp.concatenate([X[:, :n // 2], jnp.zeros_like(X), X[:, n // 2:]], axis=1)
        Z = _interp(jnp.fft.ifft(X, axis=1) * 2, pos, taps) * jnp.asarray(post.astype(cdt))
        img = np.asarray(jnp.fft.ifft(Z, axis=0))
    return img, r, x


def backproject(data, p, x, r, compressed=False, rwin=None, awin=None, umax=1.0, up=16, taps=8):
    """Exact time-domain backprojection in float64 (numpy) at zero-Doppler points (x, r) (any equal shapes). Each
    pulse's range-compressed echo is upsampled by FFT zero padding (by `up`) and read with a Kaiser-windowed sinc,
    which reproduces the band-limited signal to about -120 dB."""
    x, r = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(r, np.float64))
    shape, x, r = x.shape, x.ravel(), r.ravel()
    na, nr = data.shape
    n = _fast(nr + _ntp(p))
    f = np.fft.fftfreq(n, 1 / p.fs)
    X = np.fft.fft(np.asarray(data, np.complex128), n, axis=1)
    if not compressed:
        X *= matched_filter(p, n) * window(rwin)(f / (p.B / 2))
    wa = window(awin)
    h = taps // 2
    k = np.arange(-h + 1, h + 1)
    out = np.zeros(x.size, np.complex128)
    for i, eta in enumerate(p.eta):
        dx = x - p.v * eta
        R = np.sqrt(r * r + dx * dx)
        u = p.La * (dx / R - np.sin(p.squint)) / p.lam
        m = np.abs(u) <= umax
        if not m.any():
            continue
        s = np.fft.ifft(np.concatenate([X[i, :n // 2], np.zeros(n * (up - 1)), X[i, n // 2:]])) * up
        pos = (2 * R[m] / C - p.t0) * up * p.fs
        i0 = np.floor(pos)
        d = (pos - i0)[:, None] - k[None]
        w = np.sinc(d) * np.i0(2.2 * h * np.sqrt(np.clip(1 - (d / h) ** 2, 0, 1))) / np.i0(2.2 * h)
        val = np.sum(s[(i0.astype(np.int64)[:, None] + k[None]) % s.size] * w, 1)
        out[m] += wa(u[m] / umax) * val * np.exp(1j * 4 * np.pi * R[m] / p.lam)
    return out.reshape(shape)


def focus_stripmap(data, p, algorithm='omegak', compressed=False, rows=None, cols=None, **kw):
    """Focus raw echoes (or range-compressed ones, compressed=True) [na, nr] with 'omegak', 'rda' or 'bp'.
    -> (complex image [rows, columns], slant range of each column (m), along-track position of each row (m)).
    Keywords: rwin, awin (see window), umax, r_ref, dtype, taps; src for RDA. For 'bp', rows and cols (index arrays
    into the full grid) restrict the output, which otherwise covers all na x nr points."""
    if algorithm == 'omegak':
        return omegak(data, p, compressed, **kw)
    if algorithm == 'rda':
        return rda(data, p, compressed, **kw)
    if algorithm != 'bp':
        raise ValueError("algorithm must be 'omegak', 'rda' or 'bp'")
    r, x = axes(p, kw.pop('r_ref', None), data.shape[1], data.shape[0])
    kw.pop('dtype', None)
    r = r if cols is None else r[cols]
    x = x if rows is None else x[rows]
    return backproject(data, p, x[:, None], r[None, :], compressed, **kw), r, x


# ------------------------------------------------------------------ theory

def irf_width(w, n=4096, pad=64):
    """3 dB width of |integral of w(t) exp(j 2 pi t y) dt| over t in [-1/2, 1/2], in units of y."""
    t = (np.arange(n) - (n - 1) / 2) / n
    P = np.abs(np.fft.fft(w(t), n * pad)) ** 2
    P = np.fft.fftshift(P) / P.max()
    c = int(np.argmax(P))
    hi = c + np.argmax(P[c:] < 0.5)
    lo = c - np.argmax(P[c::-1] < 0.5)
    yr = hi - 1 + (P[hi - 1] - 0.5) / (P[hi - 1] - P[hi])
    yl = lo + 1 - (P[lo + 1] - 0.5) / (P[lo + 1] - P[lo])
    return (yr - yl) / pad


def resolution(p, rwin=None, awin=None, umax=1.0):
    """Expected 3 dB resolution (m): slant range c/(2B) times the range window's broadening, and along-track
    (La/2)/umax times the broadening of the antenna pattern times the azimuth window over |u| <= umax (Doppler
    f - f_dc = 2 v u/La). Both are for the principal axes; a squinted zero-Doppler image is sheared."""
    rw, aw = window(rwin), window(awin)
    rho_r = C / (2 * p.B) * irf_width(lambda t: rw(2 * t))
    rho_a = p.La / (4 * umax) * irf_width(lambda t: aw(2 * t) * pattern(p, 2 * t * umax))
    return rho_r, rho_a
