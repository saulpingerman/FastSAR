"""Burst-mode SAR on a straight, constant-velocity track: ScanSAR (a fixed beam per burst) and TOPS (the beam steered
from backward to forward within each burst), with a point-target simulator, per-burst focusing by omega-k or RDA
from fastsar.stripmap, a mosaic of burst images and float64 time-domain backprojection as the reference.

    from fastsar import burst as bm
    bs = bm.make_bursts('tops', nburst=2)                  # list of BurstParams, one per burst
    raw = bm.simulate(bs[0], targets)                       # [pulses of the burst, fast-time samples]
    img, r, x = bm.focus_burst(raw, bs[0], rwin='taylor')   # zero-Doppler image of the burst

Conventions are those of fastsar.stripmap (zero-Doppler target coordinates (x_n, r_n), look angle theta positive
forward, Doppler f = 2 v sin(theta)/lambda, up-chirp, exp(-j 2 pi f t) transforms). A burst is a StripParams with
na the pulses of the burst and eta0 its first pulse, plus a steering rate: the beam points at
    psi(eta) = squint + kpsi (eta - eta_mid),        eta_mid = eta0 + (na - 1)/(2 prf),
and the two-way pattern of pulse p is G(u_p) with u_p = La (sin(theta_p) - sin(psi(eta_p)))/lambda, so that
f - f_dc(eta) = 2 v u/La with the Doppler centroid f_dc(eta) = 2 v sin(psi(eta))/lambda. kpsi = 0 is ScanSAR; kpsi > 0
(backward to forward) is TOPS, whose centroid rises at k_t = 2 v kpsi cos(psi)/lambda Hz/s, independently of range
on this geometry. The footprint then moves at v + r kpsi instead of v, each target is seen for a fraction
alpha = v/(v + r kpsi) of the stripmap dwell, and the azimuth resolution coarsens by 1/alpha. A ScanSAR target is seen
for the part of its dwell that falls inside the burst. Subswaths differ in fast-time window (r0, t0, nr) and in an
elevation gain exp(-ln 2 (2 (r_n - r0)/el_width)^2) on the target's zero-Doppler range, constant over a burst.

Focusing. Each burst is zero-padded in azimuth to cover the zero-Doppler times of every target it can illuminate,
so the circular azimuth processing of stripmap.omegak or stripmap.rda neither wraps nor ghosts. ScanSAR bursts are
then focused directly, with the azimuth window W_a(u/umax) applied in the two-dimensional spectrum as in stripmap.
TOPS bursts span a Doppler band k_t T_burst + 4 v/La that can be several times the PRF; they are deramped, upsampled
and reramped before the stripmap focuser:
  1. range FFT and deramp exp(-j (1 + f_tau/fc) phi(eta)), phi(eta) = 2 pi int_{eta_mid}^{eta} f_dc, which brings
     every instant to baseband: the deramped Doppler of a target at time eta is exactly 2 v u (1 + f_tau/fc)/La, so
     its band is that of the beam, 4 v/La < prf;
  2. azimuth FFT and the window W_a(u/umax) on that deramped Doppler (the stationary-phase counterpart of
     backprojection's per-pulse weight, as in stripmap);
  3. zero padding of the spectrum to L prf, L = ceil(span/prf + 1) with span the centroid excursion over the burst
     at the top of the range band, inverse FFT, and reramp exp(+j (1 + f_tau/fc) phi(eta)) on the fine grid. The
     steering in phi is held at its end values outside the burst, so the reramped signal stays inside the fine
     band everywhere and the fine data is alias free;
  4. omega-k or RDA at prf' = L prf with no further azimuth window, and every L-th row kept, which leaves the image
     on the pulse grid with backprojection's scale.
Images are on the zero-Doppler grid of stripmap, rows at the pulse spacing v/prf aligned with the pulse train, so
burst images of one subswath can be mosaicked row by row. They have the complex scale and phase of backprojection
of the same burst's pulses with the per-pulse weight W_a(u_p/umax), |u_p| <= umax, u_p relative to the beam of that
pulse; mosaic picks, per row, the burst that sees it with the most pattern energy.
"""
from dataclasses import dataclass, fields, replace

import numpy as np

from . import stripmap as sm
from .stripmap import C, window, _fast, _x64


@dataclass
class BurstParams(sm.StripParams):
    kpsi: float = 0.0           # steering rate (rad/s), positive from backward to forward; 0 for ScanSAR
    el_width: float = np.inf    # two-way 3 dB width of the elevation gain in zero-Doppler range (m)

    @property
    def eta_mid(self):
        return self.eta0 + (self.na - 1) / (2 * self.prf)

    def psi(self, eta=None):
        """Beam steering angle (rad) at slow times eta (default: the pulses)."""
        return self.squint + self.kpsi * ((self.eta if eta is None else np.asarray(eta)) - self.eta_mid)

    def fdc_t(self, eta=None):
        """Doppler centroid (Hz) at slow times eta, at the carrier."""
        return 2 * self.v * np.sin(self.psi(eta)) / self.lam

    @property
    def kt(self):
        """Doppler centroid rate (Hz/s) at mid-burst."""
        return 2 * self.v * self.kpsi * np.cos(self.squint) / self.lam

    def alpha(self, r):
        """Ratio of the dwell in the burst's steered beam to the stripmap dwell at zero-Doppler range r."""
        return self.v / (self.v + np.asarray(r) * self.kpsi)

    def strip(self, **kw):
        """The StripParams part (fields of a plain stripmap geometry), with replacements."""
        return replace(sm.StripParams(**{f.name: getattr(self, f.name) for f in fields(sm.StripParams)}), **kw)


def make_bursts(mode='tops', nburst=1, n=None, gap=None, kpsi=None, alpha=0.25, subswaths=None, el_width=None,
                **kw):
    """Bursts on one pulse train at the PRF; pulse n//2 of burst 0 leaves at eta = 0. Bursts cycle through the
    subswaths [(r0, swath), ...] (default one, from r0 and swath in kw); burst b starts at pulse b (n + gap).
    mode 'tops': n = 512 and kpsi = v (1/alpha - 1)/r0 of the first subswath by default; 'scansar': n = 128, kpsi = 0.
    gap: pulses between consecutive bursts, by default n for ScanSAR on one subswath and 0 otherwise. el_width (m), by default the
    swath, sets the elevation gain; inf turns it off. Other keywords go to stripmap.make_params (fc, B, Tp, fs, prf,
    v, La, beamwidth_deg, squint_deg, pattern, extent); squint_deg is the steering at mid-burst. The fast-time window
    covers each subswath over the steered beam."""
    if mode not in ('tops', 'scansar'):
        raise ValueError("mode: 'tops' or 'scansar'")
    n = (512 if mode == 'tops' else 128) if n is None else n
    r0d, swd = kw.pop('r0', 5e3), kw.pop('swath', 200.0)
    subswaths = [(r0d, swd)] if subswaths is None else list(subswaths)
    gap = (n if mode == 'scansar' and len(subswaths) == 1 else 0) if gap is None else gap
    q = sm.make_params(r0=subswaths[0][0], swath=subswaths[0][1], na=n, **kw)
    if kpsi is None:
        kpsi = q.v * (1 / alpha - 1) / q.r0 if mode == 'tops' else 0.0
    ds = np.abs(np.sin(q.squint + kpsi * (n - 1) / (2 * q.prf) * np.array([-1, 1])) - np.sin(q.squint)).max()
    eta_first = -(n // 2) / q.prf
    out = []
    for b in range(nburst):
        r0, sw = subswaths[b % len(subswaths)]
        q = sm.make_params(r0=r0, swath=sw, na=n, **dict(kw, extent=None))
        ext = q.extent if kw.get('extent') is None else kw['extent']
        w = sm.make_params(r0=r0, swath=sw, na=n, **dict(kw, extent=ext + q.La * ds / q.lam))
        out.append(BurstParams(**{f.name: getattr(w, f.name) for f in fields(sm.StripParams)} | dict(
            extent=ext, eta0=eta_first + b * (n + gap) / q.prf, kpsi=kpsi,
            el_width=sw if el_width is None else el_width)))
    return out


def elevation(bp, r):
    """Two-way elevation gain at zero-Doppler range r."""
    return np.exp(-np.log(2) * (2 * (np.asarray(r, np.float64) - bp.r0) / bp.el_width) ** 2)


def simulate(bp, targets, amp=None):
    """Raw echoes [na, nr] complex128 of point targets [n, 2] = (x_n, r_n) for one burst, exact in the time domain
    (stripmap.simulate with the beam of each pulse and the elevation gain)."""
    targets = np.atleast_2d(np.asarray(targets, np.float64))
    amp = np.ones(len(targets), np.complex128) if amp is None else np.asarray(amp, np.complex128)
    eta, tau, sp = bp.eta, bp.tau, np.sin(bp.psi())
    out = np.zeros((bp.na, bp.nr), np.complex128)
    for (x, r), a in zip(targets, amp):
        dx = x - bp.v * eta
        R = np.sqrt(r * r + dx * dx)
        u = bp.La * (dx / R - sp) / bp.lam
        on = np.abs(u) <= bp.extent
        d = tau[None, :] - 2 * R[on, None] / C
        out[on] += np.where(np.abs(d) <= bp.Tp / 2, (a * elevation(bp, r) * sm.pattern(bp, u[on]))[:, None]
                            * np.exp(1j * (np.pi * bp.Kr * d * d - 4 * np.pi * R[on, None] / bp.lam)), 0)
    return out


def backproject(data, bp, x, r, compressed=False, rwin=None, awin=None, umax=1.0, up=16, taps=8):
    """float64 time-domain backprojection of one burst at zero-Doppler points (x, r): stripmap.backproject with the
    weight W_a(u_p/umax), |u_p| <= umax, taken relative to the beam of each pulse."""
    if bp.kpsi == 0:
        return sm.backproject(data, bp, x, r, compressed, rwin, awin, umax, up, taps)
    return sum(sm.backproject(data[i:i + 1], replace(bp, na=1, eta0=e, squint=s), x, r, compressed, rwin, awin,
                              umax, up, taps) for i, (e, s) in enumerate(zip(bp.eta, bp.psi())))


def _ramp(bp, eta):
    """phi(eta) = 2 pi int_{eta_mid}^{eta} f_dc (rad) at the carrier, with the steering held at its end values
    outside the burst."""
    t = np.asarray(eta, np.float64) - bp.eta_mid
    tc = np.clip(t, -(bp.na - 1) / (2 * bp.prf), (bp.na - 1) / (2 * bp.prf))
    s, k = bp.squint, bp.kpsi
    integ = np.sin(s) * t if k == 0 else (np.cos(s) - np.cos(s + k * tc)) / k + np.sin(s + k * tc) * (t - tc)
    return 4 * np.pi * bp.v / bp.lam * integ


def _extent(bp, umax, r_ref, margin):
    """Pulses to pad before the burst and the padded length: the padded slow-time window holds the burst and the
    zero-Doppler times (as image rows) of every target the burst can illuminate within |u| <= umax."""
    e = min(umax, bp.extent) * bp.lam / bp.La
    r, _ = sm.axes(bp)
    eta = bp.eta[[0, -1]]
    s = (np.sin(bp.psi(eta))[:, None] + np.array([-e, e])[None]).ravel()
    zd = (np.repeat(eta, 2)[:, None] + np.array([r[0], r[-1]])[None] * (s / np.sqrt(1 - s * s))[:, None] / bp.v)
    off = r_ref * np.tan(bp.squint) / bp.v                        # image row 0 lies off after the first pulse
    lo = min(bp.eta0, zd.min() - off) - margin / bp.prf
    hi = max(eta[1], zd.max() - off) + margin / bp.prf
    pre = int(np.ceil((bp.eta0 - lo) * bp.prf))
    m = _fast(pre + int(np.ceil((hi - bp.eta0) * bp.prf)) + 1)
    return pre, m + m % 2


def upsampling(bp):
    """Azimuth upsampling factor of the TOPS chain: the centroid excursion over the burst at the top of the range
    band plus one PRF, in PRFs."""
    span = 2 * bp.v * (bp.fc + bp.B / 2) / C * np.ptp(np.sin(bp.psi(bp.eta[[0, -1]])))
    return int(np.ceil(span / bp.prf + 1))


def focus_burst(raw, bp, algorithm='omegak', compressed=False, rwin=None, awin=None, umax=1.0, r_ref=None,
                dtype='float32', taps=16, up=None, margin=32):
    """Focus one burst [na, nr] (raw, or range compressed with compressed=True) with 'omegak' or 'rda'.
    -> (complex image [rows, nr], slant range of each column (m), along-track position of each row (m)), rows at the
    pulse spacing covering every target the burst illuminates. up: TOPS upsampling factor (default upsampling(bp)).
    Other keywords as stripmap.focus_stripmap."""
    if algorithm not in ('omegak', 'rda'):
        raise ValueError("algorithm must be 'omegak' or 'rda'")
    alg = sm.omegak if algorithm == 'omegak' else sm.rda
    import jax.numpy as jnp
    na, nr = raw.shape
    r_ref = bp.r0 if r_ref is None else r_ref
    pre, m = _extent(bp, umax, r_ref, margin)
    e0 = bp.eta0 - pre / bp.prf
    data = np.zeros((m, nr), np.complex128)
    data[pre:pre + na] = raw
    kw = dict(r_ref=r_ref, dtype=dtype, taps=taps)
    if bp.kpsi == 0:
        return alg(data, bp.strip(na=m, eta0=e0), compressed, rwin=rwin, awin=awin, umax=umax, **kw)
    L = upsampling(bp) if up is None else up
    cdt = np.complex128 if dtype == 'float64' else np.complex64
    n = _fast(nr + 16)                                            # room for the deramp's range shift phi/(2 pi fc)
    g = 1 + np.fft.fftfreq(n, 1 / bp.fs)[None] / bp.fc
    u = bp.La * np.fft.fftfreq(m, 1 / bp.prf)[:, None] / (2 * bp.v * g)
    if 2 * bp.v * min(umax, bp.extent) * g.max() / bp.La >= bp.prf / 2:
        raise ValueError('the beam band at umax exceeds the PRF')
    W = window(awin)(u / umax) * (np.abs(u) <= min(umax, bp.extent))
    ec = e0 + np.arange(m) / bp.prf
    ef = e0 + np.arange(L * m) / (L * bp.prf)
    with _x64(dtype):
        X = jnp.fft.fft(jnp.asarray(data.astype(cdt)), n, axis=1)
        X = X * jnp.asarray(np.exp(-1j * _ramp(bp, ec)[:, None] * g).astype(cdt))
        X = jnp.fft.fft(X, axis=0) * jnp.asarray(W.astype(cdt))
        X = jnp.concatenate([X[:m // 2], jnp.zeros(((L - 1) * m, n), X.dtype), X[m // 2:]], axis=0)
        X = jnp.fft.ifft(X, axis=0)          # 1/L of the interpolated samples, which cancels the fine focuser's L prf
        X = X * jnp.asarray(np.exp(1j * _ramp(bp, ef)[:, None] * g).astype(cdt))
        fine = np.asarray(jnp.fft.ifft(X, axis=1))
    img, r, x = alg(fine, bp.strip(na=L * m, eta0=e0, prf=L * bp.prf, nr=n), compressed, rwin=rwin, awin=None,
                    umax=np.inf, **kw)
    return img[::L, :nr], r[:nr], x[::L]


def coverage(bp, x, r, awin=None, umax=1.0):
    """Per-pulse weight W_a(u/umax) G(u) of a target at (x, r) and its Doppler (Hz): [na], [na]."""
    dx = np.asarray(x, np.float64) - bp.v * bp.eta
    R = np.sqrt(r * r + dx * dx)
    u = bp.La * (dx / R - np.sin(bp.psi())) / bp.lam
    w = window(awin)(u / umax) * sm.pattern(bp, u) * (np.abs(u) <= min(umax, bp.extent))
    return w, 2 * bp.v * dx / R / bp.lam


def resolution(bp, x, r, awin=None, umax=1.0, n=4001):
    """Expected along-track 3 dB resolution (m) of a target at (x, r) from its illumination: the width of
    |sum_p w_p exp(j 2 pi f_p y/v)| over y, with w_p and f_p from coverage (0 if the burst misses it)."""
    w, f = coverage(bp, x, r, awin, umax)
    if not w.any():
        return 0.0
    y = np.linspace(-1, 1, n) * 3 * bp.v / np.ptp(f[w > 0]).clip(1e-9)
    P = np.abs(np.exp(2j * np.pi * np.outer(y, f[w > 0]) / bp.v) @ w[w > 0]) ** 2
    P /= P.max()
    c = n // 2
    a, b = c + np.argmax(P[c:] < 0.5), c - np.argmax(P[c::-1] < 0.5)
    yr = y[a - 1] + (P[a - 1] - 0.5) / (P[a - 1] - P[a]) * (y[a] - y[a - 1])
    yl = y[b + 1] - (P[b + 1] - 0.5) / (P[b + 1] - P[b]) * (y[b + 1] - y[b])
    return yr - yl


def mosaic(images, bursts, r_ref=None, umax=1.0):
    """Mosaic of burst images [(img, r, x), ...] of one subswath (same columns) on their common row grid: each row
    from the burst that illuminates it, at r_ref, with the most pattern energy.
    -> (image, r, x, index of the burst used per row, -1 where none)."""
    p = bursts[0]
    r_ref = p.r0 if r_ref is None else r_ref
    dx = p.v / p.prf
    x0 = min(x[0] for _, _, x in images)
    rows = [np.round((x - x0) / dx).astype(np.int64) for _, _, x in images]
    nrow = max(k[-1] for k in rows) + 1
    xs = x0 + np.arange(nrow) * dx
    E = np.zeros((len(images), nrow))
    for b, (bp, k) in enumerate(zip(bursts, rows)):
        E[b, k] = [np.sum(coverage(bp, xi, r_ref, None, umax)[0] ** 2) for xi in xs[k]]
    sel = np.where(E.max(0) > 0, E.argmax(0), -1)
    out = np.zeros((nrow, images[0][0].shape[1]), images[0][0].dtype)
    for b, ((img, _, _), k) in enumerate(zip(images, rows)):
        s = sel[k] == b
        out[k[s]] = img[s]
    return out, images[0][1], xs, sel
