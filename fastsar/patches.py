"""Long apertures and arbitrary tracks: the scene is cut into patches, and each patch is formed by the spotlight
factorized backprojection (ImageFormer) from the pulses that illuminate it, with the phase history re-referenced to
the patch center and gated in range to the patch.

    from fastsar import stripmap as sm, patches
    img, r, x = patches.form_stripmap(raw, p, rwin='taylor')                    # straight track, zero-Doppler grid
    fx = patches.echoes_to_fx(raw, p)                                          # any track: FX phase history
    img = patches.form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2, beam=patches.stripmap_beam(p, ant))

Phase history. form_mosaic takes the frequency-domain (FX) form of the CPHD convention with per-pulse reference
ranges, fx = dict(S [P, K], fmin, df, ref [P], band), in which a scatterer at x contributes
exp(-j 4 pi f_k (|x - a_p| - ref_p)/c) at f_k = fmin + k df; band = (f_lo, f_hi) is the occupied band. Time-domain
echoes in the conventions of fastsar.stripmap (fast time tau_k = t0 + k/fs, carrier fc removed, up-chirp) become this
form by echoes_to_fx: the range FFT of the range-compressed pulse on the n-point grid of stripmap.backproject gives
S at f = fc + f_tau over the sampled band fc +- fs/2, with ref_p = c t0/2 (the range of sample 0), so that
    sum_k S[p, k] exp(+j 4 pi f_k (R - ref_p)/c) = s_rc(2R/c) exp(+j 4 pi R/lambda),
the per-pulse term of stripmap.backproject, s_rc read by band-limited (periodic) interpolation.

Patches. The output grid is planar: pixel (i, j) at origin + i spx e1 + j spy e2 (origin is pixel (0, 0), not the
center as in form_image). It is cut into tiles of patch = (mx, my) pixels; each is formed as a patch of
(mx + 2 crop, my + 2 crop) pixels centered on the tile, and crop pixels are dropped on each side. For a patch with
center c the pulses are those that illuminate any of 9 x 9 points spread over the patch (|u| <= umax for the
normalized azimuth coordinate u that beam returns; all pulses without a beam, or a given range), taken as one
contiguous run; the antenna path becomes ant - c and each pulse is re-referenced from ref_p to |a_p - c| (a delay
shift, a linear phase in f, as io.rereference). The range profile of each pulse (the inverse DFT along f) is then
kept over a gate of +-(largest |x - a_p| - |a_p - c| over the patch and its pulses, plus margin) about the center
and transformed back on the coarser frequency grid df' = c/(2 x gate length), so that the unambiguous range covers
the patch and the margins. Before the gate the frequency axis is extended with zeros to band +- guard where needed,
and after it the samples are weighted by a raised-cosine taper that is 1 over the band and falls to zero across the
guard. The data are band-limited to the band, so the taper does not change the image, but it makes the
reconstruction kernel short: what the gate drops reaches the patch through the kernel's tail at the margin, below
-60 dB for 25 m of margin with a 12.5 MHz guard. The default margin is 3 c/(2 guard).

Azimuth weighting. stripmap.backproject weights pulse p at pixel x by W_a(u_p(x)/umax) for |u_p(x)| <= umax, where u
is the normalized Doppler of the two-way pattern. Without an azimuth window every pulse of a patch has weight 1; with
umax at the simulated extent of the pattern (the first null of the sinc^2 pattern, the default) the per-pixel mask
then only drops pulses that hold other scatterers' returns near the pattern null, and the two agree to about -65 dB.
With a window the weight W_a(u_p(x)/umax) (the window clipped at its edges, no mask) is applied in the final stage of
one FFBP per patch (ImageFormer's aperture_weight: each final subaperture's mean weight at each final tile, with its
first-order variation across the tile). With exact=True or FASTSAR_WEIGHT_TERMS=1 the weight on the patch's pulses
and 9 x 9 points is instead split by its SVD into separable terms a_t(p) b_t(x); each term is one FFBP (or exact
backprojection) of the phase history weighted by a_t, multiplied by b_t interpolated to the pixels. Three or four
terms reach -50 dB on 40 m patches of the airborne scene in tests/test_patches.py.

Factorized backprojection. Each patch is an ImageFormer with window=False (its Taylor windows are not used; the
range window is applied in range compression, as in stripmap). The final tile size T and an optional finer grid
(tile_plan) are chosen from the predicted final-stage error: the C++ and CUDA final stages take T = 16 or 32 only,
so where T = 16 misses target_db the patch is formed on a grid finer by 2 in one or both axes and decimated, which
halves the final tiles; the JAX program also takes T = 8.
"""
import os
import threading

import numpy as np

from . import stripmap as sm
from .memory import _env_number

C = 299792458.0


def echoes_to_fx(data, p, compressed=False, rwin=None):
    """Raw (or range-compressed) echoes [P, nr] with the fast-time grid and chirp of StripParams p ->
    dict(S [P, n] complex128, fmin, df, ref [P], band) on the n-point range FFT grid of stripmap.backproject:
    matched filter and range window W_r(f_tau/(B/2)) applied unless compressed, ref = c t0/2, band fc +- B/2."""
    data = np.asarray(data)
    P, nr = data.shape
    n = sm._fast(nr + sm._ntp(p))
    f = np.fft.fftfreq(n, 1 / p.fs)
    X = np.fft.fft(data.astype(np.complex128), n, axis=1)
    if not compressed:
        X *= sm.matched_filter(p, n) * sm.window(rwin)(f / (p.B / 2))
    # sample k of the shifted spectrum sits at f_tau = (k - n//2) fs/n; with ref = c t0/2 the fast-time phase
    # exp(-j 2 pi f_tau t0) and the reference exp(+j 4 pi f ref/c) leave exp(+j 2 pi fc t0)
    S = np.fft.fftshift(X, axes=1) * (np.exp(2j * np.pi * p.fc * p.t0) / n)
    return dict(S=S, fmin=p.fc - (n // 2) * p.fs / n, df=p.fs / n, ref=np.full(P, C * p.t0 / 2),
                band=(p.fc - p.B / 2, p.fc + p.B / 2))


def stripmap_beam(p, ant, direction=(1.0, 0.0, 0.0)):
    """Normalized azimuth coordinate of the pattern of StripParams p for antenna positions ant [P, 3] (any track):
    u = La (sin(theta) - sin(theta_s))/lambda with sin(theta) = d.(x - a)/|x - a|, d the unit along-track direction
    of the antenna (one vector, or one per pulse [P, 3]). -> beam(idx, points [m, 3]) -> u [len(idx), m] for the
    pulses idx (indices or a slice)."""
    ant = np.asarray(ant, np.float64)
    d = np.asarray(direction, np.float64)
    d = np.broadcast_to(d / np.linalg.norm(d, axis=-1, keepdims=True), ant.shape)

    def beam(idx, pts):
        w = np.asarray(pts, np.float64)[None, :, :] - ant[idx][:, None, :]
        return p.La * ((w * d[idx][:, None, :]).sum(-1) / np.linalg.norm(w, axis=-1) - np.sin(p.squint)) / p.lam
    return beam


def straight_track(p, height=0.0):
    """Antenna positions [na, 3] of the straight track of stripmap: (v eta_p, 0, height). With height 0 a target
    at zero-Doppler coordinates (x, r) sits at (x, r, 0), and the (x, r) grid is the plane z = 0."""
    return np.stack([p.v * p.eta, np.zeros(p.na), np.full(p.na, float(height))], 1)


def simulate(p, ant, targets, amp=None, beam=None):
    """Raw echoes [na, nr] complex128 of point targets [n, 3] for antenna positions ant [na, 3] (any track): the
    signal model of stripmap.simulate with R = |x_n - a_p| and the pattern at u = beam(pulses, x_n) (default
    stripmap_beam(p, ant))."""
    targets = np.atleast_2d(np.asarray(targets, np.float64))
    ant = np.asarray(ant, np.float64)
    amp = np.ones(len(targets), np.complex128) if amp is None else np.asarray(amp, np.complex128)
    beam = stripmap_beam(p, ant) if beam is None else beam
    tau = p.tau
    out = np.zeros((len(ant), p.nr), np.complex128)
    for x, a in zip(targets, amp):
        R = np.linalg.norm(x[None] - ant, axis=1)
        u = beam(slice(None), x[None])[:, 0]
        on = np.abs(u) <= p.extent
        d = tau[None, :] - 2 * R[on, None] / C
        out[on] += np.where(np.abs(d) <= p.Tp / 2, (a * sm.pattern(p, u[on]))[:, None]
                            * np.exp(1j * (np.pi * p.Kr * d * d - 4 * np.pi * R[on, None] / p.lam)), 0)
    return out


def backproject(data, p, ant, points, compressed=False, rwin=None, awin=None, umax=1.0, beam=None, up=16, taps=8):
    """stripmap.backproject on any track: exact time-domain backprojection in float64 at points [..., 3], with
    R = |x - a_p| and u = beam(pulses, x) (default stripmap_beam(p, ant)). The float64 reference for form_mosaic."""
    pts = np.asarray(points, np.float64)
    shape, pts = pts.shape[:-1], pts.reshape(-1, 3)
    ant = np.asarray(ant, np.float64)
    beam = stripmap_beam(p, ant) if beam is None else beam
    na, nr = data.shape
    n = sm._fast(nr + sm._ntp(p))
    f = np.fft.fftfreq(n, 1 / p.fs)
    X = np.fft.fft(np.asarray(data, np.complex128), n, axis=1)
    if not compressed:
        X *= sm.matched_filter(p, n) * sm.window(rwin)(f / (p.B / 2))
    wa = sm.window(awin)
    h = taps // 2
    k = np.arange(-h + 1, h + 1)
    out = np.zeros(len(pts), np.complex128)
    for i in range(na):
        R = np.linalg.norm(pts - ant[i], axis=1)
        u = beam(slice(i, i + 1), pts)[0]
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


def _taper(f, band, guard):
    """1 over band, raised cosine to 0 across guard on either side."""
    t = np.clip(np.maximum(band[0] - f, f - band[1]) / guard, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * t))


def _levels(n, T, most=3):
    """Most levels (up to `most`) whose smallest split, 2 per level, does not pad n pixels beyond ceil(n/T) tiles."""
    need = -(-n // T)
    return max(1, min(most, int(np.floor(np.log2(max(need, 1))))))


SUBS = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2), (4, 4))


def tile_plan(ant, fmax, px, py, spx, spy, e1, e2, target_db=-40.0, tiles=(32, 16), subs=SUBS):
    """Cheapest final tile size T and sub-grid factors (s1, s2) whose predicted final-stage error
    (ffbp2.final_phase_error) meets target_db; the patch is then formed on the grid of spacings spx/s1, spy/s2 and
    every s-th pixel kept, which shrinks the final tiles by s at s1 s2 times the pixels. -> (T, (s1, s2), error dB);
    the smallest error found when none meets the target."""
    from . import ffbp2
    best = None
    for s1, s2 in subs:
        for T in tiles:
            err = ffbp2.final_phase_error(ant, fmax, px * s1, py * s2, spx / s1, spy / s2, e1, e2, T)
            if err <= target_db:
                return T, (s1, s2), float(err)
            if best is None or err < best[2]:
                best = (T, (s1, s2), float(err))
    return best


def _window_start(lo, hi, rows):
    """First pulse of a GPU window of `rows` profiles for a patch serving pulses lo:hi: an eighth of the window
    behind lo (patches run forward along track, a few out of order on the prefetch threads), always covering lo:hi
    when it fits."""
    return max(0, min(lo, max(hi - rows, lo - rows // 8)))


_ROWS_POOL = []
_ROWS_LOCK = threading.Lock()


def _rows(fn, n, chunk=256):
    """fn(slice) over row chunks of n rows on a thread pool shared by all calls (NumPy releases the GIL in its
    ufuncs; starting a pool per call cost 50 ms per mosaic patch)."""
    from concurrent.futures import ThreadPoolExecutor
    import os
    sl = [slice(i, min(n, i + chunk)) for i in range(0, n, chunk)]
    if len(sl) == 1:
        fn(sl[0])
        return
    with _ROWS_LOCK:
        if not _ROWS_POOL:
            _ROWS_POOL.append(ThreadPoolExecutor(os.cpu_count() or 1, thread_name_prefix='fastsar-rows'))
    list(_ROWS_POOL[0].map(fn, sl))


def _padding(fx, guard=None):
    """Frequency padding of fx to band +- guard: (guard, npl, K2, f0) with K2 a fast FFT length and f0 = fmin - npl df."""
    S, fmin, df = fx['S'], float(fx['fmin']), float(fx['df'])
    K = S.shape[1]
    band = fx.get('band') or (fmin, fmin + (K - 1) * df)
    room = min(band[0] - fmin, fmin + (K - 1) * df - band[1])
    guard = max(room, 0.125 * (band[1] - band[0])) if guard is None else float(guard)
    npl = max(0, int(np.ceil((fmin - band[0] + guard) / df)))
    npu = max(0, int(np.ceil((band[1] + guard - fmin - (K - 1) * df) / df)))
    from scipy.fft import next_fast_len
    return guard, band, npl, next_fast_len(K + npl + npu), fmin - npl * df


def range_profiles(fx, guard=None):
    """Range profiles of the whole phase history, shared by the patches of a mosaic: the inverse FFT of each pulse
    zero-padded to band +- guard (_padding), complex64 [P, K2], range bin c / (2 K2 df), relative to the pulse's
    reference range. patch_history(..., prof=range_profiles(fx)) then gates each patch from these instead of
    transforming the full history again (a long spotlight is P K2 ~ 4e9 samples per patch)."""
    import scipy.fft
    S = fx['S']
    P, K = S.shape
    guard, band, npl, K2, f0 = _padding(fx, guard)
    Q = np.empty((P, K2), np.complex64)

    def rows(sl):
        z = np.zeros((sl.stop - sl.start, K2), np.complex64)
        z[:, npl:npl + K] = S[sl]
        Q[sl] = scipy.fft.ifft(z, axis=1, workers=1)
    _rows(rows, P, chunk=512)
    return dict(Q=Q, guard=guard, K2=K2, npl=npl, f0=f0)


def _gate_bins(a, rc, center, pts, margin, K2, df):
    """Range bins (of c / (2 K2 df)) a patch's gate keeps: the points' extent in range from the antenna positions a
    [n, 3] (rc their distances to center) plus margin on each side, an even fast FFT length."""
    # range changes smoothly along the track: up to 513 evenly spaced pulses (the ends included) set the extent
    k = np.unique(np.linspace(0, len(a) - 1, min(len(a), 513)).astype(np.int64))
    dR = np.linalg.norm(pts[None, :, :] - a[k, None, :], axis=2) - rc[k, None]
    half = np.abs(dR).max() + margin
    from scipy.fft import next_fast_len
    N = next_fast_len(2 * int(np.ceil(half * 2 * K2 * df / C)))
    return N + N % 2


def gate_bins(fx, ant, center, pts, lo, hi, margin=None, guard=None):
    """The range gate patch_history would keep for this patch, in bins, before any widening: -> (N, K2)."""
    guard, band, npl, K2, f0 = _padding(fx, guard)
    margin = 3 * C / (2 * guard) if margin is None else float(margin)
    a = np.asarray(ant[lo:hi], np.float64)
    return _gate_bins(a, np.linalg.norm(a - center, axis=1), center, pts, margin, K2, float(fx['df'])), K2


def patch_history(fx, ant, center, pts, lo, hi, margin=None, guard=None, prof=None, gate_len=None):
    """Phase history of one patch: pulses lo:hi of fx re-referenced to center [3] and gated in range to the extent
    of the points pts [m, 3] (the patch outline) plus margin (m), with the guard taper. prof: range_profiles(fx, guard)
    computed once for all patches (faster; the same result up to the gate's edge samples). gate_len: maps the gate's
    length in range bins to a length at least as long (a wider gate), so that patches share frequency samples.
    -> (S [hi - lo, K'] complex128, ant - center, fmin', df')."""
    S, df = fx['S'], float(fx['df'])
    K = S.shape[1]
    guard, band, npl, K2, f0 = _padding(fx, guard)
    margin = 3 * C / (2 * guard) if margin is None else float(margin)
    f = f0 + df * np.arange(K2)
    a = np.asarray(ant[lo:hi], np.float64)
    rc = np.linalg.norm(a - center, axis=1)
    shift = np.asarray(fx['ref'], np.float64)[lo:hi] - rc
    N = _gate_bins(a, rc, center, pts, margin, K2, df)
    if gate_len is not None and N < K2:
        N = min(gate_len(N), K2)
    gate = N < K2
    df2 = K2 * df / N if gate else df
    f2 = f0 + df2 * np.arange(N if gate else K2)
    T = _taper(f2, band, guard)
    keep = np.nonzero(T > 0)[0]
    k0, k1 = int(keep[0]), int(keep[-1]) + 1
    out = np.empty((hi - lo, k1 - k0), np.complex128)
    j = (np.arange(N) + N // 2) % N - N // 2

    if gate and prof is not None and prof.get('device') is not None:
        # the same on the GPU (cuda backend): profiles in device memory, or gathered on the host and uploaded
        import cupy as cp
        tm = _Timer('patch_history_gpu')
        Q = prof['Qd'] if prof.get('Qd') is not None else prof['Q']
        sft = 2 * df * K2 * shift / C
        m = np.rint(sft).astype(np.int64)
        tm('host geometry')
        win = prof.get('win')
        with prof['lock']:               # patches are prepared on several threads; the window is shared, and the
            # GPU work takes turns with the patch forming (form_mosaic), whose groups are sized to the memory left
            if isinstance(Q, np.ndarray) and win is not None and hi - lo <= win['rows']:
                # a window of the profiles on the GPU, moved along the strip when a patch's pulses leave it (patches
                # run along track, so a window serves many of them; it starts an eighth of its length behind the
                # patch, for patches prepared slightly out of order)
                if not (win['w0'] <= lo and hi <= win['w1']):
                    win['buf'] = None
                    w0 = _window_start(lo, hi, win['rows'])
                    win['w0'], win['w1'] = w0, min(Q.shape[0], w0 + win['rows'])
                    win['buf'] = cp.asarray(Q[win['w0']:win['w1']])
                    win['loads'] = win.get('loads', 0) + 1
                Q, lo_q = win['buf'], lo - win['w0']
            else:
                lo_q = lo
            if isinstance(Q, np.ndarray):
                u = cp.asarray(Q[lo + np.arange(hi - lo)[:, None], (j[None, :] - m[:, None]) % K2])
            else:
                md = cp.asarray(m)
                cols = (cp.asarray(j)[None, :] - md[:, None]) % K2
                u = Q[cp.arange(lo_q, lo_q + hi - lo)[:, None], cols]
            tm('gather')
            U = cp.fft.fft(u, axis=1)[:, k0:k1]
            tm('fft')
            kk = cp.arange(k0, k1, dtype=cp.float32)
            dd = cp.asarray((sft - m).astype(np.float32))
            cph = cp.asarray((np.exp(-4j * np.pi * f0 * shift / C) * (K2 / N)).astype(np.complex64))
            ramp = cp.exp((-2j * np.pi / N) * dd[:, None] * kk[None, :]).astype(cp.complex64)
            out = U * ramp * (cph[:, None] * cp.asarray(T[k0:k1].astype(np.float32))[None, :])
            tm('ramp')
            return out.astype(cp.complex64), a - center, float(f2[k0]), float(df2)

    if gate and prof is not None:
        # exp(-j 4 pi f shift / c) with f = f0 + k df is a delay of s = 2 df K2 shift / c bins of the profile times the
        # constant exp(-j 4 pi f0 shift / c): the integer part m selects the gate's bins, the fraction d is a phase
        # ramp on the gated spectrum (frequencies f0 + k df2, k = 0 .. N - 1)
        Q = prof['Q']
        s = 2 * df * K2 * shift / C
        m = np.rint(s).astype(np.int64); d = s - m
        cph = np.exp(-4j * np.pi * f0 * shift / C) * (K2 / N)
        kk = np.arange(k0, k1)

        def rows(sl):
            import scipy.fft
            u = Q[lo + sl.start + np.arange(sl.stop - sl.start)[:, None], (j[None, :] - m[sl, None]) % K2]
            U = scipy.fft.fft(u, axis=1, workers=1)[:, k0:k1]
            ramp = np.exp((-2j * np.pi / N) * d[sl, None] * kk[None, :])
            out[sl] = U * (ramp * (cph[sl, None] * T[None, k0:k1]))
        _rows(rows, hi - lo)
        return out, a - center, float(f2[k0]), float(df2)

    def rows(sl):        # pulse blocks, so only the gated history is held whole (a long spotlight is tens of GB at K2)
        import scipy.fft
        S2 = np.zeros((sl.stop - sl.start, K2), np.complex128)
        S2[:, npl:npl + K] = S[lo + sl.start:lo + sl.stop]
        S2 *= np.exp(-4j * np.pi * f[None, :] / C * shift[sl, None])
        if gate:
            q = scipy.fft.ifft(S2, axis=1, workers=1)
            S2 = scipy.fft.fft(q[:, j % K2], axis=1, workers=1) * (K2 / N)          # the sum over N samples, not K2
        out[sl] = S2[:, k0:k1] * T[k0:k1][None, :]
    _rows(rows, hi - lo)
    return out, a - center, float(f2[k0]), float(df2)


def weight_terms(W, wtol_db=-50.0, most=8):
    """Separable approximation of per-pixel pulse weights W [P, m] (pulses x sample points): W ~ sum_t a_t b_t^T,
    from the SVD, keeping the terms whose singular value is within wtol_db of the largest (at most `most`).
    -> (a [t, P], b [t, m])."""
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    n = max(1, min(most, int(np.sum(s >= s[0] * 10 ** (wtol_db / 20)))))
    return U[:, :n].T, s[:n, None] * Vt[:n]


class _Timer:
    """FASTSAR_TIMING=1: accumulated wall time per step (GPU synchronized), printed by report_timing()."""
    acc = {}

    def __init__(self, name):
        import time
        self.on = os.environ.get('FASTSAR_TIMING') == '1'
        self.name, self.time = name, time
        if self.on:
            self._sync(); self.t = time.perf_counter()

    def _sync(self):
        try:
            import cupy
            cupy.cuda.Stream.null.synchronize()
        except Exception:
            pass

    def __call__(self, step):
        if self.on:
            self._sync(); t = self.time.perf_counter()
            k = f'{self.name}: {step}'
            _Timer.acc[k] = _Timer.acc.get(k, 0.0) + t - self.t
            self.t = t


def report_timing():
    """The accumulated FASTSAR_TIMING steps, as text."""
    return '\n'.join(f'{v:9.2f} s  {k}' for k, v in sorted(_Timer.acc.items(), key=lambda x: -x[1]))


def _xp(a):
    """numpy, or cupy for a device array."""
    if type(a).__module__.startswith('cupy'):
        import cupy
        return cupy
    return np


def beam_span(beam, P, pts, umax, step=64):
    """Pulses [lo, hi) for which |beam(p, x)| <= umax at some of the points pts, or None: the beam evaluated on every
    step-th pulse, then on every pulse only next to the first and last pulses found (the illuminated pulses of a point
    are contiguous)."""
    def on(idx):
        return (np.abs(beam(idx, pts)) <= umax).any(1)
    coarse = np.unique(np.r_[np.arange(0, P, step), P - 1])
    hit = np.nonzero(on(coarse))[0]
    if len(hit) == 0:
        full = np.nonzero(on(np.arange(P)))[0]
        return None if len(full) == 0 else (int(full[0]), int(full[-1]) + 1)
    a0 = int(coarse[hit[0] - 1]) if hit[0] > 0 else 0
    a = np.arange(a0, int(coarse[hit[0]]) + 1)
    lo = int(a[np.nonzero(on(a))[0][0]])
    b1 = int(coarse[hit[-1] + 1]) if hit[-1] + 1 < len(coarse) else P - 1
    b = np.arange(int(coarse[hit[-1]]), b1 + 1)
    hi = int(b[np.nonzero(on(b))[0][-1]]) + 1
    return lo, hi


def fill_gaps(a):
    """Dropped pulses: where the step between consecutive antenna positions a [n, 3] is m >= 2 times the median step,
    m - 1 positions are inserted on the line between them. -> (positions [n', 3], index [n] of each original pulse
    among them). Factorized backprojection filters along the pulse axis assuming near-even spacing; zero pulses at
    the inserted positions keep the spacing even and add nothing to the image."""
    a = np.asarray(a, np.float64)
    step = np.linalg.norm(np.diff(a, axis=0), axis=1)
    med = float(np.median(step)) if len(step) else 0.0
    if not med > 0:                       # one position, or a platform that does not move: no spacing to keep
        return a, np.arange(len(a))
    m = np.maximum(np.rint(step / med).astype(np.int64), 1)
    if m.max() < 2:
        return a, np.arange(len(a))
    idx = np.concatenate([[0], np.cumsum(m)])
    out = np.empty((idx[-1] + 1, 3))
    for k in np.nonzero(m >= 2)[0]:
        f = np.arange(1, m[k])[:, None] / m[k]
        out[idx[k] + 1:idx[k + 1]] = (1 - f) * a[k] + f * a[k + 1]
    out[idx] = a
    return out, idx


def choose_buckets(counts, compile_cost, step=256):
    """Pulse counts for the compiled programs of a JAX or TPU mosaic: counts [n] (the patches' pulse counts) ->
    sorted bucket counts (multiples of step), each patch to take the smallest bucket at least its count. Minimizes
    compile_cost x (number of buckets) + the patches' padding work, sum of (bucket / count - 1), both in units of one
    patch's formation (dynamic programming over the distinct counts; counts below 1 are ignored)."""
    c = np.asarray(counts, np.int64).ravel()
    u, w = np.unique(step * -(-c[c > 0] // step), return_counts=True)
    m = len(u)
    if m == 0:
        return []
    # padding of the counts i .. j to u[j]: u[j] (A[j+1] - A[i]) - (W[j+1] - W[i]) with prefix sums A of w / u and W of w
    A = np.concatenate([[0.0], np.cumsum(w / u)])
    W = np.concatenate([[0.0], np.cumsum(w.astype(np.float64))])
    before = np.zeros(m + 1)               # before[i]: the best cost of the counts below i (0 for none)
    prev = np.empty(m, np.int64)
    for j in range(m):
        cost = compile_cost + u[j] * (A[j + 1] - A[:j + 1]) - (W[j + 1] - W[:j + 1]) + before[:j + 1]
        i = int(np.argmin(cost))
        before[j + 1], prev[j] = cost[i], i - 1
    out, j = [], m - 1
    while j >= 0:
        out.append(int(u[j]))
        j = prev[j]
    return out[::-1]


def form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0), patch=(128, 128), crop=0,
                beam=None, umax=1.0, awin=None, pulses=None, margin=None, guard=None, backend='cpu', T='auto',
                levels='auto', target_db=-40.0, sub='auto', wtol_db=None, exact=False, info=None):
    """Complex image [nx, ny] (complex64) on the grid origin + i spx e1 + j spy e2 from the FX phase history fx
    (see echoes_to_fx and the module docstring) and the antenna positions ant [P, 3] of any track, formed patch by
    patch with ImageFormer on backend ('cpu', 'jax', 'cuda', 'tpu').
    beam(idx, points) -> u [len(idx), m]: normalized azimuth coordinate of pulses idx (stripmap_beam); a pulse
    serves a patch when |u| <= umax somewhere on it. With an azimuth window awin (stripmap.window) pulse p is
    weighted at pixel x by W_a(u_p(x)/umax), the window clipped at its edges. The weight is applied in FFBP's final
    stage (ImageFormer's aperture_weight: each final subaperture's mean weight at each final tile, with its
    first-order variation across the tile). With exact=True or FASTSAR_WEIGHT_TERMS=1, the weight on pulses x 9 x 9
    points of the patch is split into separable terms
    (weight_terms, down to wtol_db, default target_db - 10), each formed by one FFBP of the weighted phase history
    and multiplied by its pixel factor, interpolated by bicubic splines.
    pulses: (lo, hi) or a function of the patch center returning (lo, hi), used when beam is None (default: all).
    beam, pulses and the window are evaluated on the prefetch worker threads, several at once when
    FASTSAR_MOSAIC_PREFETCH is above 1 (the default on a GPU or TPU), so callables must be thread-safe (a pure NumPy
    function is; one reading a file through a shared handle may not be) or FASTSAR_MOSAIC_PREFETCH set to 0 or 1.
    margin, guard: range gate margin (m) and spectral guard (Hz), see patch_history. T, target_db: as for ImageFormer,
    with sub (s1, s2) or 'auto' chosen with T by tile_plan (T = 8 only on 'jax'; the C++ and CUDA final stages take 16
    or 32). levels: 'auto' takes up to 3, as many as the patch allows without padding. exact=True forms each patch by
    exact backprojection (fastsar.backproject, cpu) instead, to separate FFBP's error from the patching.
    info: a list to which a dict per patch is appended (center, pulse range, K, T, sub, levels, predicted error,
    number of weight terms)."""
    from scipy.interpolate import RectBivariateSpline
    from .api import ImageFormer, _backend, _check_history, _check_positions, _check_grid
    backend = _backend(backend)
    missing = [k for k in ('S', 'fmin', 'df', 'ref') if k not in fx]
    if missing:
        raise ValueError(f'fx lacks {missing}: a dict(S, fmin, df, ref[, band]) as echoes_to_fx returns')
    _check_history(fx['S'], len(_check_positions(ant)))
    if np.shape(fx['ref']) != (len(ant),):
        raise ValueError(f"fx['ref'] must hold one reference range per pulse, shape ({len(ant)},), got {np.shape(fx['ref'])}")
    nx, ny, e1, e2 = _check_grid(nx, ny, spx, spy, e1, e2)
    if len(patch) != 2 or min(patch) < 1 or crop < 0:
        raise ValueError(f'patch must be two positive pixel counts and crop >= 0, got patch={patch!r}, crop={crop!r}')
    tiles = ((32, 16, 8) if backend == 'jax' else (32, 16)) if T == 'auto' else (T,)
    subs = SUBS if sub == 'auto' else (tuple(sub),)
    wtol_db = target_db - 10.0 if wtol_db is None else wtol_db
    ant = np.asarray(ant, np.float64)
    o = np.asarray(origin, np.float64)
    mx, my = (int(v) for v in patch)
    px, py = mx + 2 * crop, my + 2 * crop
    wa = sm.window(awin)
    si, sj = np.linspace(0, px - 1, 9), np.linspace(0, py - 1, 9)          # sample points, in patch pixels
    gi, gj = np.meshgrid(si - px / 2, sj - py / 2, indexing='ij')
    # the azimuth window in the final stage of a single FFBP; FASTSAR_WEIGHT_TERMS=1 and exact=True use the separable
    # terms, one FFBP (or exact backprojection) each
    in_kernel = not exact and os.environ.get('FASTSAR_WEIGHT_TERMS', '0') != '1'
    out = np.zeros((nx, ny), np.complex64)
    npatch = -(-nx // mx) * -(-ny // my)
    prof = range_profiles(fx, guard) if npatch > 1 and os.environ.get('FASTSAR_SHARED_PROFILES', '1') != '0' else None
    if prof is not None and backend == 'cuda' and not exact:
        import cupy as cp
        prof['device'], prof['lock'] = True, threading.Lock()
        from .memory import cuda_free
        free = cuda_free()
        row = prof['Q'].shape[1] * 8
        force = _env_number('FASTSAR_PROFILE_WINDOW_ROWS', 0) or 0              # testing: the window, of this many rows
        if prof['Q'].nbytes < 0.4 * free and not force:      # profiles in GPU memory when they fit
            prof['Qd'] = cp.asarray(prof['Q'])
        else:                                  # else a window of them, a quarter of the free memory
            prof['win'] = dict(w0=0, w1=0, buf=None, rows=force or int(0.25 * free // row), free=free, row=row)
    # JAX and TPU: each pulse count compiles a program. The first count is padded to a multiple of 256; a patch
    # reuses any count already taken that exceeds its own by no more than FASTSAR_PULSE_SLACK (default 0.04), and
    # otherwise takes the next count on the grid first count x (1 + slack)^k (a sliding spotlight's patches see
    # different spans). The grid bounds the programs by the spread of the spans, whatever order patches arrive in
    slack = _env_number('FASTSAR_PULSE_SLACK', 0.0, float)
    slack = 0.04 if slack is None else slack
    taken, gates, taken_lock = [], [], threading.Lock()

    planned_gates = []                    # gate lengths chosen from every patch's gate before forming

    def padded_gate(n):
        """A range gate of n bins widened onto the planned lengths, or as padded_pulses pads pulses (same slack and
        grid), to an even fast FFT length: patches of one gate length share their frequency samples and so their
        compiled program."""
        from scipy.fft import next_fast_len
        with taken_lock:
            fit = [v for v in planned_gates if v >= n]
            if fit:
                return min(fit)
            fit = [v for v in gates if n <= v <= n * (1 + slack) + 16]
            if fit:
                return min(fit)
            v = n
            if gates and slack > 0:
                k = np.ceil(np.log(n / gates[0]) / np.log1p(slack) - 1e-9)
                v = max(n, int(np.ceil(gates[0] * (1 + slack) ** k)))
            v = next_fast_len(v)
            while v % 2:
                v = next_fast_len(v + 1)
            gates.append(v)
            return v

    planned = []                          # bucket counts chosen from every patch's span before forming (choose_buckets)

    def padded_pulses(n):
        with taken_lock:
            fit = [v for v in planned if v >= n]
            if fit:
                return min(fit)
            fit = [v for v in taken if n <= v <= n * (1 + slack) + 256]
            if fit:
                return min(fit)
            if taken and slack > 0:
                k = np.ceil(np.log(n / taken[0]) / np.log1p(slack) - 1e-9)
                v = max(n, int(np.ceil(taken[0] * (1 + slack) ** k)))
            else:
                v = n
            v = 256 * -(-v // 256)
            taken.append(v)
            return v

    def span_of(i0, j0):
        """Center, points and pulse span (lo, hi) of patch (i0, j0); span None when no pulse serves it."""
        c = o + (i0 - crop + px / 2) * spx * e1 + (j0 - crop + py / 2) * spy * e2
        pts = c + (gi.ravel() * spx)[:, None] * e1 + (gj.ravel() * spy)[:, None] * e2
        if beam is not None:
            return c, pts, beam_span(beam, len(ant), pts, umax)
        return c, pts, ((0, len(ant)) if pulses is None else (pulses(c) if callable(pulses) else pulses))

    def prep(i0, j0, span=None):
        """Everything for patch (i0, j0) up to its former: -> dict, or None when no pulse serves it."""
        tm = _Timer('form_mosaic')
        c, pts, span = span_of(i0, j0) if span is None else span
        if span is None:
            return None
        lo, hi = span
        wt = None
        if beam is not None and awin is not None and not in_kernel:
            wt = weight_terms(wa(np.clip(beam(slice(lo, hi), pts) / umax, -1, 1)), wtol_db)
        tm('beam span')
        S, a, f0, df = patch_history(fx, ant, c, pts, lo, hi, margin, guard, prof,
                                     gate_len=padded_gate if backend in ('jax', 'tpu') and not exact else None)
        tm('patch history')
        if exact:
            from .bp import backproject as bpx, plane_points
            xyz = plane_points(px, py, spx, spy, e1, e2)
            form = lambda S_: bpx(S_, a, f0, df, xyz, backend='cpu', window=False, upsample=16)
            Tp, s12, nlev, err = None, None, None, None
        else:
            af, idx = fill_gaps(a)
            if backend in ('jax', 'tpu'):         # pulses padded (padded_pulses) so that patches share compiled
                # programs: zero pulses continuing the track, half before and half after (the final stage centres
                # its plane-wave model on the mean over the aperture)
                extra = padded_pulses(len(af)) - len(af)
                if extra:
                    e0, e1_ = extra // 2, extra - extra // 2
                    af = np.concatenate([af[0] - (af[1] - af[0]) * np.arange(e0, 0, -1)[:, None], af,
                                         af[-1] + (af[-1] - af[-2]) * np.arange(1, e1_ + 1)[:, None]])
                    idx = idx + e0
            Tp, s12, err = tile_plan(af, f0 + S.shape[1] * df, px, py, spx, spy, e1, e2, target_db, tiles, subs)
            s1, s2 = s12
            nlev = _levels(min(px * s1, py * s2), Tp) if levels == 'auto' else levels
            aw = None
            if beam is not None and awin is not None and in_kernel:
                def aw(q, pul, c=c, lo=lo, idx=idx):
                    # the window at points q (relative to the patch center) for pulses pul of af (the nearest
                    # recorded pulse for one inserted at a gap)
                    k = np.clip(np.searchsorted(idx, pul), 0, len(idx) - 1)
                    return wa(np.clip(beam(lo + k, np.asarray(q) + c) / umax, -1, 1))
            former = ImageFormer(af, f0, df, S.shape[1], px * s1, py * s2, spx / s1, spy / s2, e1, e2, backend,
                                 window=False, T=Tp, levels=nlev, aperture_weight=aw)

            def form(S_, former=former, idx=idx, n=len(af), s1=s1, s2=s2):
                xp = _xp(S_)
                if n > len(idx):                  # zero pulses at the dropped ones
                    Z = xp.zeros((n, S_.shape[1]), np.complex64)
                    Z[xp.asarray(idx)] = S_
                    S_ = Z
                return former(S_.astype(np.complex64))[::s1, ::s2]
        tm('plan and former')
        staged = None
        if backend in ('jax', 'tpu') and not exact and wt is None:
            # checks, scaling and upload of the history here, on the worker thread, not in the formation loop
            S_ = S
            if len(af) > len(idx):
                S_ = np.zeros((len(af), S.shape[1]), np.complex64)
                S_[idx] = S
            staged = former.stage(S_)
            tm('stage')
        return dict(i0=i0, j0=j0, c=c, lo=lo, hi=hi, S=S, df=df, wt=wt, form=form, Tp=Tp, s12=s12, nlev=nlev, err=err,
                    staged=staged and (lambda former=former, st=staged, s1=s1, s2=s2: former(st)[::s1, ::s2]))

    # the next patches are prepared (pulse span, range gate, plan, weights; host work) on worker threads while the
    # current one forms: FASTSAR_MOSAIC_PREFETCH workers (0: one patch at a time, in turn), by default 1 on the CPU,
    # whose formation uses every core, and up to 4 (a quarter of the cores) for a GPU or TPU, whose formation would
    # otherwise wait on the host
    order = [(i0, j0) for i0 in range(0, nx, mx) for j0 in range(0, ny, my)]
    nw = _env_number('FASTSAR_MOSAIC_PREFETCH', 0)
    nw = nw if nw is not None else (1 if backend == 'cpu' else max(1, min(4, (os.cpu_count() or 4) // 4)))
    nw = nw if len(order) > 1 else 0
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    ex = ThreadPoolExecutor(nw) if nw else None
    pending, nxt = deque(), 0
    spans = {}
    plan_programs = backend in ('jax', 'tpu') and not exact and len(order) > 1
    win = prof.get('win') if prof is not None else None
    if plan_programs or win is not None:
        # every patch's span first (cheap; in turn, on this thread), so that the pulse counts of the compiled programs
        # are chosen for the whole mosaic and the GPU window is checked against the longest span
        tm = _Timer('form_mosaic')
        spans = {q: span_of(*q) for q in order}
        tm('pulse spans')
    if win is not None:
        need = max((sp[2][1] - sp[2][0] for sp in spans.values() if sp[2] is not None), default=0)
        if need > win['rows']:
            from .memory import _warn, _gb
            _warn(f'cuda mosaic: a patch\'s range profiles ({_gb(need * win["row"])}) do not fit a quarter of the free '
                  f'GPU memory, so the gates of such patches are cut in host memory, which is slower; full speed needs '
                  f'about {_gb(4 * need * win["row"])} of GPU memory, {_gb(win["free"])} is free')
    if plan_programs:
        # FASTSAR_COMPILE_PATCHES: the compile time of one program in patch formations (docs/performance.md)
        tm = _Timer('form_mosaic')
        cc = _env_number('FASTSAR_COMPILE_PATCHES', 0.0, float)
        cc = (11.0 if backend == 'tpu' else 3.0) if cc is None else cc
        counts = [len(fill_gaps(ant[sp[2][0]:sp[2][1]])[0]) for sp in spans.values() if sp[2] is not None]
        planned[:] = choose_buckets(counts, cc) if counts else []
        # and the gate lengths, as fast FFT lengths at least the chosen ones
        from scipy.fft import next_fast_len
        g = [gate_bins(fx, ant, sp[0], sp[1], sp[2][0], sp[2][1], margin, guard) for sp in spans.values() if sp[2] is not None]
        K2 = g[0][1] if g else 0
        g = [n for n, _ in g if n < K2]
        fl = []
        for u in (choose_buckets(g, cc, step=2) if g else []):
            v = next_fast_len(u)
            while v % 2:
                v = next_fast_len(v + 1)
            fl.append(v if v < K2 else u)       # a gate of K2 or more is no gate: keep the bucket's own length
        planned_gates[:] = sorted(set(fl))
        tm('pulse buckets')

    # on the GPU, a patch forms while no worker gathers a patch history on the device (patch_history takes the same
    # lock), so the former's first-level groups, sized to the memory free when it starts, are not undercut by those
    # gathers. Histories already prepared stay on the device (at most one per worker) and count as used memory; a
    # former built meanwhile uploads only its plan arrays, which are small
    import contextlib
    gpu_turn = prof['lock'] if prof is not None and prof.get('device') is not None else contextlib.nullcontext()

    def submit():
        nonlocal nxt
        while nxt < len(order) and len(pending) < nw:
            pending.append(ex.submit(prep, *order[nxt], spans.get(order[nxt])))
            nxt += 1
    try:
        for k in range(len(order)):
            if ex is not None:
                submit()
                job = pending.popleft().result()
                submit()
            else:
                job = prep(*order[k], spans.get(order[k]))
            if job is None:
                continue
            i0, j0, c, lo, hi, S, df, wt, form = (job[q] for q in ('i0', 'j0', 'c', 'lo', 'hi', 'S', 'df', 'wt', 'form'))
            Tp, s12, nlev, err = (job[q] for q in ('Tp', 's12', 'nlev', 'err'))
            tm = _Timer('form_mosaic')
            if wt is None:
                with gpu_turn:
                    img = job['staged']() if job.get('staged') else form(S)
                tm('form')
            else:
                img = 0
                with gpu_turn:
                    for at, bt in zip(*wt):
                        img = img + form(S * at[:, None]) * RectBivariateSpline(si, sj, bt.reshape(len(si), len(sj)))(
                            np.arange(px), np.arange(py))
            ci, cj = min(mx, nx - i0), min(my, ny - j0)
            out[i0:i0 + ci, j0:j0 + cj] = img[crop:crop + ci, crop:crop + cj]
            if info is not None:
                info.append(dict(center=c, pulses=(lo, hi), K=S.shape[1], df=df, T=Tp, sub=s12, levels=nlev,
                                 predicted_error_db=err, terms=1 if wt is None else len(wt[0])))
    finally:
        if ex is not None:            # after an error in either thread, a patch not yet started is dropped
            ex.shutdown(cancel_futures=True)
    if os.environ.get('FASTSAR_TIMING') == '1' and prof is not None and prof.get('win') is not None:
        _Timer.acc['form_mosaic: profile window moves (count)'] = float(prof['win'].get('loads', 0))
    return out


def form_stripmap(data, p, compressed=False, rwin=None, awin=None, umax=1.0, rows=None, cols=None, patch=(128, 128),
                  crop=0, backend='cpu', **kw):
    """Patch-wise factorized backprojection of stripmap echoes on the straight track, on the zero-Doppler grid of
    stripmap.focus_stripmap (rows x along track, columns slant range r), which is the plane z = 0 of straight_track.
    rows, cols: (start, stop) of the grid to form (default all). -> (image, r, x) as focus_stripmap. Further keywords
    go to form_mosaic."""
    r, x = sm.axes(p, None, data.shape[1], data.shape[0])
    i0, i1 = (0, len(x)) if rows is None else rows
    j0, j1 = (0, len(r)) if cols is None else cols
    fx = echoes_to_fx(data, p, compressed, rwin)
    img = form_mosaic(fx, straight_track(p), (x[i0], r[j0], 0.0), i1 - i0, j1 - j0, x[1] - x[0], r[1] - r[0],
                      patch=patch, crop=crop, beam=stripmap_beam(p, straight_track(p)), umax=umax, awin=awin,
                      backend=backend, **kw)
    return img, r[j0:j1], x[i0:i1]
