"""Band-limited resampling: one axis, and the two-pass warp of a sheared lattice whose spectrum wraps."""
import numpy as np
import pytest

from fastsar import resample


def _db(err, ref):
    return 20 * np.log10(np.sqrt((np.abs(err) ** 2).mean() / (np.abs(ref) ** 2).mean()))


def _tones(rng, band, n=200):
    return (rng.random(n) - 0.5) * band, rng.normal(size=n) + 1j * rng.normal(size=n)


@pytest.mark.parametrize('band, taps, floor', [(0.5, 8, -58.0), (0.96, 8, -55.0), (0.96, 12, -65.0)])
def test_lines_band_limited(band, taps, floor):
    rng = np.random.default_rng(1)
    f, a = _tones(rng, band)
    n = 4096
    ev = lambda x: (a * np.exp(2j * np.pi * f * np.asarray(x)[..., None])).sum(-1)
    src = np.stack([ev(np.arange(n)), 2 * ev(np.arange(n) + 0.25)]).astype(np.complex64)
    pos = 1000 + rng.random((2, 3000)) * (n - 2000)
    out = resample.lines(src, pos, taps=taps)
    assert out.dtype == np.complex64
    assert _db(out[0] - ev(pos[0]), ev(pos[0])) < floor
    assert _db(out[1] - 2 * ev(pos[1] + 0.25), 2 * ev(pos[1])) < floor


def test_lines_outside_is_zero_and_samples_are_kept():
    rng = np.random.default_rng(2)
    src = (rng.normal(size=(3, 300)) + 1j * rng.normal(size=(3, 300))).astype(np.complex64)
    pos = np.tile(np.arange(-5.0, 305.0), (3, 1))
    out = resample.lines(src, pos)
    assert np.all(out[:, :5] == 0) and np.all(out[:, 305:] == 0)
    assert np.abs(out[:, 5:305] - src).max() < 2e-3 * np.abs(src).max()            # integer positions return the samples


def test_real_cubic_and_linear():
    x = np.linspace(0, 6, 400)
    src = np.stack([np.sin(x), x ** 2]).astype(np.float32)
    pos = np.tile(np.linspace(3, 395, 777), (2, 1))
    xs = pos[0] * (x[1] - x[0])
    for kernel, tol in (('cubic', 1e-5), ('linear', 1e-4)):
        out = resample.lines(src, pos, kernel=kernel)
        assert out.dtype == np.float32
        assert np.abs(out[0] - np.sin(xs)).max() < tol and np.abs(out[1] - xs ** 2).max() < 20 * tol


def test_knot_map_is_bilinear():
    lk, ok = np.array([0.0, 40.0, 99.0]), np.array([0.0, 64.0, 128.0, 149.0])
    f = lambda i, j: 3.0 + 0.5 * i - 0.25 * j + 0.01 * i * j
    km = resample.KnotMap(lk, ok, f(lk[:, None], ok[None, :]), 150)
    got = km(slice(10, 100))
    assert got.shape == (90, 150)
    assert np.abs(got - f(np.arange(10, 100)[:, None], np.arange(150)[None, :])).max() < 1e-9


def test_warp_sheared_lattice_with_wrapped_spectrum():
    """A field band limited to a rectangle in (a, r), sampled on (x, y) with a = al x + be y + curvature and
    r = ga y + curvature at 4% above the critical rates: its (x, y) spectrum is a parallelogram
    wider than the y sampling band, and the two-pass warp must still return the field on the (a, r) lattice."""
    rng = np.random.default_rng(3)
    al, be, ga, c1, c2, c3 = -0.57, 0.78, 0.32, 2e-5, -1e-5, 1.5e-5
    Ba = Br = 1.0
    fa, amp = _tones(rng, Ba, 120)
    fr = (rng.random(120) - 0.5) * Br

    def F(a, r):
        out = np.zeros(np.broadcast(a, r).shape, np.complex128)
        for k in range(len(amp)):
            out += amp[k] * np.exp(2j * np.pi * (fa[k] * a + fr[k] * r))
        return out
    A = lambda x, y: al * x + be * y + c1 * x * y + c2 * y * y
    R = lambda x, y: ga * y + c3 * (x * x + y * y)
    # sample 4% above the largest local band over |x| < 400, |y| < 700: da/dx Ba + dr/dx Br along x at fixed y,
    # and dr/dy Br along y at fixed a (dx/dy = -be/al there)
    sx = 0.96 / ((abs(al) + c1 * 700) * Ba + 2 * c3 * 400 * Br)
    sy = 0.96 / ((ga + 2 * c3 * 700 + 2 * c3 * 400 * abs(be / al)) * Br)
    n0, n1 = 460, 400
    x, y = (np.arange(n0) - n0 / 2) * sx, (np.arange(n1) - n1 / 2) * sy
    src = F(A(x[:, None], y[None, :]), R(x[:, None], y[None, :])).astype(np.complex64)
    assert abs(be) * Ba + ga * Br > 1 / sy                                       # the y spectrum does wrap
    m0, m1 = 96, 110
    a, r = (np.arange(m0) - m0 / 2) * 0.96 / Ba, (np.arange(m1) - m1 / 2) * 0.96 / Br
    X = lambda a_, y_: (a_ - be * y_ - c2 * y_ * y_) / (al + c1 * y_)              # x of slant azimuth a_ on column y_
    pos0 = X(a[None, :], y[:, None]) / sx + n0 / 2                                 # [n1, m0]
    yy = np.broadcast_to(r[None, :] / ga, (m0, m1)).copy()
    for _ in range(8):                                                             # y of (a_i, r_k), by Newton
        h = 1e-3
        g = R(X(a[:, None], yy), yy) - r[None, :]
        yy -= g * h / (R(X(a[:, None], yy + h), yy + h) - r[None, :] - g)
    pos1 = yy / sy + n1 / 2
    need = X(a[:, None], yy) / sx + n0 / 2                                         # the source rows the output uses
    assert need.min() > 20 and need.max() < n0 - 20 and pos1.min() > 20 and pos1.max() < n1 - 20
    out = resample.warp(src, pos0, pos1, (m0, m1), taps=12)
    ref = F(np.broadcast_to(a[:, None], (m0, m1)), np.broadcast_to(r[None, :], (m0, m1)))
    assert _db(out - ref, ref) < -42.0
    # knot maps reproduce the full maps for this smooth warp
    k0, k1 = np.linspace(0, n1 - 1, 80), np.linspace(0, m0 - 1, 20)
    kk = np.linspace(0, m1 - 1, 23)
    p0 = resample.KnotMap(k0, k1, X(np.interp(k1, np.arange(m0), a)[None, :], np.interp(k0, np.arange(n1), y)[:, None]) / sx + n0 / 2, m0)
    i1, j1 = np.rint(k1).astype(int), np.rint(kk).astype(int)
    p1 = resample.KnotMap(i1.astype(float), j1.astype(float), pos1[np.ix_(i1, j1)], m1)
    out2 = resample.warp(src, p0, p1, (m0, m1), taps=12)
    assert _db(out2 - ref, ref) < -38.0


DEVICES = [pytest.param('cupy', marks=pytest.mark.cuda), 'jax']


def _device(lib, need):
    """The array module of a device library, and the largest and rms differences from the CPU result that its
    positions allow, relative to the largest sample: CuPy's are the CPU's (double); JAX's are float32 about an exact
    integer, which moves a sample in a hundred to the next row of the table."""
    return (need('cupy'), 3e-6, 3e-6) if lib == 'cupy' else (need('jax.numpy'), 2e-3, 2e-4)


def _same(got, ref, lib, tol, rms):
    from fastsar import _xp
    assert _xp.kind(got) == lib
    got = _xp.host(got)
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert np.array_equal(got == 0, ref == 0)
    d, s = np.abs(got - ref), np.abs(ref).max()
    assert d.max() < tol * s and np.sqrt((d ** 2).mean()) < rms * s, (d.max() / s, np.sqrt((d ** 2).mean()) / s)


@pytest.mark.parametrize('lib', DEVICES)
def test_lines_on_device(lib, need):
    xp, tol, rms = _device(lib, need)
    rng = np.random.default_rng(4)
    src = (rng.normal(size=(150, 331)) + 1j * rng.normal(size=(150, 331))).astype(np.complex64)
    pos = rng.random((150, 257)) * 345 - 7                                         # some outside, at both ends
    pos[:, :3] = [0.0, 330.0, 17.0]
    _same(resample.lines(xp.asarray(src), pos), resample.lines(src, pos), lib, tol, rms)
    got = resample.lines(xp.asarray(src), lambda sl: pos[sl], 257, taps=12)
    _same(got, resample.lines(src, pos, taps=12), lib, tol, rms)
    for kernel in ('cubic', 'sinc'):                                               # real lines: float32
        _same(resample.lines(xp.asarray(src.real), pos, kernel=kernel), resample.lines(src.real, pos, kernel=kernel),
              lib, tol, rms)
    # positions on the device: float32 ones for JAX, which hold a 300-sample line to 2e-5 sample
    got = resample.lines(xp.asarray(src), xp.asarray(pos[:, 3:]))
    _same(got, resample.lines(src, pos[:, 3:]), lib, 5 * tol, 5 * rms)


@pytest.mark.parametrize('lib', DEVICES)
def test_warp_on_device(lib, need):
    xp, tol, rms = _device(lib, need)
    rng = np.random.default_rng(5)
    n0, n1, m0, m1 = 300, 260, 281, 310
    src = (rng.normal(size=(n0, n1)) + 1j * rng.normal(size=(n0, n1))).astype(np.complex64)
    lk, ok = np.linspace(0, n1 - 1, 9), np.linspace(0, m0 - 1, 11)
    p0 = resample.KnotMap(lk, ok, 1.09 * ok[None] + 0.03 * lk[:, None] + 1e-4 * lk[:, None] * ok[None] - 7.3, m0)
    lk, ok = np.linspace(0, m0 - 1, 10), np.linspace(0, m1 - 1, 12)
    p1 = resample.KnotMap(lk, ok, 0.87 * ok[None] + 0.02 * lk[:, None] - 5.6, m1)
    for s, kw in ((src, {}), (src.real, dict(kernel='cubic')), (src.imag, dict(kernel='sinc', pad=3))):
        ref = resample.warp(s, p0, p1, (m0, m1), **kw)
        assert (ref == 0).any() and (ref != 0).mean() > 0.8
        _same(resample.warp(xp.asarray(s), p0, p1, (m0, m1), **kw), ref, lib, tol, rms)
        # the transposed warp of the device modules (planes.to_ground): no transposed copy of the image
        st = xp.asarray(np.ascontiguousarray(s.T))
        got = resample._device(st).warp(st, p0, p1, (m0, m1), transposed=True, **kw)
        _same(got, np.ascontiguousarray(ref.T), lib, tol, rms)
    # full maps as arrays give what the knot maps give
    ref = resample.warp(src, p0(slice(0, n1)), p1(slice(0, m0)), (m0, m1))
    _same(resample.warp(xp.asarray(src), p0(slice(0, n1)), p1(slice(0, m0)), (m0, m1)), ref, lib, tol, rms)
