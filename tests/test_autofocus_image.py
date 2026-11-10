"""fastsar.autofocus in the image domain (quadratic, spatially_variant, image_autofocus): synthetic baseband images
with a known phase error in the azimuth frequency domain, on NumPy, CuPy and JAX arrays, and a simulated spotlight
collection with squint and a curved track whose phase history carries a phase error per pulse."""
import dataclasses

import numpy as np
import pytest

import fastsar
from fastsar import autofocus as af, io, planes, quality, sim

LAT, LON, H0, HEADING = 40.79, -77.86, 351.0, 30.0
TG = np.array([[0, 0, 0], [50, 35, 0], [-55, 45, 0], [40, -60, 0], [-45, -50, 0], [65, 0, 0], [0, -70, 0.]])
AMP = np.exp(1j * np.arange(len(TG)))


def _scene(nx, ny, frac, npt=40, seed=0, k0=0, clutter=0.05):
    """A focused image: point targets (one per column at most, away from the azimuth ends) over complex clutter,
    band-limited along azimuth to the fraction frac of the sampled band, centered on bin k0.
    -> (image complex64 [nx, ny], v [nx]: offset from the band center in half band widths (FFT order), (i, j))."""
    rng = np.random.default_rng(seed)
    img = clutter * (rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny)))
    i, j = rng.integers(nx // 8, nx - nx // 8, npt), rng.choice(ny, npt, replace=False)
    img[i, j] += rng.uniform(5, 20, npt) * np.exp(2j * np.pi * rng.random(npt))
    v = (((np.arange(nx) - k0) / nx + 0.5) % 1.0 - 0.5) / (frac / 2)
    return np.fft.ifft(np.fft.fft(img, axis=0) * (np.abs(v) <= 1)[:, None], axis=0).astype(np.complex64), v, (i, j)


def _defocus(img, phi):
    """img with exp(1j * phi) on its azimuth spectrum; phi [nx] or [nx, ny] (a different error per column)."""
    phi = phi[:, None] if phi.ndim == 1 else phi
    return np.fft.ifft(np.fft.fft(img, axis=0) * np.exp(1j * phi), axis=0).astype(np.complex64)


def _quad(v, c):
    return np.where(np.abs(v) <= 1, c * v ** 2, 0.0)


def _rel(a, b, gain=False):
    """Relative rms difference of a from b; gain: after the best complex factor on a (pga leaves a phase open)."""
    g = np.vdot(a, b) / np.vdot(a, a) if gain else 1.0
    return float(np.sqrt(np.mean(np.abs(g * a - b) ** 2) / np.mean(np.abs(b) ** 2)))


def _target(img, i, j, half=24, up=16):
    """Peak amplitude and -3 dB width (pixels) of the azimuth cut through the target near (i, j)."""
    cut = np.fft.fft(img[i - half:i + half, j])
    pad = np.zeros(2 * half * up, complex)
    pad[:half], pad[-half:] = cut[:half], cut[half:]
    p = np.abs(np.fft.ifft(pad) * up) ** 2
    k, n = int(np.argmax(p)), len(p)
    edge = []
    for d in (-1, 1):                                                 # the half-power crossing on each side
        m = k
        while p[(m + d) % n] >= 0.5 * p[k] and abs(m - k) < n // 2:
            m += d
        y0, y1 = p[m % n], p[(m + d) % n]
        edge.append(abs(m - k) + (y0 - 0.5 * p[k]) / (y0 - y1))
    return np.sqrt(p[k]), sum(edge) / up


@pytest.mark.parametrize('nx,ny,frac,k0', [(512, 384, 0.8, 0), (1024, 256, 0.5, 0), (768, 300, 0.5, 24), (512, 256, 1.0, 0)],
                         ids=['0.8 band', '2x oversampled', '2x off center', 'full band'])
def test_quadratic(nx, ny, frac, k0):
    foc, v, (ti, tj) = _scene(nx, ny, frac, k0=k0)
    out, phi, info = af.quadratic(foc)                                # a focused image is left alone
    assert out is foc and not info['applied'] and abs(info['edge_rad']) < 0.02 and not phi.any()
    assert abs(info['band'][0] - k0 / nx) < 1.0 / nx and abs(info['band'][1] - frac / 2) < 1.0 / nx
    for c in (2.0, -2.0, 6.0, -15.0, 40.0, -40.0):
        img = _defocus(foc, _quad(v, c))
        keep = img.copy()
        out, phi, info = af.quadratic(img)
        assert info['applied'] and out.dtype == np.complex64 and np.array_equal(img, keep)
        assert abs(info['edge_rad'] / c - 1) < 0.03, (c, info['edge_rad'])
        assert info['contrast_after'] > info['contrast_before'] and info['iterations'] <= 6
        # phi is what was removed, per bin in fftfreq order, zero outside the band
        again = np.fft.ifft(np.fft.fft(img, axis=0) * np.exp(-1j * phi)[:, None], axis=0)
        assert np.abs(out - again).max() < 1e-5 * np.abs(foc).max() and not phi[np.abs(v) > 1 + 2 / (frac * nx)].any()
        assert np.abs(phi - _quad(v, c))[np.abs(v) < 0.98].max() < 0.03 * abs(c) + 0.02
        for i, j in list(zip(ti, tj))[:8]:                            # peak and width of the targets restored
            (p0, w0), (p1, w1), (p2, w2) = _target(foc, i, j), _target(img, i, j), _target(out, i, j)
            assert abs(p2 / p0 - 1) < 0.02 and abs(w2 / w0 - 1) < 0.02, (c, p2 / p0, w2 / w0)
            assert p1 < 0.9 * p0
    out, phi, info = af.quadratic(_defocus(foc, _quad(v, 6.0)).astype(np.complex128), lines=1.0, max_lines=64)
    assert out.dtype == np.complex64 and info['lines'] <= 64 and abs(info['edge_rad'] / 6 - 1) < 0.03


def test_zeros():
    nx, ny = 640, 400
    foc, v, _ = _scene(nx, ny, 0.5, npt=80, seed=4)
    img = _defocus(foc, _quad(v, 9.0))
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    hole = (ii < 40) | (ii >= nx - 25) | (jj < 30) | (jj >= ny - 12) | (ii / nx + jj / ny > 1.6)   # border and a corner
    img[hole] = 0
    assert hole[:, 300:].mean(0).max() > 0.6                          # some columns are mostly zero
    for mode, kw in (('quadratic', {}), ('pga', {}), ('spatially_variant', dict(block=256)),
                     ('spatially_variant_pga', dict(block=256))):
        out, info = af.image_autofocus(img, mode, **kw)
        assert info['applied'] and info['mode'] == mode, mode
        assert not out[hole].any() and np.count_nonzero(out[~hole]) > 0.999 * (~hole).sum(), mode
        inner = np.s_[80:-65, 30:200]
        e = _rel(out[inner], foc[inner], gain='pga' in mode)
        assert e < 0.15 < 0.5 < _rel(img[inner], foc[inner]), (mode, e)
    assert abs(af.quadratic(img)[2]['edge_rad'] / 9 - 1) < 0.03
    out, phi, info = af.quadratic(np.zeros((64, 32), np.complex64))
    assert not info['applied'] and not out.any() and 'no data' in info['reason']
    assert not af.spatially_variant(np.zeros((64, 32), np.complex64))[1]['applied']


def test_axis1_independence():
    nx, ny = 512, 320
    foc, v, _ = _scene(nx, ny, 0.5, seed=5)
    img = _defocus(foc, _quad(v, 11.0))
    rng = np.random.default_rng(6)
    factors = (np.exp(2j * np.pi * rng.random(ny)),                   # any phase per column
               np.exp(2j * np.pi * 0.37 * np.arange(ny)))             # a ramp along axis 1: its spectrum wraps
    for mode, kw in (('quadratic', {}), ('pga', {}), ('spatially_variant', dict(block=(256, 160)))):
        ref, info = af.image_autofocus(img, mode, **kw)
        assert info['applied']
        for f in factors:
            f = f.astype(np.complex64)[None, :]
            out, info2 = af.image_autofocus(img * f, mode, **kw)
            assert np.abs(out - ref * f).max() < 2e-4 * np.abs(ref).max(), mode
            if mode == 'quadratic':
                assert abs(info2['edge_rad'] - info['edge_rad']) < 1e-3


def test_pga_mode():
    nx, ny = 1024, 256
    foc, v, _ = _scene(nx, ny, 0.5, seed=7)
    err = np.where(np.abs(v) <= 1, 5 * v ** 2 + 6 * v ** 3 + 1.5 * np.sin(7 * v), 0.0)        # not a quadratic
    img = _defocus(foc, err)
    q, qi = af.image_autofocus(img, 'quadratic')
    p, pi = af.image_autofocus(img, 'pga')
    assert qi['applied'] and pi['applied'] and pi['contrast_after'] > qi['contrast_after']
    inb = np.abs(v) < 0.95
    res = af._detrend((err - pi['phi'])[inb], v[inb])                 # what pga leaves, less a phase and a shift
    assert res.std() < 0.1 < 1.0 < af._detrend((err - qi['phi'])[inb], v[inb]).std(), res.std()
    con = lambda a: np.std(np.abs(a) ** 2) / np.mean(np.abs(a) ** 2)
    assert con(p) > 0.97 * con(foc) and con(q) < 0.8 * con(foc) and not pi['phi'][np.abs(v) > 1.02].any()


def _variant(nx, ny, frac, npt, seed):
    foc, v, _ = _scene(nx, ny, frac, npt=npt, seed=seed)
    x, y = (np.arange(nx) - (nx - 1) / 2) / (nx / 2), (np.arange(ny) - (ny - 1) / 2) / (ny / 2)
    return foc, v, x, y, np.abs(v) <= 1


def _no_seams(out, foc, step, edge):
    """Against the exact reference: the error at the rows where a strip of the cross-fade starts or ends (every
    `step` rows) is no larger than elsewhere, the difference between adjacent rows no larger there than the
    reference's own, and no pixel is far off."""
    d = (np.abs(out - foc) ** 2).mean(1)
    cut = np.zeros(len(d), bool)
    for k in (-1, 0):
        cut[np.arange(step, len(d) - step, step) + k] = True
    cut[:edge] = cut[-edge:] = False
    rest = ~cut
    rest[:edge] = rest[-edge:] = False
    assert d[cut].mean() < 1.5 * d[rest].mean(), (d[cut].mean(), d[rest].mean())
    jump = lambda a: (np.abs(np.diff(a, axis=0)) ** 2).mean(1)[np.arange(step, len(d) - step, step) - 1].mean()
    assert abs(jump(out) / jump(foc) - 1) < 0.1, jump(out) / jump(foc)
    assert np.abs(out - foc)[edge:-edge].max() < 0.12 * np.abs(foc).max()


def test_spatially_variant_range():
    nx, ny = 1024, 768
    foc, v, x, y, inb = _variant(nx, ny, 0.5, 600, 1)
    c = 8 + 5 * y                                                     # the coefficient across range (rad)
    img = _defocus(foc, np.where(inb[:, None], c[None, :] * v[:, None] ** 2, 0.0))
    g = af.quadratic(img)[0]
    for method, tol in (('quadratic', 0.06), ('pga', 0.1)):
        out, info = af.spatially_variant(img, block=256, method=method, order=1)
        assert info['applied'] and out.dtype == np.complex64 and info['block'] == (256, 256)
        assert info['terms'] == [(0, 0), (0, 1), (1, 0)] and info['coefficients'].shape == (3, len(info['powers']))
        assert len(info['positions']) == len(info['estimates']) == len(info['weights']) == 7 * 5
        co = info['coefficients'][:, 0]
        assert abs(co[0] - 8) < 0.2 and abs(co[1] - 5) < 0.4 and abs(co[2]) < 0.3 and info['rms'][0] < 0.4, (co, info['rms'])
        e = _rel(out[32:-32], foc[32:-32])
        assert e < tol and e < 0.2 * _rel(g[32:-32], foc[32:-32]), (method, e)
        _no_seams(out, foc, 128, 32)
    out2, info2 = af.spatially_variant(img, block=256, order=2)       # the default degree, more terms than needed
    assert len(info2['terms']) == 6 and _rel(out2[32:-32], foc[32:-32]) < 0.08


def test_spatially_variant_azimuth():
    nx, ny = 1024, 512
    foc, v, x, y, inb = _variant(nx, ny, 0.5, 500, 2)
    c = 7 + 2.5 * x                                                   # the coefficient of a scatterer at row x' (rad)
    E = np.exp(-2j * np.pi * np.outer(np.arange(nx), np.arange(nx)) / nx
               + 1j * np.where(inb[:, None], c[None, :] * v[:, None] ** 2, 0.0))
    img = np.fft.ifft(E @ foc.astype(np.complex128), axis=0).astype(np.complex64)             # exact, space variant
    g = af.quadratic(img)[0]
    out, info = af.spatially_variant(img, block=192, overlap=0.75, order=1)
    co = info['coefficients'][:, 0]
    assert info['applied'] and abs(co[0] - 7) < 0.15 and abs(co[2] - 2.5) < 0.25 and abs(co[1]) < 0.2, co
    e = _rel(out[48:-48], foc[48:-48])
    assert e < 0.05 and e < 0.25 * _rel(g[48:-48], foc[48:-48]), (e, _rel(g[48:-48], foc[48:-48]))
    _no_seams(out, foc, 48, 48)


def test_spatially_variant_invariant():
    nx, ny = 768, 512
    foc, v, x, y, inb = _variant(nx, ny, 0.8, 400, 3)
    img = _defocus(foc, _quad(v, -9.0))
    g, _, gi = af.quadratic(img)
    out, info = af.spatially_variant(img, block=256)
    lo, hi = info['edge_rad']
    assert info['applied'] and abs(lo - gi['edge_rad']) < 0.15 and abs(hi - gi['edge_rad']) < 0.15, (lo, hi)
    assert _rel(out[32:-32], g[32:-32]) < 0.03 and info['rms'][0] < 0.05
    same, info = af.spatially_variant(foc, block=256)                 # nothing to correct
    assert same is foc and not info['applied'] and 'below tol' in info['reason']
    assert af.spatially_variant(img, block=4096)[1]['block'] == (nx, ny)    # one block: the image


def _backends():
    out = [pytest.param('numpy', id='numpy')]
    try:
        import cupy
        cupy.zeros(1).sum().get()
        out.append(pytest.param('cupy', id='cupy'))
    except Exception:
        out.append(pytest.param('cupy', id='cupy', marks=pytest.mark.skip(reason='no CuPy or no GPU')))
    try:
        import jax  # noqa: F401
        out.append(pytest.param('jax', id='jax'))
    except Exception:
        out.append(pytest.param('jax', id='jax', marks=pytest.mark.skip(reason='no JAX')))
    return out


@pytest.mark.parametrize('lib', _backends())
def test_devices(lib):
    nx, ny = 640, 448
    foc, v, x, y, inb = _variant(nx, ny, 0.5, 300, 8)
    img = _defocus(foc, np.where(inb[:, None], (9 + 3 * y)[None, :] * v[:, None] ** 2 + 2 * v[:, None] ** 3, 0.0))
    img[:20] = 0
    img[:, -9:] = 0
    if lib == 'cupy':
        import cupy as xp
    elif lib == 'jax':
        import jax.numpy as xp
    else:
        xp = np
    dev = xp.asarray(img)
    assert af._xp.kind(dev) == lib
    for mode, kw in (('quadratic', {}), ('pga', {}), ('spatially_variant', dict(block=(192, 128))),
                     ('spatially_variant_pga', dict(block=256, poly=3)), ('quadratic', dict(max_lines=100, lines=1.0))):
        ref, info = af.image_autofocus(img, mode, **kw)
        out, dinfo = af.image_autofocus(dev, mode, **kw)
        assert info['applied'] and dinfo['applied'], mode
        assert af._xp.kind(out) == lib and type(out) is type(dev) and out.dtype == np.complex64, mode
        got = af._xp.host(out)
        assert np.abs(got - ref).max() < 2e-5 * np.abs(ref).max(), (mode, np.abs(got - ref).max() / np.abs(ref).max())
        assert not got[img == 0].any()
        if 'phi' in info:
            assert isinstance(dinfo['phi'], np.ndarray) and np.abs(dinfo['phi'] - info['phi']).max() < 1e-4
    same, info = af.image_autofocus(dev, None)
    assert same is dev and info == dict(mode=None, applied=False)
    ref = af.quadratic(img)[0]
    for size in (1 << 14, 77 * nx):                                   # the correction in several column blocks
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(af, '_BLOCK', dict(numpy=size, cupy=size, jax=size))
            for mode, kw, r in (('quadratic', {}, ref), ('spatially_variant', dict(block=(192, 128)), None)):
                out, _ = af.image_autofocus(dev, mode, **kw)
                r = af.image_autofocus(img, mode, **kw)[0] if r is None else r
                assert out.shape == img.shape and np.abs(af._xp.host(out) - r).max() < 2e-5 * np.abs(r).max(), (size, mode)
    same, _, info = af.quadratic(xp.asarray(foc))                     # nothing to do: the array itself comes back
    assert same is not None and not info['applied'] and af._xp.kind(same) == lib


def test_dispatch_and_errors():
    foc, v, _ = _scene(256, 96, 0.5, seed=9)
    img = _defocus(foc, _quad(v, 5.0))
    for off in (None, False):
        out, info = af.image_autofocus(img, off)
        assert out is img and info == dict(mode=None, applied=False)
    out, info = af.image_autofocus(img, 'quadratic', tol=0.05)
    ref, phi, rinfo = af.quadratic(img, tol=0.05)
    assert info['mode'] == 'quadratic' and np.array_equal(out, ref) and np.array_equal(info['phi'], phi)
    assert set(rinfo) >= {'edge_rad', 'band', 'iterations', 'contrast_before', 'contrast_after', 'applied'}
    out, info = af.image_autofocus(img, 'spatially_variant_pga', block=128)
    assert info['mode'] == 'spatially_variant_pga' and info['method'] == 'pga' and info['powers'] == (2, 3, 4)
    with pytest.raises(ValueError, match='quadratic.*pga.*spatially_variant.*spatially_variant_pga'):
        af.image_autofocus(img, 'mapdrift')
    with pytest.raises(ValueError):
        af.quadratic(np.abs(img))                                     # not complex
    with pytest.raises(ValueError):
        af.quadratic(img[:, 0])                                       # not an image
    with pytest.raises(ValueError):
        af.spatially_variant(img, method='mapdrift')
    with pytest.raises(ValueError):
        af.spatially_variant(img, overlap=1.0)
    # the guard: whatever a structureless image gives, a result is kept only if the contrast did not drop
    rng = np.random.default_rng(10)
    noise = (rng.standard_normal((256, 96)) + 1j * rng.standard_normal((256, 96))).astype(np.complex64)
    for mode in af._MODES:
        out, info = af.image_autofocus(noise, mode)
        assert (info['contrast_after'] >= info['contrast_before']) if info['applied'] else (out is noise), mode
    out, phi, info = af.quadratic(img, guard=False)
    assert info['applied'] and set(af.__all__) >= {'pga', 'autofocus', 'quadratic', 'spatially_variant', 'image_autofocus'}


def _con(a):
    """Contrast of an image: std / mean of its intensity."""
    p = np.abs(np.asarray(a, np.complex128)) ** 2
    return float(p.std() / p.mean())


def _relief(seed, nx=768, ny=512, c=2.5):
    """A scene whose two halves in range carry quadratic errors -c and +c (rad at the band edge), as relief does on
    a curved track -> (focused, defocused)."""
    foc, v, _ = _scene(nx, ny, 0.5, npt=200, seed=seed)
    cj = np.where(np.arange(ny) < ny // 2, -c, c)
    return foc, _defocus(foc, np.where((np.abs(v) <= 1)[:, None], cj[None, :] * v[:, None] ** 2, 0.0))


@pytest.mark.parametrize('seed', [11, 12, 13])
def test_guard_parts_disagree(seed):
    """One coefficient for a scene whose halves ask for opposite ones sharpens one half and blurs the other
    further: the global modes apply nothing and say why, the spatially variant mode corrects both halves."""
    foc, img = _relief(seed)
    h = img.shape[1] // 2
    forced, _, fi = af.quadratic(img, guard=False)                    # the estimate as it is
    assert fi['applied'] and abs(abs(fi['edge_rad']) - 2.5) < 0.2 and fi['edge_rad_std'] is None
    assert min(_con(forced[:, :h]) / _con(img[:, :h]), _con(forced[:, h:]) / _con(img[:, h:])) < 0.8
    for mode in ('quadratic', 'pga'):
        out, info = af.image_autofocus(img, mode)
        assert out is img and not info['applied'] and 'disagree' in info['reason'], (mode, info['reason'])
        assert info['contrast_after'] == info['contrast_before'] == pytest.approx(_con(img), rel=1e-4)
    out, phi, info = af.quadratic(img)
    assert not phi.any() and abs(info['edge_rad'] - fi['edge_rad']) < 1e-6          # the estimate is still reported
    sub = info['subsets']
    assert len(sub) == 8 and (sub < -2).sum() >= 3 and (sub > 2).sum() >= 3
    assert info['subset_significance'].min() > af.MIN_SIG
    assert info['significance'] > af.MIN_SIG and af.image_autofocus(img, 'pga')[1]['rms_rad_std'] > 0.2
    out, info = af.spatially_variant(img, block=256)
    assert info['applied'] and info['contrast_after'] == pytest.approx(_con(out), rel=1e-4)
    assert min(_con(out[:, :h]) / _con(img[:, :h]), _con(out[:, h:]) / _con(img[:, h:])) > 1.2
    assert info['part_gain'] > 2 * info['part_gain_std'] > 0 and info['gain'] > 2 * info['gain_std'] > 0


def test_guard_whole_image():
    """The columns the estimate reads carry a real quadratic error and agree on it, but the brightest reflectors of
    the image, in columns it does not read, are in focus: the correction would lower the contrast of the image,
    which is measured on every column, and is not applied."""
    nx, ny = 768, 1024
    foc, v, _ = _scene(nx, ny, 0.5, npt=150, seed=21)
    img = _defocus(foc, _quad(v, 1.5))
    read = np.unique(np.round(np.linspace(0, ny - 1, ny // 4)).astype(np.int64))
    cols = np.setdiff1d(np.arange(ny), read)[np.linspace(0, ny - len(read) - 1, 9).astype(int)[1:-1]]
    bright = np.zeros((nx, ny), complex)
    bright[np.linspace(150, nx - 150, len(cols)).astype(int), cols] = 150.0
    img = img + np.fft.ifft(np.fft.fft(bright, axis=0) * (np.abs(v) <= 1)[:, None], axis=0).astype(np.complex64)
    forced, _, fi = af.quadratic(img, guard=False, max_lines=ny // 4)
    assert fi['applied'] and abs(fi['edge_rad'] - 1.5) < 0.1 and _con(forced) < 0.9 * _con(img)
    assert fi['contrast_before'] == pytest.approx(_con(img), rel=1e-4)                # reported as measured
    assert fi['contrast_after'] == pytest.approx(_con(forced), rel=1e-4)
    out, phi, info = af.quadratic(img, max_lines=ny // 4)
    assert out is img and not info['applied'] and not phi.any() and 'lowers the contrast' in info['reason']
    assert abs(info['edge_rad'] - 1.5) < 0.1 and info['edge_rad_std'] < 0.05          # a consistent estimate
    assert info['contrast_after'] < 0.9 * info['contrast_before'] and info['gain'] < 0
    assert info['part_gain'] > 2 * info['part_gain_std']                              # most of the image gains
    # one reflector that gains while the rest of the image loses: the contrast of the image rises, the median of
    # its parts falls
    foc, v, _ = _scene(512, 256, 0.5, npt=60, seed=22)
    one = np.zeros((512, 256), complex)
    one[200, 100] = 400.0
    img = foc + _defocus(np.fft.ifft(np.fft.fft(one, axis=0) * (np.abs(v) <= 1)[:, None], axis=0), _quad(v, 2.0))
    info = {}
    out = af._keep(img, af._apply(img, _quad(v, 2.0)), info, True)
    assert out is img and not info['applied'] and 'most parts' in info['reason']
    assert info['contrast_after'] > info['contrast_before'] and info['part_gain'] < -2 * info['part_gain_std']
    info = {}
    assert af._keep(img, af._apply(img, _quad(v, 2.0)), info, False) is not img and info['applied']


def test_guard_no_structure():
    """Speckle alone: the two looks do not correlate, whatever the largest value of their cross-correlation gives is
    not an estimate, and nothing is applied."""
    rng = np.random.default_rng(23)
    noise = (rng.standard_normal((1024, 512)) + 1j * rng.standard_normal((1024, 512))).astype(np.complex64)
    out, phi, info = af.quadratic(noise)
    assert out is noise and not info['applied'] and not phi.any() and 'not significant' in info['reason']
    assert info['significance'] < af.MIN_SIG and info['subset_significance'].max() < af.MIN_SIG


@pytest.mark.parametrize('lib', _backends())
def test_guard_devices(lib):
    """The guard's sums are taken on the device and agree with the host's; a refused correction returns the array
    it was given."""
    foc, img = _relief(11)
    img[:30] = 0
    if lib == 'cupy':
        import cupy as xp
    elif lib == 'jax':
        import jax.numpy as xp
    else:
        xp = np
    dev = xp.asarray(img)
    ref = af._colsums(img)
    for size in (1 << 30, 100 * img.shape[0]):                        # one block of columns, and several
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(af, '_BLOCK', dict(numpy=size, cupy=size, jax=size))
            got = af._colsums(dev)
            assert got.shape == (3, img.shape[1]) and np.array_equal(got[0], ref[0])
            assert np.abs(got / ref - 1).max() < 1e-4
    for mode in ('quadratic', 'pga'):
        out, info = af.image_autofocus(dev, mode)
        assert out is dev and not info['applied'] and 'disagree' in info['reason'], mode
    out, info = af.image_autofocus(dev, 'spatially_variant', block=256)
    rinfo = af.image_autofocus(img, 'spatially_variant', block=256)[1]
    assert info['applied'] and af._xp.kind(out) == lib
    for k in ('contrast_before', 'contrast_after', 'gain', 'gain_std', 'part_gain', 'part_gain_std'):
        assert info[k] == pytest.approx(rinfo[k], rel=1e-3, abs=1e-5), k


def _collect(squint):
    col = sim.make_collect(res=0.5, scene=170.0, r0=10e3, graze_deg=35.0)
    y = col.ant[:, 1] / np.ptp(col.ant[:, 1])
    ant = col.ant.copy()
    ant[:, 1] -= 10e3 * np.cos(np.radians(35)) * np.tan(np.radians(squint))
    ant[:, 0] += 160.0 * y ** 2 + 120.0 * y ** 3                      # a curved track
    ant[:, 2] += -100.0 * y ** 2 + 6.0 * y
    return dataclasses.replace(col, ant=ant)


def _az_metrics(o, tg, fr, image=None):
    """Azimuth resolution (m), peak sidelobe ratio (dB) and peak amplitude of each target in a ground or slant image,
    and the number of nonzero pixels on the 41 pixels of its column centered on it."""
    res = []
    for p in tg:
        if o['plane'] == 'slant':
            a, r = planes.ground_to_slant(fr, p)
            ie, je = a / o['spx'] + o['scp_pixel'][0], r / o['spy'] + o['scp_pixel'][1]
        else:
            ie, je = (p - o['origin']) @ o['e1'] / o['spx'], (p - o['origin']) @ o['e2'] / o['spy']
        m = quality.point_target(o['image'] if image is None else image, (int(round(ie)), int(round(je))), o['spx'],
                                 o['spy'], half=16)
        res.append((m['res_az'], m['pslr_az'], m['peak'],
                    np.count_nonzero(o['image'][int(round(ie)) - 20:int(round(ie)) + 21, int(round(je))])))
    return np.array(res)


def test_collection(tmp_path):
    """A phase error per pulse in the phase history of a squinted, curved-track collection, removed from the
    ground-plane image on line-of-sight axes and from the slant-plane image."""
    pytest.importorskip('finufft')                                    # sim.simulate
    col = _collect(20.0)
    S = sim.simulate(col, TG, AMP)
    got = {}

    def plan(g):
        fr = planes.slant_frame(g['meta'], g['center'])
        t = np.linspace(-85, 85, 5)
        pts = g['center'] + (t[:, None, None] * g['e1'] + t[None, :, None] * g['e2']).reshape(-1, 3)
        got['b'] = planes.sampling(fr, g['ant'], (g['fmin'], g['fmin'] + (g['S'].shape[1] - 1) * g['df']), pts,
                                   g['e1'], g['e2'])
        los = g['ant'] - g['center']
        got['u'] = los @ g['e1'] / np.linalg.norm(los, axis=1)        # each pulse's azimuth frequency, up to a scale
        return dict(spacing=(0.96 / got['b']['bx'], 0.96 / got['b']['by']), bounds=(-85, 85, -85, 85))

    def form(S, name):
        path = str(tmp_path / name)
        sim.write_cphd(path, col, S, LAT, LON, H0, heading=HEADING, scene=160.0)
        return planes.deramp(fastsar.form_cphd(path, axes='los', window=False, backend='cpu', _plan=plan))
    ref = form(S, 'ref.cphd')
    b, u, fr = got['b'], got['u'], planes._frame(ref)
    assert len(u) == len(S)
    tg = io.ecf_to_local(sim.to_ecf(TG, LAT, LON, H0, HEADING), ref['meta'])
    slant = lambda o: planes.to_slant(o, (0.8 / b['ba'], 0.8 / b['br']), margin=24)
    refs = {'ground': ref, 'slant': slant(ref)}
    m0 = {k: _az_metrics(o, tg, fr) for k, o in refs.items()}
    vp = (2 * u - u.max() - u.min()) / (u.max() - u.min())            # -1 .. 1 across the aperture's azimuth band
    tp = 2 * np.arange(len(S)) / (len(S) - 1) - 1                     # -1 .. 1 across the pulses
    for a in (8.0, -14.0):
        # a quadratic in the pulse's azimuth frequency: the model of quadratic, with the sign and scale of the pulses
        bad = form(S * np.exp(1j * a * vp ** 2)[:, None], 'bad.cphd')
        for k, o in (('ground', bad), ('slant', slant(bad))):
            assert o['image'].shape == refs[k]['image'].shape
            m1 = _az_metrics(o, tg, fr)
            # the slant image is zero beyond the ground image, which cuts the blurred response of the targets
            # next to it: what is cut off is lost, so those (two of seven) are left out
            ok = m1[:, 3] == 41
            assert ok.sum() >= (5 if k == 'slant' else 7) and np.all(m1[ok, 0] > 1.5 * m0[k][ok, 0]) and np.all(m1[ok, 1] > -6)
            out, info = af.image_autofocus(o['image'], 'quadratic')
            assert info['applied'] and abs(info['edge_rad'] / a - 1) < 0.05, (k, a, info['edge_rad'])
            assert not out[o['image'] == 0].any()
            m2 = _az_metrics(o, tg, fr, out)
            assert np.abs(m2[ok, 0] / m0[k][ok, 0] - 1).max() < 0.02, (k, a, m2[:, 0] / m0[k][:, 0])
            assert np.abs(m2[ok, 1] - m0[k][ok, 1]).max() < 0.5, (k, a, m2[:, 1] - m0[k][:, 1])
            assert np.abs(m2[ok, 2] / m0[k][ok, 2] - 1).max() < 0.02, (k, a, m2[:, 2] / m0[k][:, 2])
    # a quadratic in the pulse index: with squint and a curved track the pulses are not evenly spaced in azimuth
    # frequency, so cubic and higher terms come with it; quadratic takes the quadratic part (same sign, about the
    # same scale) and the 'pga' mode the rest
    bad = form(S * np.exp(1j * 8.0 * tp ** 2)[:, None], 'bad.cphd')
    out, info = af.image_autofocus(bad['image'], 'quadratic')
    assert info['applied'] and abs(info['edge_rad'] / 8.0 - 1) < 0.1, info['edge_rad']
    mq = _az_metrics(bad, tg, fr, out)
    assert np.abs(mq[:, 0] / m0['ground'][:, 0] - 1).max() < 0.03 and np.all(mq[:, 1] > m0['ground'][:, 1] + 1.5)
    out, info = af.image_autofocus(bad['image'], 'pga')
    mp = _az_metrics(bad, tg, fr, out)
    assert info['applied'] and np.abs(mp[:, 0] / m0['ground'][:, 0] - 1).max() < 0.02
    assert np.abs(mp[:, 1] - m0['ground'][:, 1]).max() < 0.6, mp[:, 1] - m0['ground'][:, 1]


def test_jax_correction_in_column_blocks(monkeypatch):
    """With little device memory the correction of a JAX image runs in narrower column blocks: the same image, its
    zeros kept exactly. The values agree to float32 rounding and not to the bit: an FFT library transforms several
    columns at a time in vector registers and the rest one by one, so a column's rounding depends on the width of
    the block it is in (seen on AVX2 and ARM hosts, not on an AVX-512 one)."""
    jnp = pytest.importorskip('jax.numpy')
    rng = np.random.default_rng(31)
    x = (rng.normal(size=(300, 257)) + 1j * rng.normal(size=(300, 257))).astype(np.complex64)
    x[:40, :50] = 0
    phi = 0.7 * np.linspace(-1, 1, 300) ** 2
    monkeypatch.delenv('FASTSAR_JAX_BLOCK_MB', raising=False)
    ref = np.asarray(af._apply(jnp.asarray(x), phi))
    monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', '0.5')                # 27 columns a block
    got = np.asarray(af._apply(jnp.asarray(x), phi))
    assert np.array_equal(got == 0, ref == 0) and not got[:40, :50].any() and not np.array_equal(got, x)
    assert np.abs(got - ref).max() < 1e-5 * np.abs(ref).max()
    monkeypatch.setenv('FASTSAR_JAX_BLOCK_MB', 'none')
    with pytest.raises(ValueError):
        af._apply(jnp.asarray(x), phi)
