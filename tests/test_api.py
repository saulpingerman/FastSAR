"""form_image on a simulated spotlight scene: every available backend against the plain JAX program, and polar
format against factorized backprojection (amplitude, since polar format's residual is larger)."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

import fastsar
from fastsar import sim

# the backends fastsar.available_backends() can return; those this machine lacks are skipped (or fail with --require)
BACKENDS = [pytest.param('tpu', marks=pytest.mark.tpu), pytest.param('cuda', marks=pytest.mark.cuda), 'cpu']
GRID = dict(nx=128, ny=128, spx=0.5, spy=0.5, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))   # the simulator's track runs along y, looking along +x


def rel_db(a, b):
    return 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))


@pytest.fixture(scope='module')
def scene():
    rng = np.random.default_rng(1)
    col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
    pos = np.stack([rng.uniform(-25, 25, 40), rng.uniform(-25, 25, 40), np.zeros(40)], 1)
    amp = rng.standard_normal(40) + 1j * rng.standard_normal(40)
    S = sim.simulate_brute(col, pos, amp).astype(np.complex64)
    print('backends here:', fastsar.available_backends())
    return SimpleNamespace(col=col, S=S)


@pytest.fixture(scope='module')
def ref(scene):
    c = scene.col
    return fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, backend='jax', T=16, levels=2)


@pytest.mark.parametrize('b', BACKENDS)
def test_backend_vs_jax(scene, ref, b):
    c = scene.col
    img = fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, backend=b, T=16, levels=2)
    e = rel_db(img, ref)
    print(f'{b:5s} vs jax: {e:.1f} dB')
    assert e < -60, (b, e)


@pytest.mark.cuda
def test_cuda_float16(scene):
    c, S = scene.col, scene.S
    ref32 = fastsar.form_image(S, c.ant, c.fmin, c.df, **GRID, backend='jax', T=32, levels=2)
    img = fastsar.form_image(S, c.ant, c.fmin, c.df, **GRID, backend='cuda', precision='float16', T=32, levels=2)
    e = rel_db(img, ref32)
    print(f'cuda float16 vs jax: {e:.1f} dB')
    assert e < -45, e


def test_pfa(scene, ref):
    c = scene.col
    pf = fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, algorithm='pfa', pfa_guard=10.0)
    cc = np.corrcoef(np.abs(pf).ravel(), np.abs(ref).ravel())[0, 1]
    print(f'pfa amplitude correlation with ffbp: {cc:.3f}')
    assert cc > 0.9, cc


def test_former_reuse(scene):
    """A reused former gives the same image as form_image, and the second call skips setup."""
    c, S = scene.col, scene.S
    b0 = fastsar.available_backends()[0]
    former = fastsar.ImageFormer(c.ant, c.fmin, c.df, S.shape[1], **GRID, backend=b0, T=16, levels=2)
    former(S)
    t = time.perf_counter()
    img2 = former(S)
    t2 = time.perf_counter() - t
    ref_b = fastsar.form_image(S, c.ant, c.fmin, c.df, **GRID, backend=b0, T=16, levels=2)
    print(f'ImageFormer ({b0}) vs form_image: {rel_db(img2, ref_b):.1f} dB; second call {t2:.3f} s')
    assert rel_db(img2, ref_b) < -100


def test_stage(scene):
    """former(former.stage(S)) equals former(S): on the JAX backend the staged history is the scaled device planes,
    on the cpu backend stage returns S itself."""
    c, S = scene.col, scene.S
    for b in ('jax', 'cpu'):
        former = fastsar.ImageFormer(c.ant, c.fmin, c.df, S.shape[1], **GRID, backend=b, T=16, levels=2)
        staged = former.stage(S)
        assert (staged is S) == (b == 'cpu')
        img, img_staged = former(S), former(staged)
        assert np.array_equal(img, img_staged), b
        print(f'ImageFormer.stage ({b}): identical images')


@pytest.fixture(scope='module')
def ref4(scene):
    c = scene.col
    return fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, backend='jax', T=16, levels=4)


@pytest.mark.parametrize('b', BACKENDS)
def test_four_levels(scene, ref4, b):
    """Four levels (a grid that three cannot split falls back to more): every backend against the JAX program."""
    c = scene.col
    e = rel_db(fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, backend=b, T=16, levels=4), ref4)
    print(f'{b:5s} four levels vs jax: {e:.1f} dB')
    assert e < -60, (b, e)


def test_four_levels_chosen(scene):
    c = scene.col
    assert fastsar.ImageFormer(c.ant, c.fmin, c.df, scene.S.shape[1], 128, 128 * 600, 0.5, 0.5, GRID['e1'], GRID['e2'],
                               backend='cpu', T=16).levels == 4


def test_pfa_support(scene):
    """Polar format on the inscribed rectangle: a narrower raster than the union of the pulses' bands, and an
    image that still matches backprojection to the reference's level."""
    import fastsar.pfa2 as pfa2
    from fastsar.sim import Collect
    c = scene.col
    col = Collect(fmin=c.fmin, df=c.df, K=c.K, ant=c.ant, res=c.res)
    gu = pfa2.geometry(col, GRID['nx'], GRID['ny'], GRID['spx'], GRID['spy'], e1=GRID['e1'], e2=GRID['e2'], guard=10.0)
    gi = pfa2.geometry(col, GRID['nx'], GRID['ny'], GRID['spx'], GRID['spy'], e1=GRID['e1'], e2=GRID['e2'], guard=10.0, support='inscribed')
    assert gi['nkr'] <= gu['nkr'] and gi['nka'] <= gu['nka'] and (gi['nkr'] < gu['nkr'] or gi['nka'] < gu['nka'])
    img = fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, algorithm='pfa', pfa_guard=10.0, pfa_support='inscribed')
    ref = fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, algorithm='pfa', pfa_guard=10.0)
    assert img.shape == ref.shape and np.isfinite(img).all()
    e = rel_db(img, ref)
    print(f'pfa inscribed against union: raster {gi["nkr"]} x {gi["nka"]} vs {gu["nkr"]} x {gu["nka"]}, difference {e:.1f} dB')
    assert e < -15
    with pytest.raises(ValueError, match='support'):
        fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, algorithm='pfa', pfa_guard=10.0, pfa_support='circumscribed')


@pytest.fixture(scope='module')
def wide_band():
    """An image narrow in range from a wide-band collection (24,000 samples for an image of 40 m): points in the
    image, and clutter over 3 km of range around it."""
    rng = np.random.default_rng(3)
    res, r0, K, P = 0.5, 30e3, 24000, 384
    df = sim.C / (2 * res) / K
    y = (np.arange(P) - (P - 1) / 2) * (r0 * sim.C / 9.6e9 / (2 * res) / P)
    psi = np.radians(40.0)
    ant = np.stack([np.full(P, -r0 * np.cos(psi)), y, np.full(P, r0 * np.sin(psi))], 1)
    col = sim.Collect(fmin=9.6e9 - (K // 2) * df, df=df, K=K, ant=ant, res=res)
    pts = np.array([[0.0, 0.0, 0], [11.3, -7.1, 0], [-14.2, 12.4, 0], [15.1, 16.2, 0], [-9.3, -15.7, 0]])
    cl = np.stack([(rng.random(4000) - 0.5) * 3000, (rng.random(4000) - 0.5) * 60, np.zeros(4000)], 1)
    amp = np.concatenate([np.full(5, 30.0), (rng.standard_normal(4000) + 1j * rng.standard_normal(4000)) / np.sqrt(2)])
    S = sim.simulate(col, np.concatenate([pts, cl]), amp).astype(np.complex64)
    return SimpleNamespace(col=col, S=S, grid=dict(nx=96, ny=96, spx=0.4, spy=0.4, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0)))


@pytest.mark.needs('finufft')
@pytest.mark.parametrize('window', [True, False])
def test_range_gate(wide_band, window):
    """The phase history gated in range to the image's swath: fewer samples, a first level that decimates less,
    and the image of the whole history to the former's accuracy."""
    c = wide_band.col
    args = (c.ant, c.fmin, c.df, c.K)
    full = fastsar.ImageFormer(*args, **wide_band.grid, backend='cpu', window=window, range_gate=False)
    gated = fastsar.ImageFormer(*args, **wide_band.grid, backend='cpu', window=window, range_gate=True)
    assert full.range_gate is None and fastsar.ImageFormer(*args, **wide_band.grid, backend='cpu').range_gate is None
    g = gated.range_gate
    assert g['K'] == c.K and g['kept'] < c.K // 20 and 'range gate' in g['note']
    assert gated._plan['levels'][0]['Dk'] < 4 < full._plan['levels'][0]['Dk']
    a, b = full(wide_band.S), gated(wide_band.S)
    db = rel_db(b, a)
    print(f'window {window}: gated to {g["kept"]} of {c.K} samples; against the whole history {db:.1f} dB')
    assert db < -55                                                  # measured -68 dB
    assert np.array_equal(fastsar.form_image(wide_band.S, *args[:3], **wide_band.grid, backend='cpu', window=window, range_gate=True), b)
    with pytest.raises(ValueError, match='range_gate'):
        fastsar.ImageFormer(*args, **wide_band.grid, backend='cpu', range_gate='yes')
    bad = wide_band.S.copy()
    bad[7, 100] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        gated(bad)
    with pytest.raises(ValueError, match='phase history must be'):
        gated(wide_band.S[:, :-1])
    # an image that fills the swath is not gated
    small = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
    assert fastsar.ImageFormer(small.ant, small.fmin, small.df, small.K, **GRID, backend='cpu', range_gate=True).range_gate is None


@pytest.mark.cuda
@pytest.mark.needs('finufft')
def test_cuda_first_level_window(wide_band, monkeypatch):
    """A range decimation whose filter window no variant of the cuda first-level kernel can hold in shared memory
    (an image narrow in range from a wide-band collection: refused through 0.2.0): the history is gated in range
    first, and the image is the cpu backend's of the whole history."""
    from fastsar import ffbp_cuda
    c = wide_band.col
    args = (c.ant, c.fmin, c.df, c.K)
    monkeypatch.setenv('FASTSAR_CUDA_SMEM_KB', '99')                 # the shared memory of an Ada-generation card
    with pytest.raises(ValueError, match='first-level window'):
        fastsar.ImageFormer(*args, **wide_band.grid, backend='cuda', range_gate=False)(wide_band.S)
    former = fastsar.ImageFormer(*args, **wide_band.grid, backend='cuda')
    assert not ffbp_cuda.first_level_fits(fastsar.ImageFormer(*args, **wide_band.grid, backend='cpu')._plan)
    assert former.range_gate is not None and 'shared memory' in former.range_gate['note']
    img = former(wide_band.S)
    ref = fastsar.ImageFormer(*args, **wide_band.grid, backend='cpu', range_gate=False)(wide_band.S)
    db = rel_db(img, ref)
    print(f'cuda, gated to {former.range_gate["kept"]} of {c.K} samples, against the cpu image of the whole history: {db:.1f} dB')
    assert db < -55                                                  # measured -68 dB
    import cupy as cp
    assert np.allclose(former(cp.asarray(wide_band.S)), img, rtol=1e-4, atol=1e-4 * np.abs(img).max())       # a device history
    bad = wide_band.S.copy()
    bad[7, 100] = np.inf
    with pytest.raises(ValueError, match='non-finite'):
        former(bad)
    # where the window fits, nothing is gated
    monkeypatch.delenv('FASTSAR_CUDA_SMEM_KB')
    small = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
    assert fastsar.ImageFormer(small.ant, small.fmin, small.df, small.K, **GRID, backend='cuda').range_gate is None


@pytest.mark.cuda
def test_cuda_range_gate_transfer(monkeypatch):
    """The device range gate with small blocks and a stream that is behind (work queued ahead of every block):
    every gated history is the cpu's and the same bits each time, on the default stream and on a caller's. The gate
    first staged its blocks in two page-locked buffers that it rewrote while the asynchronous copy of the block
    before last was still queued, and then formed an image from the wrong pulses; and a transfer that delivers
    other samples is refused, not imaged."""
    import cupy as cp
    from fastsar import ffbp_cuda
    rng = np.random.default_rng(8)
    res, r0, K, P = 0.5, 30e3, 6000, 768
    df = sim.C / (2 * res) / K
    y = (np.arange(P) - (P - 1) / 2) * (r0 * sim.C / 9.6e9 / (2 * res) / P)
    ant = np.stack([np.full(P, -r0 * np.cos(np.radians(40.0))), y, np.full(P, r0 * np.sin(np.radians(40.0)))], 1)
    grid = dict(nx=96, ny=96, spx=0.4, spy=0.4, e1=(0.0, 1.0, 0.0), e2=(1.0, 0.0, 0.0))
    args = (ant, 9.6e9 - (K // 2) * df, df, K)
    S = (rng.standard_normal((P, K), np.float32) + 1j * rng.standard_normal((P, K), np.float32)).astype(np.complex64)
    ref = fastsar.ImageFormer(*args, **grid, backend='cpu', range_gate=True)._gate(S)
    former = fastsar.ImageFormer(*args, **grid, backend='cuda', range_gate=True)
    monkeypatch.setattr(ffbp_cuda, 'BLOCK_BYTES', 4 << 20)           # 34 pulses a block: 23 blocks
    heavy = cp.random.standard_normal((3000, 3000), dtype=cp.float32)
    copy_rows = ffbp_cuda._copy_rows

    def behind(dst, src, nth):                                       # the device still busy when the host moves on
        heavy @ heavy
        return copy_rows(dst, src, nth)
    monkeypatch.setattr(ffbp_cuda, '_copy_rows', behind)
    first, worst = None, -np.inf
    for k in range(24):
        if k % 2:
            with cp.cuda.Stream(non_blocking=True) as stream:        # a caller that works on a stream of its own
                out = former._gate(S)
                stream.synchronize()
        else:
            out = former._gate(S)
        out = cp.asnumpy(out)
        first = out if first is None else first
        worst = max(worst, rel_db(out, ref))
        assert np.array_equal(out, first), k
    print(f'cuda range gate, 24 times in 23 blocks behind a busy stream: at worst {worst:.1f} dB from the cpu gate')
    assert worst < -90
    img, cpu = former(S), fastsar.ImageFormer(*args, **grid, backend='cpu', range_gate=True)(S)
    assert rel_db(img, cpu) < -55
    # the check across the transfer: a block that arrives as other samples
    def torn(dst, src):
        dst.data.copy_from_host(src.ctypes.data, src.nbytes)
        dst[-1] = dst[0]                                              # a pulse of another place in the block
    monkeypatch.setattr(ffbp_cuda, '_to_device_now', torn)
    with pytest.raises(RuntimeError, match='did not reach the GPU'):
        former(S)


@pytest.mark.cuda
def test_device_copy_is_complete_when_it_returns(scene):
    """_xp.device on the image a cuda former returns (page-locked host memory) while the device is busy: the copy is
    complete when the call returns, so the host image written right afterwards does not reach it, on the default
    stream and on a caller's (cupy.asarray reads a page-locked source after it returns: through 0.2.0 the device
    copy then took what the host wrote)."""
    import contextlib
    import cupy as cp
    from fastsar import _xp
    from fastsar.resample_cuda import host_empty
    c = scene.col
    heavy = cp.random.standard_normal((4000, 4000), dtype=cp.float32)
    rng = np.random.default_rng(2)
    big = (rng.standard_normal((1024, 1024), np.float32) + 1j * rng.standard_normal((1024, 1024), np.float32)).astype(np.complex64)
    for own in (False, True):
        for k in range(4):
            if k % 2:
                img = fastsar.form_image(scene.S, c.ant, c.fmin, c.df, **GRID, backend='cuda')
            else:
                img = host_empty(big.shape, big.dtype)
                img[:] = big
            want = img.copy()
            with cp.cuda.Stream(non_blocking=True) if own else contextlib.nullcontext():
                for _ in range(3):
                    heavy @ heavy                                    # work queued ahead of the copy
                dev = _xp.device(img, 'cuda')
                img[:] = 0
                assert np.array_equal(cp.asnumpy(dev), want), (own, k)
    assert _xp.device(dev, 'cuda') is dev
    assert np.array_equal(cp.asnumpy(_xp.device(want[::2, 1::3], 'cuda')), want[::2, 1::3])      # any layout
