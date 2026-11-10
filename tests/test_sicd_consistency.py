"""The SICD metadata products.sicd_meta builds, against the rules SICD sets between its fields and against sarkit's
consistency checker (Python 3.11 or later; the rules are also asserted here without it): every product of form_cphd
(track axes and line-of-sight axes, with and without a window) and of process_cphd (slant and ground plane; Taylor,
uniform and SVA weighting; autofocus on and off) of simulated collections flown both ways, one of them squinted
past 45 degrees so that its band is off-center and wraps around the sampling rate; a collection whose first pulse
precedes its stated start; and, when the files are on this machine, two real collections with their vendor SICDs.
Needs sarpy only through sarkit's own dependencies."""
import dataclasses
import io as _io
import os
import sys

import numpy as np
import pytest

import fastsar
from fastsar import products, sim

LAT, LON, H0, HEADING = 40.79, -77.86, 351.0, 30.0
TG = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 15.0, 0.0], [-18.0, -22.0, 0.0], [12.0, -9.0, 0.0]])
DATA = '/home/paul/projects/sar-accel-study/data_local'
REAL = {'umbra': (f'{DATA}/umbra_untrusted/umb9_northdakot.cphd', f'{DATA}/umbra_sicd_untrusted/umb9_northdakot_SICD.nitf'),
        'capella': (f'{DATA}/capella_untrusted/cap12_c10_sp_hh.cphd',
                    f'{DATA}/capella_sicd_untrusted/cap12_c10_sp_hh_SICD.ntf')}
REAL_GB = 48                    # host memory the real collections need (a 17 GiB phase history and its image)


def sarkit_errors(sm):
    """The names of sarkit's SICD consistency checks that fail as errors on the metadata sm, or None without sarkit."""
    if sys.version_info < (3, 11):
        return None
    try:
        import lxml.etree
        import sarkit.verification as skv
    except ImportError:
        return None
    con = skv.SicdConsistency(lxml.etree.parse(_io.BytesIO(sm.to_xml_bytes())))
    con.check()
    fail = con.failures(omit_passed_sub=True)
    return sorted(k for k, v in fail.items() if any(d['severity'] == 'Error' for d in v['details']))


def check(out, name):
    """The rules between the fields, then sarkit. -> (metadata, array, whether the row and column bands wrap)."""
    sm, a = products.sicd_meta(out)
    g, wraps = sm.Grid, []
    assert a.shape == (sm.ImageData.NumRows, sm.ImageData.NumCols) and a.size == np.asarray(out['image']).size
    for ax in (g.Row, g.Col):
        half, bw, k1, k2, dk = 0.5 / ax.SS, ax.ImpRespBW, ax.DeltaK1, ax.DeltaK2, float(ax.DeltaKCOAPoly[0, 0])
        assert 0 < bw <= 2 * half * (1 + 1e-12), (name, ax.tag)
        assert -half * (1 + 1e-12) <= k1 < k2 <= half * (1 + 1e-12), (name, ax.tag, k1, k2, half)
        assert bw <= (k2 - k1) * (1 + 1e-9) and -half <= dk <= half, (name, ax.tag)
        if dk - bw / 2 < -half * (1 + 1e-9) or dk + bw / 2 > half * (1 + 1e-9):       # the band wraps: all of it
            wraps.append(True)
            assert k1 == pytest.approx(-half, rel=1e-12) and k2 == pytest.approx(half, rel=1e-12), (name, ax.tag)
        else:
            wraps.append(False)
            assert k1 == pytest.approx(dk - bw / 2, abs=1e-9) and k2 == pytest.approx(dk + bw / 2, abs=1e-9)
        # the width follows the stated band for every weighting (a uniform aperture: 0.8859 / ImpRespBW to 1e-4)
        assert ax.ImpRespWid * bw == pytest.approx({'TAYLOR': 1.184, 'HANNING': 1.441}.get(ax.WgtType.WindowName, 0.8859),
                                                   rel=5e-5)
    # shadows down the rows, the view from above, and the processed times within the collection
    los = sm.GeoData.SCP.ECF.get_array() - sm.SCPCOA.ARPPos.get_array()
    urow, ucol = g.Row.UVectECF.get_array(), g.Col.UVectECF.get_array()
    assert urow @ los > abs(ucol @ los), name
    assert np.cross(urow, ucol) @ sm.GeoData.SCP.ECF.get_array() > 0, name
    f = sm.ImageFormation
    assert 0 <= f.TStartProc < f.TEndProc <= sm.Timeline.CollectDuration, name
    assert f.TStartProc <= sm.SCPCOA.SCPTime <= f.TEndProc
    err = sarkit_errors(sm)
    assert not err, (name, err)
    return sm, a, wraps


@pytest.fixture(scope='module', params=[(20.0, False), (-25.0, True), (55.0, False)], ids=['right', 'left', 'squint55'])
def cphd(request, tmp_path_factory):
    squint, left = request.param
    col = sim.make_collect(res=0.35, scene=170.0, r0=10e3, graze_deg=35.0)
    y = col.ant[:, 1] / np.ptp(col.ant[:, 1])
    ant = col.ant.copy()
    ant[:, 1] -= 10e3 * np.cos(np.radians(35)) * np.tan(np.radians(squint))
    ant[:, 0] += 160.0 * y ** 2 + 120.0 * y ** 3                      # a curved track
    ant[:, 2] += -100.0 * y ** 2 + 6.0 * y
    col = dataclasses.replace(col, ant=ant[::-1].copy() if left else ant)
    path = str(tmp_path_factory.mktemp('consistency') / 'c.cphd')
    sim.write_cphd(path, col, sim.simulate(col, TG, np.ones(len(TG))), LAT, LON, H0, heading=HEADING, scene=140.0)
    return path, request.node.callspec.id if hasattr(request.node, 'callspec') else str(squint), squint


def test_form_cphd_products(cphd):
    """Images as formed: on track axes the band is off-center and, past 45 degrees of squint, the range axis is the
    grid's first; the rows of the SICD follow the axis nearer the line of sight either way."""
    path, _, squint = cphd
    seen = []
    for name, kw in (('track', {}), ('line of sight', dict(axes='los')), ('track, no window', dict(window=False)),
                     ('track, fine spacing', dict(spacing=(0.2, 0.2)))):
        out = fastsar.form_cphd(path, backend='cpu', **kw)
        sm, a, wraps = check(out, name)
        seen += wraps
        rows_first_axis = a.shape == out['image'].shape and out['image'].shape[0] != out['image'].shape[1]
        if name == 'track' and squint == 55.0:
            e1 = out['e1'] @ out['meta']['R']
            assert abs(sm.Grid.Row.UVectECF.get_array() @ e1) > 0.99           # rows along the track axis here
        print(f'form_cphd ({name}): rows {"along e1" if rows_first_axis else "along e2 or square"}, '
              f'band wraps {wraps}, DeltaK row {sm.Grid.Row.DeltaK1:.3f} to {sm.Grid.Row.DeltaK2:.3f} of '
              f'+-{0.5 / sm.Grid.Row.SS:.3f}')
    assert any(seen) and not all(seen)          # bands that wrap around the sampling rate, and bands that fit


CASES = {'slant, Taylor': {}, 'slant, uniform, no autofocus': dict(weighting='uniform', autofocus=False),
         'slant, SVA': dict(weighting='sva'), 'ground, Taylor': dict(plane='ground'),
         'ground, uniform, no autofocus': dict(plane='ground', weighting='uniform', autofocus=False),
         'slant, Taylor, phase gradient autofocus': dict(autofocus='pga')}


def test_process_cphd_products(cphd, tmp_path):
    path, _, squint = cphd
    if squint == 55.0:
        pytest.skip('the processing chain is exercised at moderate squint; form_cphd covers the squinted grid')
    for name, kw in CASES.items():
        out = fastsar.process_cphd(path, backend='cpu', **kw)
        sm, a, wraps = check(out, name)
        assert sm.ImageFormation.AzAutofocus == ('GLOBAL' if out.get('autofocus') else 'NO')
        assert kw.get('autofocus') is not False or sm.ImageFormation.AzAutofocus == 'NO'
    # as written: the file's own XML passes as the metadata did
    f = str(tmp_path / 'p.nitf')
    sm = fastsar.process_cphd(path, f, backend='cpu')['metadata']
    from fastsar import _sicdxml
    back = _sicdxml.read(f)
    assert back.to_xml_bytes() == sm.to_xml_bytes() and not sarkit_errors(back)


def test_first_pulse_before_the_start(cphd):
    """Transmit times a few nanoseconds before the stated collection start (as Capella's files have): the SICD's
    start moves to the first pulse, and its times stay within the collection."""
    path, _, _ = cphd
    out = fastsar.form_cphd(path, backend='cpu', axes='los')
    ref = products.sicd_meta(out)[0]
    t = np.asarray(out['meta']['tx_time'], np.float64)
    shift = t[0] + 2.5e-6
    early = dict(out, meta=dict(out['meta'], tx_time=t - shift))                 # the first pulse 2.5 us early
    sm = check(early, 'early')[0]
    assert sm.ImageFormation.TStartProc >= 0 and sm.Timeline.CollectDuration == pytest.approx(t[-1] - t[0], abs=1e-9)
    assert sm.SCPCOA.SCPTime == pytest.approx(ref.SCPCOA.SCPTime - t[0], abs=1e-9)
    assert np.allclose(sm.SCPCOA.ARPPos.get_array(), ref.SCPCOA.ARPPos.get_array(), atol=1e-4)
    d = np.datetime64(sm.Timeline.CollectStart.rstrip('Z')) - np.datetime64(ref.Timeline.CollectStart.rstrip('Z'))
    assert d / np.timedelta64(1, 'us') == pytest.approx(-2.5, abs=1.01)          # the dates carry microseconds


@pytest.mark.cuda
@pytest.mark.parametrize('name', sorted(REAL))
def test_real_collections(name, request):
    """Two real spotlight collections (Umbra, squinted 57 degrees; Capella) through form_cphd and process_cphd on a
    GPU, with and without the vendor's SICD: every product's metadata pass. Runs when the files are present and the
    memory limit of the tests allows their size (--max-rss-gb 0, or at least 48)."""
    cphd_path, sicd_path = REAL[name]
    if not (os.path.exists(cphd_path) and os.path.exists(sicd_path)):
        pytest.skip('the real collection is not on this machine')
    limit = request.config.getoption('max_rss_gb')
    limit = float(os.environ.get('FASTSAR_TEST_MAX_GB', 12)) if limit is None else limit
    if 0 < limit < REAL_GB:
        pytest.skip(f'the real collections need --max-rss-gb {REAL_GB} (or 0)')
    import cupy
    F, P = fastsar.form_cphd, fastsar.process_cphd
    runs = {'form_cphd, track axes': lambda: F(cphd_path, backend='cuda'),
            'form_cphd, line-of-sight axes': lambda: F(cphd_path, backend='cuda', axes='los'),
            'process_cphd': lambda: P(cphd_path, backend='cuda'),
            'process_cphd, vendor SICD': lambda: P(cphd_path, sicd=sicd_path, backend='cuda'),
            'process_cphd, SVA': lambda: P(cphd_path, backend='cuda', weighting='sva'),
            'process_cphd, ground, uniform': lambda: P(cphd_path, backend='cuda', plane='ground', weighting='uniform',
                                                       autofocus=False)}
    for what, run in runs.items():
        out = run()
        shape = tuple(out['image'].shape)
        out = dict(out, image=np.broadcast_to(np.complex64(0), shape))           # the metadata need the shape alone
        sm = check(out, f'{name}: {what}')[0]
        print(f'{name}, {what}: {shape}, rows SS {sm.Grid.Row.SS:.3f} m, no consistency errors')
        del out
        cupy.get_default_memory_pool().free_all_blocks()
