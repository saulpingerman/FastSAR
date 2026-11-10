"""The native CPHD reader and writer (fastsar._cphdxml) against sarpy's reader and sarkit's consistency checker:
simulated files from sim.write_cphd (spotlight, moving beam); a two-channel file with integer samples, AmpSF, added
per-vector parameters, an antenna block and a gain support array, written natively and by sarpy; io.read_cphd through
either reader; a compressed signal block; and, when present, two real collections (header, metadata and per-vector
parameters only). Needs sarpy; the consistency checks need sarkit (Python 3.11 or later)."""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from fastsar import _cphdxml, cphd, io, sim

pytestmark = pytest.mark.needs('sarpy')

NS = 'http://api.nsgreg.nga.mil/schema/cphd/1.0.1'
NS11 = 'http://api.nsgreg.nga.mil/schema/cphd/1.1.0'
REAL = ('/home/paul/projects/sar-accel-study/data_local/umbra_untrusted/umb9_northdakot.cphd',
        '/home/paul/projects/sar-accel-study/data_local/capella_untrusted/cap12_c10_sp_hh.cphd')
# the metadata FastSAR reads (io.read_cphd, cphd._image_area, rgiqe.Radiometry), as paths under the names both readers use
FIELDS = ('Data.SignalArrayFormat', 'Data.SignalCompressionID', 'Data.NumBytesPVP', 'Global.DomainType', 'Global.SGN',
          'Global.Timeline.CollectionStart', 'CollectionID.CollectorName', 'CollectionID.CoreName',
          'CollectionID.RadarMode.ModeType', 'SceneCoordinates.IARP.ECF', 'SceneCoordinates.IARP.LLH.HAE',
          'SceneCoordinates.ReferenceSurface.Planar.uIAX', 'SceneCoordinates.ReferenceSurface.Planar.uIAY',
          'SceneCoordinates.ImageArea.X1Y1.X', 'SceneCoordinates.ImageArea.X1Y1.Y', 'SceneCoordinates.ImageArea.X2Y2.X',
          'SceneCoordinates.ImageArea.X2Y2.Y', 'SceneCoordinates.ImageGrid.IAXExtent.FirstLine',
          'SceneCoordinates.ImageGrid.IAXExtent.NumLines', 'SceneCoordinates.ImageGrid.IAXExtent.LineSpacing',
          'SceneCoordinates.ImageGrid.IAYExtent.FirstSample', 'SceneCoordinates.ImageGrid.IAYExtent.NumSamples',
          'SceneCoordinates.ImageGrid.IAYExtent.SampleSpacing')
LISTS = {'Data.Channels': ('Identifier', 'NumVectors', 'NumSamples', 'SignalArrayByteOffset', 'PVPArrayByteOffset'),
         'Channel.Parameters': ('Identifier', 'Polarization.TxPol', 'Polarization.RcvPol', 'FxC', 'Antenna.TxAPCId',
                                'Antenna.TxAPATId', 'Antenna.RcvAPCId', 'Antenna.RcvAPATId', 'TgtRefLevel.PTRef',
                                'NoiseLevel.PNRef', 'NoiseLevel.BNRef'),
         'SceneCoordinates.ImageAreaCornerPoints': ('Lat', 'Lon'),
         'Antenna.AntCoordFrame': ('Identifier', 'XAxisPoly.X', 'XAxisPoly.Y', 'XAxisPoly.Z', 'YAxisPoly.X', 'YAxisPoly.Y',
                                   'YAxisPoly.Z'),
         'Antenna.AntPhaseCenter': ('Identifier', 'ACFId'),
         'Antenna.AntPattern': ('Identifier', 'EB.DCXPoly', 'EB.DCYPoly', 'Array.GainPoly', 'Array.AntGPId',
                                'Element.GainPoly', 'Element.AntGPId'),
         'SupportArray.AntGainPhase': ('Identifier', 'X0', 'Y0', 'XSS', 'YSS')}


def get(m, path):
    for k in path.split('.'):
        if m is None:
            return None
        m = getattr(m, k, None)
    return m


def same(a, b, what):
    """A value from sarpy's metadata against the native one."""
    if a is None or b is None:
        assert a is None and b is None, (what, a, b)
    elif hasattr(a, 'get_array'):
        # to an ulp: sarpy renormalizes unit vectors
        np.testing.assert_allclose(np.asarray(a.get_array(), np.float64), b.get_array(), rtol=1e-14, atol=0, err_msg=what)
    elif isinstance(a, np.datetime64):
        assert str(a.astype('datetime64[us]')) == _cphdxml.start_time(b), (what, a, b)
    elif isinstance(a, str):
        assert a == b and isinstance(b, str), (what, a, b)
    elif what.endswith('LLH.HAE'):                  # sarpy recomputes the height from the ECF point
        assert abs(a - b) < 1e-6, (what, a, b)
    else:
        assert a == b and type(b) in (int, float, bool), (what, a, b)


def compare(path, samples=True):
    """Everything FastSAR reads from the file, sarpy's reader against the native one. -> number of values compared."""
    from sarpy.io.phase_history.converter import open_phase_history
    rs, rn = open_phase_history(path), _cphdxml.open_cphd(path)
    ms, mn = rs.cphd_meta, rn.cphd_meta
    assert str(rs.cphd_version) == rn.cphd_version
    count = 0
    for k in ('XML_BLOCK_SIZE', 'XML_BLOCK_BYTE_OFFSET', 'PVP_BLOCK_SIZE', 'PVP_BLOCK_BYTE_OFFSET', 'SIGNAL_BLOCK_SIZE',
              'SIGNAL_BLOCK_BYTE_OFFSET', 'SUPPORT_BLOCK_SIZE', 'SUPPORT_BLOCK_BYTE_OFFSET', 'CLASSIFICATION', 'RELEASE_INFO'):
        assert getattr(rs.cphd_header, k, None) == getattr(rn.cphd_header, k, None), k
    assert rs.cphd_details.get_cphd_bytes() == rn.cphd_details.get_cphd_bytes()
    for f in FIELDS:
        same(get(ms, f), get(mn, f), f)
        count += get(mn, f) is not None
    for lst, fields in LISTS.items():
        a, b = get(ms, lst), get(mn, lst)
        assert (a is None or len(a) == 0) == (b is None), lst
        for i in range(len(b or [])):
            for f in fields:
                same(get(a[i], f), get(b[i], f), f'{lst}[{i}].{f}')
                count += get(b[i], f) is not None
    for c, ch in enumerate(mn.Data.Channels):
        a, b = rs.read_pvp_array(c), rn.read_pvp_array(c)
        assert set(a.dtype.names) == set(b.dtype.names) and a.dtype.itemsize == b.dtype.itemsize and a.shape == b.shape
        for n in a.dtype.names:
            assert a.dtype.fields[n][1] == b.dtype.fields[n][1] and a.dtype[n] == b.dtype[n].newbyteorder('>'), (c, n)
            assert b.dtype[n].base.byteorder in '=|'
            assert a[n].tobytes() == b[n].astype(a.dtype[n].base).tobytes(), (c, n)         # bit for bit, NaN included
            v = rn.read_pvp_variable(n, ch.Identifier)
            assert v.tobytes() == b[n].tobytes()
            count += 1
        assert rn.read_pvp_variable('NoSuchParameter', c) is None
        if rn.compression is None:
            P, K = ch.NumVectors, ch.NumSamples
            sel = ((0, P), (0, K)) if samples else ((P // 2, P // 2 + 2), (K // 2, K // 2 + 64))
            np.testing.assert_array_equal(rs.read_chip(*sel, index=c).reshape(sel[0][1] - sel[0][0], -1),
                                          rn.read_chip(*sel, index=c))
    for s in mn.Data.SupportArrays or []:
        a, b = rs.read_support_array(s.Identifier), rn.read_support_array(s.Identifier)
        # sarpy gives a format of like components (Gain=F4;Phase=F4;) as a trailing axis, the native reader as fields
        assert a.shape[:2] == b.shape and len(b.dtype.names) == (a.shape[2] if a.ndim == 3 else len(a.dtype.names))
        for i, n in enumerate(b.dtype.names):
            ai = a[..., i] if a.ndim == 3 else a[n]
            assert ai.tobytes() == b[n].astype(ai.dtype).tobytes(), (s.Identifier, n)
        count += 1
    return count


def consistent(path, need):
    """sarkit's CPHD consistency checks: no errors. -> the names of the warnings."""
    if sys.version_info < (3, 11):
        pytest.skip('sarkit needs Python 3.11 or later')
    skv = need('sarkit.verification')
    with open(path, 'rb') as fh:
        con = skv.CphdConsistency.from_file(fh)
    con.check()
    fail = con.failures(omit_passed_sub=True)
    err = sorted(k for k, v in fail.items() if any(d['severity'] == 'Error' for d in v['details']))
    assert not err, {k: fail[k] for k in err}
    return sorted(set(fail))


def read_both(path, monkeypatch, **kw):
    """io.read_cphd through the native reader and through sarpy's."""
    from sarpy.io.phase_history.converter import open_phase_history
    native = io.read_cphd(path, meta=True, **kw)
    with monkeypatch.context() as mp:
        mp.setattr(io, '_open_cphd', open_phase_history)
        through = io.read_cphd(path, meta=True, **kw)
    return native, through


def assert_same_read(native, through):
    (c1, m1), (c2, m2) = native, through
    np.testing.assert_array_equal(c1['S'], c2['S'])
    assert set(c1) == set(c2) and set(m1) == set(m2)
    for d1, d2 in ((c1, c2), (m1, m2)):
        for k in d1:
            if isinstance(d1[k], np.ndarray):
                np.testing.assert_array_equal(d1[k], d2[k], err_msg=k)
            else:
                assert d1[k] == d2[k], (k, d1[k], d2[k])


@pytest.fixture(scope='module')
def files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('native_cphd')
    col = sim.make_collect(res=0.5, scene=60.0, r0=6e3, graze_deg=35.0)
    tg = np.array([[0.0, 0.0, 0.0], [10.0, -12.0, 0.0], [-12.0, 8.0, 0.0]])
    S = sim.simulate_brute(col, tg, np.ones(len(tg))).astype(np.complex64)
    spot, moving = str(tmp / 'spot.cphd'), str(tmp / 'moving.cphd')
    sim.write_cphd(spot, col, S, 40.7934, -77.86, 351.0, heading=30.0, tropo=3e-9)
    srp = np.zeros_like(col.ant)
    srp[:, 0] = np.linspace(-5.0, 5.0, len(srp))
    sim.write_cphd(moving, col, S, -33.1, 151.2, 20.0, heading=200.0, srp=srp, pol='HV')
    return tmp, col, S, spot, moving


def test_formats():
    f = _cphdxml.format_dtype
    assert f('F8') == np.dtype('>f8') and f('I2', '=') == np.dtype('i2') and f('S24') == np.dtype('S24')
    assert f('X=F8;Y=F8;Z=F8;') == np.dtype(('>f8', 3)) and f('DCX=F8;DCY=F8;') == np.dtype(('>f8', 2))
    assert f('Gain=F4;Phase=F4;') == np.dtype([('Gain', '>f4'), ('Phase', '>f4')])
    assert f('A=U1;B=I8;') == np.dtype([('A', '>u1'), ('B', '>i8')]) and f('CF8') == np.dtype('>c8')
    assert f('CI4') == np.dtype([('real', '>i2'), ('imag', '>i2')])
    with pytest.raises(ValueError, match='binary format'):
        f('F3')


def test_simulated_files(files, need, monkeypatch):
    """sim.write_cphd's files: read alike by sarpy and natively, consistent by sarkit, and read_cphd and the image
    area the same through either reader."""
    _, col, S, spot, moving = files
    for path in (spot, moving):
        n = compare(path)
        np.testing.assert_array_equal(_cphdxml.open_cphd(path).read_chip(), S)
        warn = consistent(path, need)
        native, through = read_both(path, monkeypatch)
        assert_same_read(native, through)
        from sarpy.io.phase_history.converter import open_phase_history
        area = cphd._image_area(path, native[1])
        with monkeypatch.context() as mp:
            mp.setattr(io, '_open_cphd', open_phase_history)
            area2 = cphd._image_area(path, native[1])
        np.testing.assert_allclose(area[0], area2[0], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(area[1], area2[1])
        print(f'{os.path.basename(path)}: {n} metadata values and per-vector parameters equal to sarpy; sarkit: no errors'
              + (f', warnings {warn}' if warn else ''))
    assert native[1]['mode'] == 'STRIPMAP' and native[1]['polarization'] == 'HV'
    assert native[1]['start'] == '2026-01-01T00:00:00.000000'


def poly2(tag, c):
    c = np.atleast_2d(c)
    return (f'<{tag} order1="{c.shape[0] - 1}" order2="{c.shape[1] - 1}">' + ''.join(
        f'<Coef exponent1="{i}" exponent2="{j}">{float(c[i, j])!r}</Coef>' for i in range(c.shape[0]) for j in range(c.shape[1]))
        + f'</{tag}>')


def poly1(tag, c):
    return f'<{tag} order1="{len(c) - 1}">' + ''.join(f'<Coef exponent1="{i}">{float(v)!r}</Coef>' for i, v in enumerate(c)) + f'</{tag}>'


def rich_xml(xml, P, K, fmt):
    """The simulated file's XML as CPHD 1.1.0, extended to two channels of format fmt with AmpSF, SIGNAL, two added parameters, an
    antenna block and a gain and phase support array."""
    q = lambda t: f'{{{NS11}}}{t}'
    root = ET.fromstring(xml.replace(NS.encode(), NS11.encode()))           # AntGPId came with CPHD 1.1.0
    sub = lambda text: ET.fromstring(f'<r xmlns="{NS11}">{text}</r>')
    data, chan, pvp = root.find(q('Data')), root.find(q('Channel')), root.find(q('PVP'))
    data.find(q('SignalArrayFormat')).text = fmt
    nb = int(data.find(q('NumBytesPVP')).text)
    words = nb // 8
    isz = {'CF8': 8, 'CI4': 4, 'CI2': 2}[fmt]
    data.find(q('NumBytesPVP')).text = str(nb + 8 * 5)
    data.find(q('NumCPHDChannels')).text = '2'
    c0 = data.find(q('Channel'))
    c1 = ET.fromstring(ET.tostring(c0))
    c1.find(q('Identifier')).text = 'second'
    c1.find(q('SignalArrayByteOffset')).text = str(P * K * isz)
    c1.find(q('PVPArrayByteOffset')).text = str(P * (nb + 40))
    data.insert(list(data).index(c0) + 1, c1)
    data.find(q('NumSupportArrays')).text = '1'
    data.extend(sub('<SupportArray><Identifier>AGP</Identifier><NumRows>5</NumRows><NumCols>7</NumCols>'
                    '<BytesPerElement>8</BytesPerElement><ArrayByteOffset>0</ArrayByteOffset></SupportArray>'))
    p0 = chan.find(q('Parameters'))
    p0.extend(sub('<Antenna><TxAPCId>APC</TxAPCId><TxAPATId>PAT</TxAPATId><RcvAPCId>APC</RcvAPCId><RcvAPATId>PAT</RcvAPATId>'
                  '</Antenna><TgtRefLevel><PTRef>0.5</PTRef></TgtRefLevel><NoiseLevel><PNRef>0.25</PNRef><BNRef>0.9</BNRef>'
                  '</NoiseLevel>'))
    p0.insert(list(p0).index(p0.find(q('SRPFixed'))) + 1, sub('<SignalNormal>false</SignalNormal>')[0])
    p1 = ET.fromstring(ET.tostring(p0))
    p1.find(q('Identifier')).text = 'second'
    p1.find(q('Polarization')).find(q('RcvPol')).text = 'H'
    chan.append(p1)
    one = lambda name, off, f='F8': f'<{name}><Offset>{off}</Offset><Size>1</Size><Format>{f}</Format></{name}>'
    pvp.insert(list(pvp).index(pvp.find(q('SRPPos'))) + 1, sub(one('AmpSF', words))[0])
    pvp.extend(sub(one('SIGNAL', words + 1, 'I8') + ''.join(
        f'<AddedPVP><Name>{n}</Name><Offset>{words + o}</Offset><Size>{s}</Size><Format>{f}</Format></AddedPVP>'
        for n, o, s, f in (('Count', 2, 1, 'U4'), ('Label', 3, 2, 'S16')))))
    xyz = lambda tag, v: f'<{tag}>' + ''.join(poly1(a, c) for a, c in zip('XYZ', v)) + f'</{tag}>'
    gp = lambda tag: (f'<{tag}>' + poly2('GainPoly', np.array([[0.0, 0.0, -30.0], [0.0, 0.0, 0.0], [-40.0, 0.0, 0.0]]))
                      + poly2('PhasePoly', np.zeros((1, 1))) + '<AntGPId>AGP</AntGPId>' + f'</{tag}>')
    root.insert(list(root).index(root.find(q('PVP'))) + 1, sub(
        '<SupportArray><AntGainPhase><Identifier>AGP</Identifier><ElementFormat>Gain=F4;Phase=F4;</ElementFormat>'
        '<X0>-0.02</X0><Y0>-0.03</Y0><XSS>0.01</XSS><YSS>0.01</YSS></AntGainPhase></SupportArray>')[0])
    root.insert(list(root).index(root.find(q('ReferenceGeometry'))) + 1, sub(
        '<Antenna><NumACFs>1</NumACFs><NumAPCs>1</NumAPCs><NumAntPats>1</NumAntPats><AntCoordFrame><Identifier>ACF</Identifier>'
        + xyz('XAxisPoly', ([1.0, 1e-3], [0.0, 2e-3], [0.0])) + xyz('YAxisPoly', ([0.0], [1.0], [0.0, -1e-3]))
        + '</AntCoordFrame><AntPhaseCenter><Identifier>APC</Identifier><ACFId>ACF</ACFId><APCXYZ><X>0.0</X><Y>0.0</Y><Z>0.0</Z>'
        '</APCXYZ></AntPhaseCenter><AntPattern><Identifier>PAT</Identifier><FreqZero>9.6e9</FreqZero><EB>'
        + poly1('DCXPoly', [0.0, 1e-4]) + poly1('DCYPoly', [0.01]) + '</EB>' + gp('Array') + gp('Element')
        + '</AntPattern></Antenna>')[0])
    return ET.tostring(root).replace(b'ns0:', b'').replace(b'xmlns:ns0=', b'xmlns=')


@pytest.mark.parametrize('fmt', ['CF8', 'CI4', 'CI2'])
def test_two_channels(files, need, monkeypatch, fmt):
    """Two channels, integer samples with AmpSF, added per-vector parameters (an unsigned integer and a string), an
    antenna block and a support array: the file written natively and the one sarpy writes from the same metadata
    read alike through both readers; the native one is consistent by sarkit; a channel by identifier."""
    from sarpy.io.phase_history.cphd import CPHDWriter1
    from sarpy.io.phase_history.cphd1_elements.CPHD import CPHDType
    tmp, col, S, spot, _ = files
    r = _cphdxml.open_cphd(spot)
    P, K = S.shape
    xml = rich_xml(r.cphd_details.get_cphd_bytes(), P, K, fmt)
    dt = _cphdxml.pvp_dtype(ET.fromstring(xml).find(f'{{{NS11}}}PVP'), r.cphd_meta.Data.NumBytesPVP + 40, '=')
    assert {'AmpSF', 'SIGNAL', 'Count', 'Label'} < set(dt.names) and dt['Label'] == np.dtype('S16')
    v0 = r.read_pvp_array(0)
    scale = {'CF8': 1.0, 'CI4': 3000.0, 'CI2': 100.0}[fmt] / np.abs(S).max()
    pvps, sigs = [], []
    for c in range(2):
        v = np.zeros(P, dt)
        for n in v0.dtype.names:
            v[n] = v0[n]
        v['AmpSF'] = (1.0 + 0.001 * np.arange(P)) * (c + 1) / scale
        v['SIGNAL'], v['Count'], v['Label'] = 1, np.arange(P) * (c + 1), [f'vector {i}'.encode() for i in range(P)]
        v['SIGNAL'][3] = 0
        stored = S * scale / (1.0 + 0.001 * np.arange(P)[:, None]) / (c + 1)
        pvps.append(v)
        sigs.append((stored if fmt == 'CF8' else np.rint(stored.real) + 1j * np.rint(stored.imag)).astype(np.complex64))
    gain = np.zeros((5, 7), [('Gain', '>f4'), ('Phase', '>f4')])
    gain['Gain'] = -3.0 * np.add.outer(np.arange(-2, 3) ** 2, np.arange(-3, 4) ** 2)
    gain['Phase'] = 0.25
    native = str(tmp / f'two_{fmt}.cphd')
    _cphdxml.write_cphd(native, xml, pvps, sigs, support=gain.tobytes())
    n = compare(native)
    warn = consistent(native, need)
    rn = _cphdxml.open_cphd(native)
    assert rn.signal_map(1).dtype == np.dtype(_cphdxml._SIGNAL[fmt]) and rn.read_pvp_variable('Label', 'second')[5] == b'vector 5'
    tol = {'CF8': 1e-6, 'CI4': 2.0 / 3000.0, 'CI2': 2.0 / 100.0}[fmt]       # rounding of the stored integers, times AmpSF
    for c in range(2):
        err = float(np.abs(rn.read_chip(index=c) - S).max() / np.abs(S).max())
        assert err <= tol, (c, err)
    np.testing.assert_array_equal(rn.read_support_array('AGP')['Gain'], gain['Gain'])
    (c1, m1), through = read_both(native, monkeypatch, channel='second')
    assert_same_read((c1, m1), through)
    assert m1['channel_index'] == 1 and m1['polarization'] == 'VH' and any('SIGNAL=0' in s for s in m1['notes'])
    assert float(np.abs(c1['S'] - io.read_cphd(spot)['S']).max() / np.abs(S).max()) <= 2 * tol
    # the same metadata and arrays written by sarpy
    if fmt == 'CF8':
        theirs = str(tmp / 'two_sarpy.cphd')
        with CPHDWriter1(theirs, CPHDType.from_xml_string(xml.decode()), check_existence=False) as w:
            w.write_file({i: p for i, p in zip(('VV', 'second'), pvps)},
                         {i: np.asarray(s * p['AmpSF'][:, None], np.complex64) for i, s, p in zip(('VV', 'second'), sigs, pvps)},
                         {'AGP': np.stack([gain['Gain'], gain['Phase']], -1)})
        compare(theirs)
        a, b = _cphdxml.open_cphd(theirs), rn
        for c in range(2):
            pa, pb = a.read_pvp_array(c), b.read_pvp_array(c)
            assert all(np.array_equal(pa[k], pb[k]) for k in pb.dtype.names)
            np.testing.assert_allclose(a.read_chip(index=c), b.read_chip(index=c), rtol=2e-6, atol=1e-6 * np.abs(S).max())
    print(f'two channels, {fmt}: {n} values equal to sarpy; sarkit: no errors' + (f', warnings {warn}' if warn else ''))


def test_radiometry_reads_the_antenna(files):
    """The radiometric model finds the antenna, the gain array and the reference levels through the native reader
    as it does through sarpy's."""
    from sarpy.io.phase_history.converter import open_phase_history
    from fastsar import rgiqe
    tmp = files[0]
    path = str(tmp / 'two_CF8.cphd')
    if not os.path.exists(path):
        pytest.skip('needs the file of test_two_channels[CF8]')
    _, meta = io.read_cphd(path, meta=True)
    a, b = rgiqe.Radiometry(path, meta), rgiqe.Radiometry(open_phase_history(path), meta)
    assert a.has_antenna and b.has_antenna and a.level_source == b.level_source == 'cphd'
    assert any('gain arrays' in s for s in a.notes) and a.notes == b.notes
    X = np.array([[0.0, 0.0, 0.0], [20.0, 10.0, 0.0], [-15.0, 25.0, 0.0]])
    np.testing.assert_array_equal(a.gain_db(X), b.gain_db(X))
    np.testing.assert_array_equal(a.nesz_db(X, 2.0, 1e6), b.nesz_db(X, 2.0, 1e6))


def test_compressed_signal(files):
    """A file declaring a compressed signal block: metadata and per-vector parameters read, the samples are refused."""
    tmp, col, S, spot, _ = files
    r = _cphdxml.open_cphd(spot)
    xml = r.cphd_details.get_cphd_bytes().replace(
        b'</NumCPHDChannels>', b'</NumCPHDChannels><SignalCompressionID>vendor codec</SignalCompressionID>')
    path = str(tmp / 'compressed.cphd')
    _cphdxml.write_cphd(path, xml, [r.read_pvp_array(0)], [S])
    c = _cphdxml.open_cphd(path)
    assert c.compression == 'vendor codec' and np.array_equal(c.read_pvp_variable('TxPos', 0), r.read_pvp_variable('TxPos', 0))
    for call in (lambda: c.signal_map(0), lambda: c.read_chip((0, 2), (0, 2)), lambda: io.read_cphd(path)):
        with pytest.raises(ValueError, match='compressed'):
            call()


def test_not_a_cphd(tmp_path):
    p = tmp_path / 'x.cphd'
    p.write_bytes(b'SICD/1.0\nnothing\n' + b'\0' * 100)
    with pytest.raises(ValueError, match='not a CPHD file'):
        io.read_cphd(str(p))


@pytest.mark.parametrize('path', REAL, ids=[os.path.basename(p) for p in REAL])
def test_real_file(path):
    """A real collection: the header, the metadata FastSAR reads, every per-vector parameter of every channel and a
    few samples, native against sarpy. The phase history is not loaded."""
    if not os.path.exists(path):
        pytest.skip(f'{path} is not on this machine')
    n = compare(path, samples=False)
    r = _cphdxml.open_cphd(path)
    print(f'{os.path.basename(path)} (CPHD {r.cphd_version}, {len(r.read_pvp_array(0).dtype.names)} per-vector parameters): '
          f'{n} metadata values and per-vector parameters equal to sarpy')
