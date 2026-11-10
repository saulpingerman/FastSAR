"""The NITF writer and reader (fastsar._nitf) against sarpy: the same SICD and SIDD written by both and compared
field by field, each one's files read by the other, images cut into several segments, the segment arithmetic at
the standards' limits, sarkit's consistency checks of the files written here, and what the reader refuses.
Needs sarpy; the consistency checks need sarkit (Python 3.11 or later)."""
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import fastsar
from fastsar import _nitf, products, sidd, sim

pytestmark = pytest.mark.needs('sarpy')

TG = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 15.0, 0.0], [-18.0, -22.0, 0.0]])
SIDD_SPEC = ('3.0', '2021-11-30T00:00:00Z', 'urn:SIDD:3.0.0')


def sarpy_sicd(path, xml, a, row_limit=None):
    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.complex.sicd import SICDWriter, SICDWritingDetails
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        details = SICDWritingDetails(SICDType.from_xml_string(xml.decode()), row_limit=row_limit)
        with SICDWriter(str(path), sicd_writing_details=details, check_existence=False) as w:
            w.write_chip(a, start_indices=(0, 0))


def sarpy_sidd(path, xml, a, sicd_xml=None, row_limit=None):
    from sarpy.io.complex.sicd_elements.SICD import SICDType
    from sarpy.io.product.sidd3_elements.SIDD import SIDDType
    from sarpy.io.product.sidd import SIDDWriter, SIDDWritingDetails
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        details = SIDDWritingDetails(SIDDType.from_xml_string(xml.decode()), row_limit=row_limit,
                                     sicd_meta=None if sicd_xml is None else SICDType.from_xml_string(sicd_xml.decode()))
        with SIDDWriter(str(path), sidd_writing_details=details, check_existence=False) as w:
            w.write_chip(a, start_indices=(0, 0))


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    """A small simulated product, and the same SICD and SIDD written by sarpy from its XML: paths, pixels, XML."""
    tmp = tmp_path_factory.mktemp('nitf')
    col = sim.make_collect(res=0.5, scene=100.0, r0=6e3, graze_deg=35.0)
    S = sim.simulate_brute(col, TG, np.ones(len(TG))).astype(np.complex64)
    sim.write_cphd(str(tmp / 'a.cphd'), col, S, 40.7934, -77.86, 351.0, heading=30.0)
    out = fastsar.form_cphd(str(tmp / 'a.cphd'), backend='cpu', spacing=0.4, height=351.0)
    sm, a = products.sicd_meta(out)
    a = np.ascontiguousarray(a, np.complex64)
    # sarpy writes the XML as it serializes the metadata it parsed, and its parsing renormalizes unit vectors, so
    # one pass can move a number by a unit in the last place: the XML is passed through until sarpy returns it
    # unchanged, and the files compared below then carry the same bytes
    sicd_xml = sm.to_xml_bytes()
    for _ in range(8):
        sarpy_sicd(tmp / 'sarpy_sicd.nitf', sicd_xml, a)
        last, sicd_xml = sicd_xml, _nitf.read(tmp / 'sarpy_sicd.nitf').xml('SICD')[0]
        if sicd_xml == last:
            break
    assert sicd_xml == last
    sidd.write_sidd(str(tmp / 'first.nitf'), out)
    first = _nitf.read(tmp / 'first.nitf')
    sidd_xml, d = first.xml('SIDD')[0], np.array(first.pixels())
    for _ in range(8):
        sarpy_sidd(tmp / 'sarpy_sidd.nitf', sidd_xml, d, sicd_xml)
        last, sidd_xml = sidd_xml, _nitf.read(tmp / 'sarpy_sidd.nitf').xml('SIDD')[0]
        if sidd_xml == last:
            break
    assert sidd_xml == last and _nitf.read(tmp / 'sarpy_sidd.nitf').xml('SICD')[0] == sicd_xml
    return SimpleNamespace(tmp=tmp, a=a, sicd_xml=sicd_xml, d=d, sidd_xml=sidd_xml)


def differences(ref, new):
    """Names of the header fields that differ between two parsed files, with the segment they are in."""
    out = []
    for name, x, y in [('file', ref.header, new.header)] + [(f'image {i}', a, b) for i, (a, b) in
                                                             enumerate(zip(ref.images, new.images))] \
            + [(f'des {i}', a, b) for i, (a, b) in enumerate(zip(ref.des, new.des))]:
        out += [(name, k) for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)]
    assert len(ref.images) == len(new.images) and len(ref.des) == len(new.des)
    return out


def arcseconds(igeolo):
    """An IGEOLO field as [4, 2] signed arcseconds of latitude and longitude."""
    out = []
    for k in range(4):
        s = igeolo[15 * k:15 * k + 15]
        lat = (int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])) * (1 if s[6] == 'N' else -1)
        lon = (int(s[7:10]) * 3600 + int(s[10:12]) * 60 + int(s[12:14])) * (1 if s[14] == 'E' else -1)
        out.append((lat, lon))
    return np.array(out)


def test_sicd_against_sarpy(scene, monkeypatch):
    """One SICD written by sarpy and here: every header field, the XML and the pixels, each read by the other."""
    from sarpy.io.complex.converter import open_complex
    ref = _nitf.read(scene.tmp / 'sarpy_sicd.nitf')
    path = scene.tmp / 'native_sicd.nitf'
    # sarpy titles the file with a name it composes from the collection; given that title, the rest must agree
    n = _nitf.write_sicd(path, scene.a, scene.sicd_xml, fields=dict(FTITLE=ref.header['FTITLE']))
    new = _nitf.read(path)
    assert n == path.stat().st_size == new.header['FL'] == ref.header['FL']
    # the corner coordinates: rounded to the arcsecond here, truncated by sarpy
    assert differences(ref, new) in ([], [('image 0', 'IGEOLO')])
    corners = _nitf._xml_facts(scene.sicd_xml)['corners']
    assert new.images[0]['IGEOLO'] == sidd._igeolo(corners)
    assert np.abs(arcseconds(new.images[0]['IGEOLO']) - arcseconds(ref.images[0]['IGEOLO'])).max() <= 1
    assert new.xml() == new.xml('SICD') == [scene.sicd_xml] and new.xml('SIDD') == []
    # the reader on both files: the map in the file's byte order, and the converted array
    for f in (ref, new):
        m = f.pixels()
        assert isinstance(m, np.memmap) and m.dtype == np.dtype('>c8') and np.array_equal(m, scene.a)
        p = f.pixels(native=True)
        assert p.dtype == np.complex64 and p.dtype.isnative and np.array_equal(p, scene.a)
    rd = open_complex(str(path))
    assert np.array_equal(rd[:, :], scene.a)
    assert rd.sicd_meta.to_xml_bytes() == open_complex(str(scene.tmp / 'sarpy_sicd.nitf')).sicd_meta.to_xml_bytes()
    # the default title, other fields, and the same bytes without the memory map and from other array layouts
    _nitf.write_sicd(path, scene.a, scene.sicd_xml, fields=dict(TGTID='T1', ICOM=['a comment'], OSTAID='here', REL='XX'))
    f = _nitf.read(path)
    assert f.header['FTITLE'] == f.images[0]['IID2'] == 'SICD: SIMULATED' and f.header['OSTAID'] == 'here'
    assert f.images[0]['TGTID'] == 'T1' and f.images[0]['ICOM'] == ['a comment']
    assert f.header['FSREL'] == f.images[0]['ISREL'] == f.des[0]['DESREL'] == 'XX'
    assert np.array_equal(open_complex(str(path))[:, :], scene.a)
    _nitf.write_sicd(path, scene.a, scene.sicd_xml)
    want = path.read_bytes()
    monkeypatch.setenv('FASTSAR_NITF_MMAP', '0')
    for a in (scene.a, scene.a.astype(np.complex128), np.asfortranarray(scene.a), scene.a[::-1][::-1]):
        _nitf.write_sicd(path, a, scene.sicd_xml)
        assert path.read_bytes() == want
    monkeypatch.undo()
    _nitf.write_sicd(path, scene.a.astype(np.complex128), scene.sicd_xml)
    assert path.read_bytes() == want
    with pytest.raises(ValueError, match='complex'):
        _nitf.write_sicd(path, np.abs(scene.a), scene.sicd_xml)


def test_sidd_against_sarpy(scene):
    """One SIDD (with its SICD) written by sarpy and here."""
    from sarpy.io.product.converter import open_product
    ref = _nitf.read(scene.tmp / 'sarpy_sidd.nitf')
    path = scene.tmp / 'native_sidd.nitf'
    sicd_spec = tuple(ref.des[1][k] for k in ('DESSHSV', 'DESSHSD', 'DESSHTN'))
    assert sicd_spec == _nitf.SICD_SPEC
    _nitf.write_sidd(path, scene.d, [(scene.sidd_xml, 'SIDD', SIDD_SPEC), (scene.sicd_xml, 'SICD', sicd_spec)])
    new = _nitf.read(path)
    # sarpy 2.1.1 truncates the corners to the arcsecond and writes its own version and date of the SIDD
    # specification in the XML segment's header
    allowed = {('image 0', 'IGEOLO'), ('des 0', 'DESSHSV'), ('des 0', 'DESSHSD'), ('des 0', 'DESSHF')}
    assert set(differences(ref, new)) <= allowed
    assert tuple(new.des[0][k] for k in ('DESSHSV', 'DESSHSD', 'DESSHTN')) == SIDD_SPEC
    assert new.images[0]['IID1'] == 'SIDD001001' and new.images[0]['IREP'] == 'MONO' and new.images[0]['IMODE'] == 'B'
    assert new.xml('SIDD') == [scene.sidd_xml] and new.xml('SICD') == [scene.sicd_xml] and len(new.xml()) == 2
    for f in (ref, new):
        assert f.pixels().dtype == np.uint8 and np.array_equal(f.pixels(), scene.d)
        assert np.array_equal(f.pixels(native=True), scene.d)
    rd, rs = open_product(str(path)), open_product(str(scene.tmp / 'sarpy_sidd.nitf'))
    assert np.array_equal(rd[:, :], scene.d)
    one = lambda m: m[0] if isinstance(m, (list, tuple)) else m
    assert one(rd.sidd_meta).to_xml_bytes() == one(rs.sidd_meta).to_xml_bytes()
    assert one(rd.sicd_meta).to_xml_bytes() == one(rs.sicd_meta).to_xml_bytes()
    # corners from a function of the pixel block, as write_sidd of fastsar.sidd gives them
    box = []
    c = _nitf._xml_facts(scene.sidd_xml)['corners']
    _nitf.write_sidd(path, scene.d, [(scene.sidd_xml, 'SIDD', SIDD_SPEC)], corners=lambda *b: box.append(b) or c + 1.0)
    assert box == [(0, scene.d.shape[0], 0, scene.d.shape[1])]
    assert _nitf.read(path).images[0]['IGEOLO'] == sidd._igeolo(c + 1.0)
    # three bands: RGB24I, interleaved by pixel
    rgb = np.stack([scene.d, scene.d[::-1], scene.d[:, ::-1]], -1)
    _nitf.write_sidd(path, rgb, [(scene.sidd_xml, 'SIDD', SIDD_SPEC)])
    f = _nitf.read(path)
    assert f.images[0]['IREP'] == 'RGB' and f.images[0]['IMODE'] == 'P' and f.images[0]['NBANDS'] == 3
    assert [b['IREPBAND'] for b in f.images[0]['bands']] == ['R', 'G', 'B']
    assert np.array_equal(f.pixels(), rgb) and np.array_equal(f.pixels(native=True), rgb)
    with pytest.raises(ValueError, match='uint8'):
        _nitf.write_sidd(path, scene.d.astype(np.float32), [(scene.sidd_xml, 'SIDD', SIDD_SPEC)])
    with pytest.raises(ValueError, match='SIDD XML'):
        _nitf.write_sidd(path, scene.d, [(scene.sicd_xml, 'SICD', _nitf.SICD_SPEC)])


@pytest.mark.parametrize('limit', ['rows', 'bytes'])
def test_segments(scene, limit):
    """Images cut into segments by a small row or byte limit: the fields against sarpy's with the same limit, the
    pixels through sarpy's reader and this one, and the corners of the segments."""
    from sarpy.io.complex.converter import open_complex
    from sarpy.io.product.converter import open_product
    rows, cols = scene.a.shape
    kw = dict(rows_max=100) if limit == 'rows' else dict(size_max=100 * cols * 8 + 7)
    kd = dict(rows_max=100) if limit == 'rows' else dict(size_max=100 * cols + 7)
    path, spath = scene.tmp / 'seg.nitf', scene.tmp / 'seg_sarpy.nitf'
    nseg = -(-rows // 100)
    assert nseg > 2
    # SICD
    sarpy_sicd(spath, scene.sicd_xml, scene.a, row_limit=100)
    ref = _nitf.read(spath)
    _nitf.write_sicd(path, scene.a, scene.sicd_xml, fields=dict(FTITLE=ref.header['FTITLE']), **kw)
    new = _nitf.read(path)
    assert len(new.images) == nseg and new.groups() == ref.groups() == [list(range(nseg))]
    assert {k for _, k in differences(ref, new)} <= {'IGEOLO'}
    assert [im['IID1'] for im in new.images] == [f'SICD{k + 1:03d}' for k in range(nseg)]
    assert [im['ILOC'] for im in new.images] == ['0000000000'] + ['0010000000'] * (nseg - 1)
    assert [(im['IDLVL'], im['IALVL']) for im in new.images] == [(k + 1, k) for k in range(nseg)]
    assert sum(im['NROWS'] for im in new.images) == rows
    for f in (ref, new):
        assert np.array_equal(f.pixels(), scene.a) and np.array_equal(f.pixels(native=True), scene.a)
    assert np.array_equal(open_complex(str(path))[:, :], scene.a)
    # the segments' corners: the image's own at its ends, shared between neighbors, and at the rows' fractions of
    # the image's sides (the image is a plane a few hundred meters across, so the sides are straight to well under
    # an arcsecond)
    corners = _nitf._xml_facts(scene.sicd_xml)['corners']
    geo = np.array([arcseconds(im['IGEOLO']) for im in new.images])
    full = arcseconds(sidd._igeolo(corners))
    assert np.array_equal(geo[0, :2], full[:2]) and np.array_equal(geo[-1, 2:], full[2:])
    assert np.array_equal(geo[1:, 0], geo[:-1, 3]) and np.array_equal(geo[1:, 1], geo[:-1, 2])
    for k in range(nseg - 1):
        u = (k + 1) * 100 / rows
        assert np.abs(geo[k, 3] - 3600 * ((1 - u) * corners[0] + u * corners[3])).max() <= 0.51
        assert np.abs(geo[k, 2] - 3600 * ((1 - u) * corners[1] + u * corners[2])).max() <= 0.51
    # SIDD
    sarpy_sidd(spath, scene.sidd_xml, scene.d, row_limit=100)
    ref = _nitf.read(spath)
    _nitf.write_sidd(path, scene.d, [(scene.sidd_xml, 'SIDD', SIDD_SPEC)], **kd)
    new = _nitf.read(path)
    assert {k for _, k in differences(ref, new)} <= {'IGEOLO', 'DESSHSV', 'DESSHSD', 'DESSHF'}
    assert [im['IID1'] for im in new.images] == [f'SIDD001{k + 1:03d}' for k in range(nseg)]
    for f in (ref, new):
        assert np.array_equal(f.pixels(), scene.d) and np.array_equal(f.pixels(native=True), scene.d)
    assert np.array_equal(open_product(str(path))[:, :], scene.d)


def test_segment_arithmetic():
    """The row bounds at the standards' limits against sarpy's, for images that are not written."""
    from sarpy.io.general.nitf import default_image_segmentation
    for rows, cols, size in ((250_000, 3000, 8), (40_000, 40_000, 8), (120_000, 100, 1), (99_999, 10, 8),
                             (100_000, 10, 8), (35_356, 35_356, 8), (35_355, 35_355, 8), (150_000, 120_000, 1),
                             (1, 1, 8), (200_001, 50_000, 3)):
        want = default_image_segmentation(rows, cols, min(99_999, (10 ** 10 - 2) // (cols * size)))
        got = _nitf.segments(rows, cols, size)
        assert got == [(r0, r1) for r0, r1, _, _ in want], (rows, cols, size)
        assert all((r1 - r0) * cols * size <= _nitf.IS_SIZE_MAX and r1 - r0 <= 99_999 for r0, r1 in got)
    with pytest.raises(ValueError, match='one row'):
        _nitf.segments(10, 2 * 10 ** 9, 8)
    assert [_nitf._clevel(s, d) for s, d in ((1 << 20, 2048), (1 << 20, 2049), (60 << 20, 100), (1 << 30, 8193),
                                             (3 << 30, 70_000), (11 << 30, 10))] == [3, 5, 5, 6, 7, 9]


def test_reader(scene):
    """Other pixel types, and what the reader refuses."""
    path = scene.tmp / 'other.nitf'
    rng = np.random.default_rng(0)
    iq = rng.integers(-3000, 3000, (37, 23, 2)).astype(np.int16)
    sub = dict(IID1='SICD000', IREP='NODISPLY', IMODE='P', IDLVL=1, IALVL=0, ILOC='0000000000')
    des = [dict(xml=scene.sicd_xml, kind='SICD', spec=_nitf.SICD_SPEC)]
    _nitf.write(path, [dict(sub, data=iq, dtype='>i2', bands=[('', 'I'), ('', 'Q')], PVTYPE='SI', ABPP=16, NBPP=16)], des)
    f = _nitf.read(path)
    assert f.pixels().dtype == np.dtype('>i2') and np.array_equal(f.pixels(), iq)
    p = f.pixels(native=True)
    assert p.dtype == np.complex64 and np.array_equal(p, iq[..., 0] + 1j * iq[..., 1])
    assert f.images[0]['ICORDS'] == '' and f.images[0]['IGEOLO'] == ''
    ap = rng.integers(0, 256, (37, 23, 2)).astype(np.uint8)
    _nitf.write(path, [dict(sub, data=ap, dtype='u1', bands=[('', 'M'), ('', 'P')], PVTYPE='INT', ABPP=8, NBPP=8)], des)
    f = _nitf.read(path)
    assert np.array_equal(f.pixels(), ap)
    # magnitude and phase bytes as complex: the amplitude table's entry (or the byte itself) at 256ths of a turn
    turn = np.exp(2j * np.pi * ap[..., 1] / 256)
    table = np.linspace(0.0, 5.0, 256) ** 2
    assert np.allclose(f.pixels(native=True), ap[..., 0] * turn, rtol=1e-6, atol=1e-4)
    assert np.allclose(f.pixels(native=True, amp_table=table), table[ap[..., 0]] * turn, rtol=1e-6, atol=1e-4)
    with pytest.raises(ValueError, match='256 entries'):
        f.pixels(native=True, amp_table=table[:100])
    # a compressed image, more than one block, bands by block
    raw = path.read_bytes()
    at = f.images[0]['offset'] - f.header['LISH001']
    for old, new, word in ((b'NC2', b'NM2', 'compression'), (b'0P00010001', b'0P00020001', 'block'),
                           (b'0P0001', b'0S0001', 'band layout')):
        k = raw.index(old, at)
        assert k < f.images[0]['offset']
        path.write_bytes(raw[:k] + new + raw[k + len(new):])
        with pytest.raises(ValueError, match=word):
            _nitf.read(path).pixels()
    path.write_bytes(b'NITF02.00' + raw[9:])
    with pytest.raises(ValueError, match='NITF 2.1'):
        _nitf.read(path)
    path.write_bytes(raw[:200])
    with pytest.raises(ValueError):
        _nitf.read(path)


def test_sarkit_consistency(scene, need):
    if sys.version_info < (3, 11):
        pytest.skip('sarkit needs Python 3.11 or later')
    skv = need('sarkit.verification')
    path = scene.tmp / 'sarkit.nitf'
    cases = (('SICD', skv.SicdConsistency, lambda: _nitf.write_sicd(path, scene.a, scene.sicd_xml)),
             ('SIDD', skv.SiddConsistency, lambda: _nitf.write_sidd(path, scene.d, [(scene.sidd_xml, 'SIDD', SIDD_SPEC)])),
             ('SIDD with its SICD', skv.SiddConsistency, lambda: _nitf.write_sidd(
                 path, scene.d, [(scene.sidd_xml, 'SIDD', SIDD_SPEC), (scene.sicd_xml, 'SICD', _nitf.SICD_SPEC)])))
    for name, kind, make in cases:
        make()
        with open(path, 'rb') as fh:
            con = kind.from_file(fh)
        con.check()
        fail = con.failures(omit_passed_sub=True)
        err = sorted(k for k, v in fail.items() if any(d['severity'] == 'Error' for d in v['details']))
        warn = sorted(set(fail) - set(err))
        print(f'sarkit consistency of the native {name}: {len(con.all())} checks, {len(err)} errors'
              + (f', warnings {warn}' if warn else ''))
        assert not err, (name, err)
