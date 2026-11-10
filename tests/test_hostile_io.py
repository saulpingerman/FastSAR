"""Damaged, inconsistent and hostile CPHD and SICD files against FastSAR's own readers (fastsar._cphdxml, _nitf,
_sicdxml): every file that cannot be read raises a ValueError that starts with the file's path and says what is wrong
(never a KeyError, TypeError, IndexError or an XML parser's error); offsets and sizes in headers are checked against
the file before anything is read; an XML document that declares a document type or entities is refused; and the
unusual but valid forms read (a root element after a long comment, UTF-16, magnitude and phase pixels with an
amplitude table). The files are a small simulated collection and its SICD with bytes changed in place."""
import os
import re

import numpy as np
import pytest

import fastsar
from fastsar import _cphdxml, _nitf, _sicdxml, _xmlmeta, io, products, sim


@pytest.fixture(scope='module')
def base(tmp_path_factory):
    """(directory, CPHD bytes, SICD bytes, the SICD's pixels) of a small collection."""
    d = tmp_path_factory.mktemp('hostile')
    tg = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    col = sim.make_collect(res=0.5, scene=60.0, r0=10e3, graze_deg=35.0)
    sim.write_cphd(str(d / 'base.cphd'), col, sim.simulate(col, tg, np.ones(len(tg))), 40.79, -77.86, 351.0, heading=30.0,
                   scene=50.0)
    fastsar.process_cphd(str(d / 'base.cphd'), str(d / 'base.nitf'), backend='cpu')
    pix = np.array(io.read_sicd(str(d / 'base.nitf'))[0])
    return d, (d / 'base.cphd').read_bytes(), (d / 'base.nitf').read_bytes(), pix


def put(d, name, data):
    p = d / name
    p.write_bytes(data)
    return str(p)


def swap(data, old, new, count=1):
    """data with `count` occurrences of old replaced by new of the same length (offsets stay where they are)."""
    assert len(old) == len(new) and old in data, old
    return data.replace(old, new, count)


def refused(path, fn, *words):
    """fn(path) raises a ValueError that starts with the path and mentions each of words."""
    with pytest.raises(ValueError) as e:
        fn(path)
    msg = str(e.value)
    assert msg.startswith(path), msg
    for w in words:
        assert w in msg, (w, msg)
    return msg


def header_value(data, key):
    return re.search(rb'%s := (\S+)\n' % key, data).group(1)


# ------------------------------------------------------------------------------------------------------- CPHD

def test_cphd_header(base):
    d, C, _, _ = base
    read = io.read_cphd
    for key in (b'XML_BLOCK_SIZE', b'XML_BLOCK_BYTE_OFFSET', b'PVP_BLOCK_SIZE', b'PVP_BLOCK_BYTE_OFFSET',
                b'SIGNAL_BLOCK_SIZE', b'SIGNAL_BLOCK_BYTE_OFFSET'):
        v = header_value(C, key)
        line = key + b' := ' + v
        refused(put(d, 'nokey.cphd', swap(C, line, key[:-1] + b'_ := ' + v)), read, 'header has no', key.decode())
        refused(put(d, 'text.cphd', swap(C, line, key + b' := ' + b'x' * len(v))), read, key.decode(), 'whole number')
        refused(put(d, 'neg.cphd', swap(C, line, key + b' := -' + b'1' * (len(v) - 1))), read, key.decode())
        refused(put(d, 'big.cphd', swap(C, line, key + b' := ' + b'9' * len(v))), read)
        if key.endswith(b'OFFSET'):                       # eight bytes either way: a block inside its neighbor,
            for delta in (8, -8):                         # or the XML not where the header says
                moved = b'%0*d' % (len(v), int(v) + delta)
                refused(put(d, 'moved.cphd', swap(C, line, key + b' := ' + moved)), read)
    for n in (0, 3, 12, 200, len(C) // 2, len(C) - 1):
        refused(put(d, 'trunc.cphd', C[:n]), read)
    refused(put(d, 'text.cphd', b'some text\n' * 50), read, 'not a CPHD file')
    refused(put(d, 'noterm.cphd', C.replace(b'\f\n', b'  ', 1)), read)
    refused(put(d, 'version.cphd', swap(C, b'CPHD/1.0.1', b'CPHD/9.0.1')), read, 'version')
    refused(put(d, 'v03.cphd', swap(C, b'CPHD/1.0.1', b'CPHD/0.3  ')), read, 'header has no')
    with pytest.raises(OSError):
        read(str(d / 'missing.cphd'))
    with pytest.raises(OSError):
        read(str(d))


def test_cphd_xml_and_layout(base):
    d, C, _, _ = base
    read = io.read_cphd
    off = int(header_value(C, b'XML_BLOCK_BYTE_OFFSET'))
    refused(put(d, 'garbage.cphd', C[:off] + b'x' + C[off + 1:]), read, 'XML')
    refused(put(d, 'unclosed.cphd', swap(C, b'</CPHD>', b'       ')), read, 'not well-formed')
    refused(put(d, 'doctype.cphd', swap(C, b'SIMULATED', b'<!DOCTYPE')), read, 'document type or entity')
    refused(put(d, 'entity.cphd', swap(C, b'SIMULATED', b'<!ENTITY ')), read, 'document type or entity')
    nv = re.search(rb'<NumVectors>(\d+)</NumVectors>', C).group(1)
    refused(put(d, 'neg.cphd', swap(C, b'<NumVectors>' + nv, b'<NumVectors>-' + nv[1:])), read, 'NumVectors')
    refused(put(d, 'text.cphd', swap(C, b'<NumVectors>' + nv, b'<NumVectors>' + b'x' * len(nv))), read, 'NumVectors')
    refused(put(d, 'more.cphd', swap(C, b'<NumVectors>' + nv, b'<NumVectors>' + b'9' * len(nv))), read, 'run past')
    refused(put(d, 'fewer.cphd', swap(C, b'<NumVectors>' + nv, b'<NumVectors>' + b'%0*d' % (len(nv), int(nv) - 1))),
            read, 'block in the header')
    ns = re.search(rb'<NumSamples>(\d+)</NumSamples>', C).group(1)
    refused(put(d, 'samples.cphd', swap(C, b'<NumSamples>' + ns, b'<NumSamples>' + b'9' * len(ns))), read, 'run past')
    nb = re.search(rb'<NumBytesPVP>(\d+)</NumBytesPVP>', C).group(1)
    for other in (int(nb) + 8, int(nb) - 8, int(nb) + 1):
        refused(put(d, 'nbytes.cphd', swap(C, b'<NumBytesPVP>' + nb, b'<NumBytesPVP>%0*d' % (len(nb), other))), read)
    refused(put(d, 'format.cphd', swap(C, b'<Format>F8</Format>', b'<Format>Q8</Format>')), read, "'Q8'")
    refused(put(d, 'format.cphd', swap(C, b'X=F8;Y=F8;Z=F8;', b'X=F8;Y;;;;Z=F8;')), read, 'NAME=TYPE')
    refused(put(d, 'format.cphd', swap(C, b'X=F8;Y=F8;Z=F8;', b'S9999999999999;')), read, 'string of that length')
    refused(put(d, 'offset.cphd', swap(C, b'<Offset>0</Offset>', b'<Offset>9</Offset>')), read, 'overlap')
    refused(put(d, 'sformat.cphd', swap(C, b'<SignalArrayFormat>CF8', b'<SignalArrayFormat>CF9')), read, 'SignalArrayFormat')
    refused(put(d, 'toa.cphd', swap(C, b'<DomainType>FX', b'<DomainType>TX')), read, 'frequency-domain')
    refused(put(d, 'sigoff.cphd', swap(C, b'<SignalArrayByteOffset>0', b'<SignalArrayByteOffset>8')), read, 'run past')
    for bad in (2, 'nope'):
        refused(str(d / 'base.cphd'), lambda p: read(p, channel=bad), 'channel')
    # the reader itself, and the form_cphd path
    refused(put(d, 'neg.cphd', swap(C, b'<NumVectors>' + nv, b'<NumVectors>-' + nv[1:])), _cphdxml.open_cphd)
    refused(put(d, 'neg.cphd', swap(C, b'<NumVectors>' + nv, b'<NumVectors>-' + nv[1:])),
            lambda p: fastsar.form_cphd(p, backend='cpu'))
    assert io.read_cphd(str(d / 'base.cphd'))['S'].shape == (int(nv), int(ns))


def test_cphd03(base):
    """The 0.3 layout: header keys, block order, the sample type and the vector parameters' sizes."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from test_cphd03 import write_cphd03
    finally:
        sys.path.pop(0)
    d, _, _, _ = base
    p = str(d / 'v03.cphd')
    write_cphd03(p, str(d / 'base.cphd'))
    C = (d / 'v03.cphd').read_bytes()
    assert io.read_cphd(p)['S'].shape == io.read_cphd(str(d / 'base.cphd'))['S'].shape
    read = io.read_cphd
    for key in (b'XML_DATA_SIZE', b'XML_BYTE_OFFSET', b'VB_DATA_SIZE', b'VB_BYTE_OFFSET', b'CPHD_DATA_SIZE', b'CPHD_BYTE_OFFSET'):
        v = header_value(C, key)
        refused(put(d, 'k03.cphd', swap(C, key + b' := ' + v, key[:-1] + b'_ := ' + v)), read, 'header has no')
        refused(put(d, 'b03.cphd', swap(C, key + b' := ' + v, key + b' := ' + b'9' * len(v))), read)
    co = header_value(C, b'CPHD_BYTE_OFFSET')
    refused(put(d, 'o03.cphd', swap(C, b'CPHD_BYTE_OFFSET := ' + co, b'CPHD_BYTE_OFFSET := %0*d' % (len(co), int(co) - 8))),
            read, 'inside the PVP block')
    refused(put(d, 's03.cphd', swap(C, b'<SampleType>RE32F_IM32F', b'<SampleType>RE99F_IM99F')), read, 'SampleType')
    refused(put(d, 'p03.cphd', swap(C, b'<TxPos>24</TxPos>', b'<TxPos>25</TxPos>')), read, 'TxPos')
    refused(put(d, 'p03.cphd', swap(C, b'<TxPos>24</TxPos>', b'<TxPos>xx</TxPos>')), read, 'TxPos')
    refused(put(d, 't03.cphd', C[:int(co) + 1000]), read, 'not within the file')


# ------------------------------------------------------------------------------------------------------- NITF, SICD

def test_nitf_structure(base):
    d, _, B, pix = base
    read = io.read_sicd
    for n in (0, 5, 9, 200, 359, 360, 400, 417, 929, len(B) // 2, len(B) - 1):
        refused(put(d, 'trunc.nitf', B[:n]), read)
    refused(put(d, 'text.nitf', b'not a NITF file at all\n' * 40), read, 'not a NITF 2.1 file')
    refused(put(d, 'v20.nitf', swap(B, b'NITF02.10', b'NITF02.00')), read, 'NITF 2.0')
    hl, lish, li, ldsh, ld = B[354:360], B[363:369], B[369:379], B[391:395], B[395:404]
    assert B[360:363] == b'001' and B[388:391] == b'001' and int(li) == pix.size * 8
    edit = lambda a, b, new: B[:a] + new + B[b:]
    refused(put(d, 'hl.nitf', edit(354, 360, b'999999')), read)
    refused(put(d, 'hl.nitf', edit(354, 360, b'000010')), read, 'HL')
    refused(put(d, 'hl.nitf', edit(354, 360, b'abcdef')), read, 'HL')
    refused(put(d, 'numi.nitf', edit(360, 363, b'999')), read)
    refused(put(d, 'numi.nitf', edit(360, 363, b'00x')), read, 'NUMI')
    refused(put(d, 'lish.nitf', edit(363, 369, b'999999')), read, 'past the end of the file')
    refused(put(d, 'lish.nitf', edit(363, 369, b'000100')), read)
    refused(put(d, 'li.nitf', edit(369, 379, b'9999999999')), read, 'past the end of the file')
    refused(put(d, 'ld.nitf', edit(395, 404, b'999999999')), read, 'past the end of the file')
    refused(put(d, 'ld.nitf', edit(395, 404, b'%09d' % (int(ld) - 40))), read, 'not well-formed')
    refused(put(d, 'ld.nitf', edit(395, 404, b'000000000')), read)
    # the image subheader against the pixels and the XML
    s0 = int(hl)
    sub = B[s0:s0 + int(lish)]
    nrows = sub.index(b'%08d%08d' % pix.shape)
    for rows in (pix.shape[0] + 1, 0, 99999999):
        bad = B[:s0 + nrows] + b'%08d' % rows + B[s0 + nrows + 8:]
        refused(put(d, 'rows.nitf', bad), read)
    refused(put(d, 'rows.nitf', B[:s0 + nrows] + b'abcdefgh' + B[s0 + nrows + 8:]), read, 'NROWS')
    ic = sub.index(b'NC', 400)
    refused(put(d, 'ic.nitf', B[:s0 + ic] + b'C8' + B[s0 + ic + 2:]), read)
    r, c = (b'<NumRows>%d</NumRows>' % pix.shape[0], b'<NumCols>%d</NumCols>' % pix.shape[1])
    msg = refused(put(d, 'xmlrows.nitf', swap(B, r, b'<NumRows>%d</NumRows>' % (pix.shape[0] + 1), 2)), read, 'rows')
    assert str(pix.shape[0] + 1) in msg and str(pix.shape[0]) in msg
    refused(put(d, 'xmlcols.nitf', swap(B, c, b'<NumCols>%d</NumCols>' % (pix.shape[1] - 1), 2)), read, 'columns')
    # the same through the functions that take a vendor SICD
    p = put(d, 'xmlrows.nitf', swap(B, r, b'<NumRows>%d</NumRows>' % (pix.shape[0] + 1), 2))
    with pytest.raises(ValueError, match='rows'):
        io.read_cphd(str(d / 'base.cphd'), sicd=p)
    with pytest.raises(ValueError, match='rows'):
        fastsar.form_cphd(str(d / 'base.cphd'), sicd=p, backend='cpu')
    with pytest.raises(OSError):
        read(str(d / 'missing.nitf'))


def test_sicd_xml(base):
    d, _, B, pix = base
    read = io.read_sicd
    x0 = B.index(b'<SICD')
    refused(put(d, 'garbage.nitf', B[:x0] + b'#' + B[x0 + 1:]), read, 'SICD XML')
    refused(put(d, 'unclosed.nitf', swap(B, b'</SICD>', b'       ')), read, 'not well-formed')
    refused(put(d, 'notsicd.nitf', swap(swap(B, b'<SICD', b'<SIDX'), b'</SICD>', b'</SIDX>')), read, 'no SICD XML')
    for decl in (b'<!DOCTYPE', b'<!ENTITY ', b'<!doctype'):
        refused(put(d, 'decl.nitf', B[:x0] + swap(B[x0:], b'SIMULATED', decl)), read,
                    'document type or entity')
    for bad in (b'<!DOCTYPE a [<!ENTITY e "x">]><SICD/>', b'<?xml version="1.0"?>\n<!DOCTYPE x SYSTEM "http://127.0.0.1/x"><a/>'):
        with pytest.raises(ValueError, match='document type or entity'):
            _sicdxml.read(bad)
        with pytest.raises(ValueError, match='document type or entity'):
            _xmlmeta.parse(bad)
    with pytest.raises(ValueError, match='not well-formed'):
        _sicdxml.read(b'<SICD><a></SICD>')
    # missing blocks and fields: named, for the read and for each use
    cphd = str(d / 'base.cphd')
    for tag, name in ((b'Grid', 'Grid'), (b'GeoData', 'GeoData'), (b'SCPPixel', 'SCPPixel'), (b'FirstRow', 'FirstRow'),
                      (b'FirstCol', 'FirstCol'), (b'UVectECF', 'UVectECF')):
        other = tag[:-1] + b'_'
        gone = swap(swap(B, b'<' + tag + b'>', b'<' + other + b'>', B.count(b'<' + tag + b'>')),
                    b'</' + tag + b'>', b'</' + other + b'>', B.count(b'</' + tag + b'>'))
        p = put(d, 'gone.nitf', gone)
        for fn in (read, _sicdxml.read, lambda q: io.read_cphd(cphd, sicd=q),
                   lambda q: products.write_sicd(str(d / 'out.nitf'), pix, q, transpose=False)):
            with pytest.raises(ValueError) as e:
                fn(p)
            assert p in str(e.value) and name in str(e.value), str(e.value)
    for tag, name in ((b'Position', 'Position.ARPPoly'), (b'TimeCOAPoly', 'TimeCOAPoly')):
        other = tag[:-1] + b'_'
        p = put(d, 'gone.nitf', B.replace(b'<' + tag, b'<' + other).replace(b'</' + tag, b'</' + other))
        a, sm = read(p)                                    # readable: the grid is there
        assert np.array_equal(a, pix)
        refused(p, lambda q: sm.project_image_to_ground(np.zeros((1, 2))), name)
        refused(p, lambda q: io.sicd_points(q, np.zeros(1), np.zeros(1), dict(origin=np.zeros(3), R=np.eye(3))), name)
    p = put(d, 'gone.nitf', B.replace(b'<ImageFormation>', b'<ImageFormatiom>').replace(b'</ImageFormation>', b'</ImageFormatiom>'))
    refused(p, lambda q: products.write_sicd(str(d / 'out.nitf'), pix, q, transpose=False), 'ImageFormation')
    ss = re.search(rb'<SS>([^<]+)</SS>', B).group(1)
    for value, word in ((b'0.0', 'SS'), (b'nan', 'SS'), (b'abc', 'not a number')):
        refused(put(d, 'ss.nitf', swap(B, b'<SS>' + ss, b'<SS>' + value.ljust(len(ss)), 2)), read, word)
    refused(put(d, 'type.nitf', swap(B, b'<Type>PLANE</Type>', b'<Type>PLAIN</Type>')),
            lambda q: read(q)[1].project_image_to_ground(np.zeros((1, 2))), 'PLAIN')


def test_valid_oddities(base):
    """A root element far into the segment, UTF-16, prefixes, a position polynomial as an array, and magnitude and
    phase pixels."""
    d, _, B, pix = base
    sm = _sicdxml.read(str(d / 'base.nitf'))
    xml = sm.to_xml_bytes()
    a = np.ascontiguousarray(pix).astype(np.complex64)
    for name, doc in (('comment', b'<?xml version="1.0"?>\n<!-- ' + b'padding ' * 2000 + b'-->\n' + xml),
                      ('padded', b' ' * 5000 + b'\n' + xml + b'\0\0'),
                      ('utf16', ('<?xml version="1.0" encoding="UTF-16"?>' + xml.decode()).encode('utf-16')),
                      ('prefix', xml.replace(b'<', b'<s:').replace(b'<s:/', b'</s:').replace(b'<s:SICD xmlns=', b'<s:SICD xmlns:s=', 1))):
        p = str(d / f'{name}.nitf')
        _nitf.write_sicd(p, a, doc, corners=sm.GeoData.ImageCorners.get_array())
        got, back = io.read_sicd(p)
        assert np.array_equal(got, pix) and back.Grid.Row.SS == sm.Grid.Row.SS and back.source == p, name
        assert _nitf.read(p).xml('SICD') and not _nitf.read(p).xml('SIDD')
    assert _xmlmeta.root_name(b'<?xml version="1.0"?><!-- <SIDD> --><!DOCTYPE x><n:SICD xmlns:n="u"/>') == 'SICD'
    # the position polynomial: its three coefficient rows
    arp = sm.Position.ARPPoly.get_array()
    assert arp.shape[0] == 3 and np.array_equal(arp[1, :len(sm.Position.ARPPoly.Y.get_array())], sm.Position.ARPPoly.Y.get_array())
    with pytest.raises(ValueError, match='not all numbers'):
        _xmlmeta.parse(b'<a><X>1</X><Y>two</Y><Z>3</Z></a>').get_array()
    with pytest.raises(TypeError, match='neither a vector nor a polynomial'):
        sm.Grid.get_array()
    # AMP8I_PHS8I: the amplitude table's entry at 256ths of a turn, and the magnitude byte itself without a table
    rng = np.random.default_rng(0)
    mp = rng.integers(0, 256, pix.shape + (2,), dtype=np.uint8)
    table = np.linspace(0.0, 5.0, 256) ** 2
    rows = ''.join(f'<Amplitude index="{i}">{v!r}</Amplitude>' for i, v in enumerate(table.tolist()))
    plain = xml.replace(b'<PixelType>RE32F_IM32F</PixelType>', b'<PixelType>AMP8I_PHS8I</PixelType>')
    tabled = plain.replace(b'</PixelType>', b'</PixelType><AmpTable size="256">' + rows.encode() + b'</AmpTable>')
    sub = dict(IID1='SICD000', IREP='NODISPLY', IMODE='P', IDLVL=1, IALVL=0, ILOC='0000000000', data=mp, dtype='u1',
               bands=[('', 'M'), ('', 'P')], PVTYPE='INT', ABPP=8, NBPP=8)
    turn = np.exp(2j * np.pi * mp[..., 1] / 256)
    for doc, amp in ((tabled, table[mp[..., 0]]), (plain, mp[..., 0].astype(np.float64))):
        p = str(d / 'amp8.nitf')
        _nitf.write(p, [sub], [dict(xml=doc, kind='SICD', spec=_nitf.SICD_SPEC)])
        raw, back = io.read_sicd(p)
        assert back.ImageData.PixelType == 'AMP8I_PHS8I' and np.array_equal(raw, mp)
        got = io.read_sicd(p, native=True)[0]
        assert got.dtype == np.complex64 and np.allclose(got, amp * turn, rtol=1e-6, atol=1e-5)
        assert io.read_cphd(str(d / 'base.cphd'), sicd=p)['nx'] * io.read_cphd(str(d / 'base.cphd'), sicd=p)['ny'] == pix.size
