"""A minimal NITF 2.1 writer and reader, sufficient for SICD and SIDD files.

    write_sicd(path, image, xml)                 # complex image [rows, cols] and its SICD XML
    write_sidd(path, pixels, [(xml, 'SIDD', spec)], corners=...)
    f = read(path); f.xml('SICD'); f.pixels()    # the XML of each data extension, the pixels as a memory map

What is implemented, and from where:

- MIL-STD-2500C (NITF 2.1): the file header (table A-1), the image subheader (table A-3) for uncompressed
  images (IC = NC) of one block, the data extension subheader (table A-8) and the XML_DATA_CONTENT user-defined
  subheader of 773 bytes (and its shorter 5 and 283 byte forms on reading), and the complexity level (table A-10)
  from the file's size and the largest image dimension. The reader steps over graphic, text and reserved
  extension segments and over tagged record extensions in the headers; the writer writes none.
- SICD Volume 2 (file format): one image of two float32 bands (I, Q) interleaved by pixel, big-endian, cut along
  rows into segments SICD001, SICD002, ... (SICD000 when there is one) of at most 99,999 rows and 9,999,999,998
  bytes, each attached to the one before (IALVL, ILOC), with the corner coordinates of each segment interpolated
  from the image's corners in ECF; and the DES carrying the SICD XML.
- SIDD Volume 2 (file format): the same segmentation for an 8-bit product image (MONO8I: one band, IREP MONO;
  RGB24I: three bands interleaved by pixel), segments SIDD001001, SIDD001002, ..., followed by one DES per SIDD
  XML and then one per SICD XML.

Pixels are converted to the file's byte order in row blocks on all cores (patches._rows), straight into a memory
map of the preallocated file (or, where the file system gives no map, into scratch blocks written with os.pwrite
at their offsets; FASTSAR_NITF_MMAP=0 forces this), so the write needs no copy of the image.

Text fields are padded with spaces on the right, numbers with zeros on the left. Latitudes and longitudes in
degrees, positive north and east; IGEOLO is rounded to the arcsecond."""
import os
import re
from datetime import datetime, timezone

import numpy as np

IS_SIZE_MAX = 9_999_999_998      # bytes of pixel data in one image segment (the LI field has 10 digits)
ILOC_ROWS_MAX = 99_999           # rows in one segment of a multi-segment image (ILOC gives the row offset in 5 digits)
BLOCK_BYTES = 8 << 20            # pixel bytes converted and written per task

SICD_SPEC = ('1.3.0', '2021-11-30T00:00:00Z', 'urn:SICD:1.3.0')
_SOURCE = {'SICD': 'SICD Volume 1 Design & Implementation Description Document',
           'SIDD': 'SIDD Volume 1 Design & Implementation Description Document'}

# (name, length) of the security group, shared by the file header (FS), image subheader (IS) and DES subheader (DES)
_SEC = (('CLAS', 1), ('CLSY', 2), ('CODE', 11), ('CTLH', 2), ('REL', 20), ('DCTP', 2), ('DCDT', 8), ('DCXM', 4),
        ('DG', 1), ('DGDT', 8), ('CLTX', 43), ('CATP', 1), ('CAUT', 40), ('CRSN', 1), ('SRDT', 8), ('CTLN', 15))
_XML_DES = (('DESCRC', 5), ('DESSHFT', 8), ('DESSHDT', 20), ('DESSHRP', 40), ('DESSHSI', 60), ('DESSHSV', 10),
            ('DESSHSD', 20), ('DESSHTN', 120), ('DESSHLPG', 125), ('DESSHLPT', 25), ('DESSHLI', 20),
            ('DESSHLIN', 120), ('DESSHABS', 200))


def _sec_names(prefix):
    return [((('DECLAS' if prefix == 'DES' else prefix + 'CLAS') if n == 'CLAS' else prefix + n), k) for n, k in _SEC]


# ------------------------------------------------------------------------------------------------------- reading

class _Cursor:
    def __init__(self, b, out):
        self.b, self.i, self.out = b, 0, out

    def raw(self, n):
        if n < 0:
            raise ValueError(f'a NITF header length field gives {n} bytes')
        v = self.b[self.i:self.i + n]
        if len(v) != n:
            raise ValueError(f'the NITF header ends {n - len(v)} bytes short of what its length fields say '
                             '(truncated file, or wrong lengths)')
        self.i += n
        return v

    def s(self, name, n):
        v = self.out[name] = self.raw(n).decode('latin-1').rstrip()
        return v

    def n(self, name, n):
        t = self.raw(n).decode('latin-1').strip()
        if t and not t.isdigit():
            raise ValueError(f'NITF header field {name} is not a number: {t!r}')
        v = self.out[name] = int(t) if t else 0
        return v

    def many(self, fields):
        for name, n in fields:
            self.s(name, n)


def _file_header(fh):
    head = fh.read(360)
    if head[:9] != b'NITF02.10':
        other = head[:4] in (b'NITF', b'NSIF')
        raise ValueError('not a NITF 2.1 file' + (' (NITF 2.0 and NSIF are not read)' if other else ''))
    if len(head) < 360 or not head[354:360].isdigit() or int(head[354:360]) < 388:
        raise ValueError(f'the NITF file header is cut short or its length field HL is wrong ({head[354:360]!r})')
    hl = int(head[354:360])
    fh.seek(0)
    h = {}
    c = _Cursor(fh.read(hl), h)
    c.many((('FHDR', 4), ('FVER', 5)))
    c.n('CLEVEL', 2)
    c.many((('STYPE', 4), ('OSTAID', 10), ('FDT', 14), ('FTITLE', 80)))
    c.many(_sec_names('FS'))
    c.many((('FSCOP', 5), ('FSCPYS', 5), ('ENCRYP', 1)))
    h['FBKGC'] = bytes(c.raw(3))
    c.many((('ONAME', 24), ('OPHONE', 18)))
    c.n('FL', 12)
    c.n('HL', 6)
    sizes = {}
    for key, num, a, na, b, nb in (('image', 'NUMI', 'LISH', 6, 'LI', 10), ('graphic', 'NUMS', 'LSSH', 4, 'LS', 6),
                                   ('label', 'NUMX', '', 0, '', 0), ('text', 'NUMT', 'LTSH', 4, 'LT', 5),
                                   ('des', 'NUMDES', 'LDSH', 4, 'LD', 9), ('res', 'NUMRES', 'LRESH', 4, 'LRE', 7)):
        sizes[key] = [(c.n(f'{a}{k + 1:03d}', na), c.n(f'{b}{k + 1:03d}', nb)) for k in range(c.n(num, 3)) if na]
    for ln, ofl, data in (('UDHDL', 'UDHOFL', 'UDHD'), ('XHDL', 'XHDLOFL', 'XHD')):
        n = c.n(ln, 5)
        if n:
            c.n(ofl, 3)
            h[data] = bytes(c.raw(n - 3))
    return h, sizes


def _image_subheader(b):
    d = {}
    c = _Cursor(b, d)
    c.many((('IM', 2), ('IID1', 10), ('IDATIM', 14), ('TGTID', 17), ('IID2', 80)))
    if d['IM'] != 'IM':
        raise ValueError('NITF image subheader not found where the file header places it')
    c.many(_sec_names('IS'))
    c.many((('ENCRYP', 1), ('ISORCE', 42)))
    c.n('NROWS', 8)
    c.n('NCOLS', 8)
    c.many((('PVTYPE', 3), ('IREP', 8), ('ICAT', 8)))
    c.n('ABPP', 2)
    c.s('PJUST', 1)
    if c.raw(1) != b' ':
        d['ICORDS'] = chr(b[c.i - 1])
        c.s('IGEOLO', 60)
    else:
        d['ICORDS'], d['IGEOLO'] = '', ''
    d['ICOM'] = [c.raw(80).decode('latin-1').rstrip() for _ in range(c.n('NICOM', 1))]
    if c.s('IC', 2) not in ('NC', 'NM'):
        c.s('COMRAT', 4)
    nb = c.n('NBANDS', 1) or c.n('XBANDS', 5)
    d['NBANDS'] = nb
    bands = d['bands'] = []
    for _ in range(nb):
        bd = {}
        cb = _Cursor(b, bd)
        cb.i = c.i
        cb.many((('IREPBAND', 2), ('ISUBCAT', 6), ('IFC', 1), ('IMFLT', 3)))
        if cb.n('NLUTS', 1):
            bd['LUT'] = bytes(cb.raw(bd['NLUTS'] * cb.n('NELUT', 5)))
        c.i = cb.i
        bands.append(bd)
    c.many((('ISYNC', 1), ('IMODE', 1)))
    for name, n in (('NBPR', 4), ('NBPC', 4), ('NPPBH', 4), ('NPPBV', 4), ('NBPP', 2), ('IDLVL', 3), ('IALVL', 3)):
        c.n(name, n)
    c.many((('ILOC', 10), ('IMAG', 4)))
    for ln, ofl, data in (('UDIDL', 'UDOFL', 'UDID'), ('IXSHDL', 'IXSOFL', 'IXSHD')):
        n = c.n(ln, 5)
        if n:
            c.n(ofl, 3)
            d[data] = bytes(c.raw(n - 3))
    return d


def _des_subheader(b):
    d = {}
    c = _Cursor(b, d)
    c.many((('DE', 2), ('DESID', 25), ('DESVER', 2)))
    if d['DE'] != 'DE':
        raise ValueError('NITF data extension subheader not found where the file header places it')
    c.many(_sec_names('DES'))
    if d['DESID'] == 'TRE_OVERFLOW':
        c.many((('DESOFLW', 6), ('DESITEM', 3)))
    n = c.n('DESSHL', 4)
    d['DESSHF'] = bytes(c.raw(n))
    if d['DESID'] == 'XML_DATA_CONTENT' and n in (5, 283, 773):
        u = _Cursor(d['DESSHF'], d)
        u.many(_XML_DES[:1] if n == 5 else _XML_DES[:8] if n == 283 else _XML_DES)
    return d


def _dtype(im):
    """(big-endian sample dtype, kind) of an image subheader: kind 'iq' for two bands I and Q (complex), 'mp' for two
    bytes of magnitude and phase (SICD's AMP8I_PHS8I), else ''."""
    pv, nbpp = im['PVTYPE'], im['NBPP']
    try:
        dt = {('R', 32): '>f4', ('R', 64): '>f8', ('SI', 8): 'i1', ('SI', 16): '>i2', ('SI', 32): '>i4',
              ('INT', 8): 'u1', ('INT', 16): '>u2', ('INT', 32): '>u4', ('C', 64): '>c8'}[(pv, nbpp)]
    except KeyError:
        raise ValueError(f'NITF pixel type PVTYPE {pv!r} with {nbpp} bits per band is not supported') from None
    sub = [b['ISUBCAT'] for b in im['bands']]
    return np.dtype(dt), ('iq' if sub == ['I', 'Q'] else 'mp' if sub == ['M', 'P'] and dt == 'u1' else '')


class Nitf:
    """A parsed NITF 2.1 file. header: dict of the file header fields. images: one dict per image segment (the
    subheader's fields under their NITF names, numbers as int, `bands` (list of dicts), and `offset`, `length` of
    the pixel data in the file, bytes). des: one dict per data extension segment (subheader fields, the user-defined
    subheader's fields for XML_DATA_CONTENT, `offset`, `length`, and `data`, the segment's bytes). path."""

    def __init__(self, path):
        from ._xmlmeta import reading
        self.path = os.fspath(path) if not isinstance(path, str) else path
        with open(self.path, 'rb') as fh, reading(self.path, 'NITF header'):
            size = os.fstat(fh.fileno()).st_size
            self.header, sizes = _file_header(fh)
            pos = self.header['HL']
            self.images, self.des = [], []
            # every segment must lie within the file, in the order the header lists them, before anything is read
            for key in ('image', 'graphic', 'text', 'des', 'res'):
                for k, (lsh, ld) in enumerate(sizes[key]):
                    if pos + lsh + ld > size:
                        raise ValueError(f'{key} segment {k + 1} (a subheader of {lsh} bytes and {ld} bytes of data at '
                                         f'byte {pos}) runs past the end of the file ({size} bytes): truncated file, '
                                         'or wrong lengths in the header')
                    if key in ('image', 'des'):
                        fh.seek(pos)
                        sub = (_image_subheader if key == 'image' else _des_subheader)(fh.read(lsh))
                        sub.update(offset=pos + lsh, length=ld)
                        if key == 'des':
                            sub['data'] = fh.read(ld)
                        (self.images if key == 'image' else self.des).append(sub)
                    pos += lsh + ld

    def xml(self, kind=None):
        """The XML documents the file carries, as bytes, in file order: the data of each XML_DATA_CONTENT segment
        (and of the SICD_XML and SIDD_XML segments of files older than SICD 1.0). kind: 'SICD' or 'SIDD' keeps
        those whose root element is that; None keeps all."""
        from ._xmlmeta import root_name
        out = []
        for d in self.des:
            if d['DESID'] not in ('XML_DATA_CONTENT', 'SICD_XML', 'SIDD_XML'):
                continue
            x = d['data']
            if kind is not None and root_name(x) != kind:
                continue
            out.append(x)
        return out

    def groups(self):
        """The image segments grouped into images: [[segment index, ...], ...]. The segments SICDnnn of a SICD make
        one image, the segments SIDDnnnmmm of a SIDD one image per product nnn; any other segment is its own."""
        out, key = [], None
        for i, im in enumerate(self.images):
            m = re.fullmatch(r'(SICD)\d{3}|(SIDD\d{3})\d{3}', im['IID1'])
            k = (m.group(1) or m.group(2)) if m else None
            if k is not None and k == key:
                out[-1].append(i)
            else:
                out.append([i])
            key = k
        return out

    def _layout(self, im):
        """(big-endian dtype, shape, kind) of an image segment's pixels, checked against the segment's length."""
        if im['IC'] != 'NC':
            raise ValueError(f"NITF image compression IC = {im['IC']} is not supported (NC, uncompressed and "
                             'unmasked, only)')
        if im['NBPR'] * im['NBPC'] != 1:
            raise ValueError('NITF images of more than one block are not supported '
                             f"(NBPR {im['NBPR']}, NBPC {im['NBPC']})")
        nb = im['NBANDS']
        if nb > 1 and im['IMODE'] != 'P':
            raise ValueError(f"NITF band layout IMODE = {im['IMODE']} with {nb} bands is not supported "
                             '(P, by pixel, only)')
        dt, kind = _dtype(im)
        rows, cols = im['NROWS'], im['NCOLS']
        if rows < 1 or cols < 1 or nb < 1:
            raise ValueError(f'NITF image segment of {rows} rows, {cols} columns and {nb} bands')
        if rows * cols * nb * dt.itemsize != im['length']:
            raise ValueError(f"NITF image segment {im['IID1']}: {rows} rows x {cols} columns x {nb} bands of "
                             f"{dt.itemsize} bytes do not make its length, {im['length']} bytes")
        if kind == 'iq' and dt == np.dtype('>f4'):
            return np.dtype('>c8'), (rows, cols), kind
        return dt, ((rows, cols) if nb == 1 else (rows, cols, nb)), kind

    def _segment(self, im):
        dt, shape, kind = self._layout(im)
        return np.memmap(self.path, dtype=dt, mode='r', offset=im['offset'], shape=shape), kind

    def _group(self, index):
        g = self.groups()
        if not g:
            raise ValueError('the NITF file has no image segment')
        if not 0 <= index < len(g):
            raise ValueError(f'image {index}: the NITF file has {len(g)} images')
        return g[index]

    def shape(self, index=0):
        """(rows, cols) of image `index`, its segments' layouts checked against their lengths in the file (ValueError
        when they disagree, or for a layout that is not read)."""
        from ._xmlmeta import reading
        with reading(self.path, 'NITF image'):
            lay = [self._layout(self.images[i]) for i in self._group(index)]
            if len({(sh[1:], dt) for dt, sh, _ in lay}) != 1:
                raise ValueError('the segments of the NITF image differ in columns or pixel type')
            return sum(sh[0] for _, sh, _ in lay), lay[0][1][1]

    def pixels(self, native=False, index=0, amp_table=None):
        """The pixels of image `index` (see groups), its segments joined along rows.

        native False: in the file's big-endian types, a read-only memory map for an image of one segment and an
        array in memory for several. [rows, cols] big-endian complex64 for float32 I and Q bands, [rows, cols] for
        one band, [rows, cols, bands] otherwise (int16 I and Q, amplitude and phase bytes, RGB).
        native True: an array in memory in the machine's byte order, converted in row blocks on all cores; I and Q
        bands of any type become complex64 [rows, cols], and so do magnitude and phase bytes (SICD's AMP8I_PHS8I):
        amp_table[magnitude] exp(2j pi phase / 256), with amp_table the 256 amplitudes of the SICD's
        ImageData.AmpTable, or the magnitude byte itself when the SICD has none (None)."""
        self.shape(index)
        segs = [self._segment(self.images[i]) for i in self._group(index)]
        maps, kind = [s[0] for s in segs], segs[0][1]
        if not native:
            return maps[0] if len(maps) == 1 else np.concatenate(maps)
        from .patches import _rows
        m0 = maps[0]
        iq = kind == 'iq' and m0.ndim == 3
        if kind == 'mp':
            table = np.arange(256, dtype=np.float32) if amp_table is None else np.asarray(amp_table, np.float32).ravel()
            if table.size != 256:
                raise ValueError(f'{self.path}: the amplitude table of an AMP8I_PHS8I image has 256 entries, not '
                                 f'{table.size}')
            turn = np.exp(2j * np.pi * np.arange(256) / 256).astype(np.complex64)
            out = np.empty((sum(len(m) for m in maps), m0.shape[1]), np.complex64)
            r0 = 0
            for m in maps:
                dst = out[r0:r0 + len(m)]

                def blk(sl, m=m, dst=dst):
                    b = m[sl]
                    np.multiply(table[b[..., 0]], turn[b[..., 1]], out=dst[sl])
                _rows(blk, len(m), _chunk(m[:1].nbytes * 4))
                r0 += len(m)
            return out
        out = np.empty((sum(len(m) for m in maps), m0.shape[1]) if iq else (sum(len(m) for m in maps),) + m0.shape[1:],
                       np.complex64 if iq else m0.dtype.newbyteorder('='))
        r0 = 0
        for m in maps:
            dst = out[r0:r0 + len(m)]

            def blk(sl, m=m, dst=dst):
                if iq:
                    v = dst[sl].view(np.float32).reshape(sl.stop - sl.start, -1, 2)
                    v[...] = m[sl]
                else:
                    dst[sl] = m[sl]
            _rows(blk, len(m), _chunk(m[:1].nbytes))
            r0 += len(m)
        return out


def read(path):
    """Parse the headers of a NITF 2.1 file -> Nitf (header, images, des, xml(kind), pixels(native, index))."""
    return Nitf(path)


# ------------------------------------------------------------------------------------------------------- writing

def _chunk(row_bytes):
    return int(max(1, BLOCK_BYTES // max(int(row_bytes), 1)))


def _a(v, n):
    b = ('' if v is None else str(v)).encode('latin-1', 'replace')[:n]
    return b + b' ' * (n - len(b))


def _n(v, n):
    s = f'{int(v):0{n}d}'
    if len(s) > n:
        raise ValueError(f'{v} does not fit a NITF field of {n} digits')
    return s.encode()


def _time(v, default=''):
    """A date and time (datetime, numpy datetime64, or text with the digits in order) as CCYYMMDDhhmmss."""
    if v is None:
        return default
    if isinstance(v, datetime):
        v = v.strftime('%Y%m%d%H%M%S')
    d = re.sub(r'[^0-9]', '', str(v).split('.')[0])
    return (d + '0' * 14)[:14] if len(d) >= 8 else default


def _security(fields):
    return b''.join(_a(fields.get(k, ''), n) for k, n in _SEC)


def _dms(latlon):
    def one(value, digits, pos, neg):
        s = int(round(abs(float(value)) * 3600))
        return f'{s // 3600:0{digits}d}{s // 60 % 60:02d}{s % 60:02d}{pos if value >= 0 else neg}'
    return ''.join(one(lat, 2, 'N', 'S') + one(lon, 3, 'E', 'W') for lat, lon in np.asarray(latlon, np.float64))


def _geo(lat, lon):
    a, e2 = 6378137.0, (2 - 1 / 298.257223563) / 298.257223563
    lat, lon = np.radians(lat), np.radians(lon)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.stack([n * np.cos(lat) * np.cos(lon), n * np.cos(lat) * np.sin(lon), n * (1 - e2) * np.sin(lat)], -1)


def segment_corners(corners, rows, cols, r0, r1, c0=0, c1=None):
    """Latitude and longitude [4, 2] (degrees) of the corners of rows [r0, r1) and columns [c0, c1) of an image
    whose own corners are `corners` [4, 2] (first row first column, first row last column, last row last column,
    last row first column): bilinear in the corners' ECF positions on the ellipsoid (SICD Volume 2), with the
    fractions r / rows and c / cols."""
    from .io import ecf_to_geodetic
    corners = np.asarray(corners, np.float64)
    c1 = cols if c1 is None else c1
    if (r0, r1, c0, c1) == (0, rows, 0, cols):
        return corners
    p = _geo(corners[:, 0], corners[:, 1])
    out = []
    for r, c in ((r0, c0), (r0, c1), (r1, c1), (r1, c0)):
        u, v = r / rows, c / cols
        x = (1 - u) * (1 - v) * p[0] + (1 - u) * v * p[1] + u * v * p[2] + u * (1 - v) * p[3]
        lat, lon, _ = ecf_to_geodetic(x)
        out.append((float(lat), float(lon)))
    return np.array(out)


def segments(rows, cols, pixel_bytes, size_max=None, rows_max=None):
    """Row bounds [(first, last + 1), ...] of the image segments of a SICD or SIDD image: one segment where the
    image has at most rows_max rows (default ILOC_ROWS_MAX) and size_max bytes (default IS_SIZE_MAX), else
    segments of the largest whole number of rows within both, the last one holding the remainder."""
    size_max = IS_SIZE_MAX if size_max is None else size_max
    rows_max = ILOC_ROWS_MAX if rows_max is None else rows_max
    limit = min(rows_max, size_max // (cols * pixel_bytes))
    if limit < 1:
        raise ValueError(f'one row of {cols} pixels exceeds the NITF image segment size')
    return [(r, min(rows, r + limit)) for r in range(0, rows, limit)]


def _image_subheader_bytes(f, bands):
    icoms = list(f.get('ICOM') or [])[:9]
    geo = f.get('IGEOLO')
    b = [b'IM', _a(f['IID1'], 10), _a(f.get('IDATIM') or '', 14), _a(f.get('TGTID', ''), 17), _a(f.get('IID2', ''), 80),
         _security(f), b'0', _a(f.get('ISORCE', ''), 42), _n(f['NROWS'], 8), _n(f['NCOLS'], 8), _a(f['PVTYPE'], 3),
         _a(f['IREP'], 8), _a(f.get('ICAT', 'SAR'), 8), _n(f['ABPP'], 2), b'R',
         (b'G' + _a(geo, 60)) if geo else b' ', _n(len(icoms), 1)] + [_a(c, 80) for c in icoms] + [b'NC', _n(len(bands), 1)]
    for irepband, isubcat in bands:
        b += [_a(irepband, 2), _a(isubcat, 6), b'N', b'   ', b'0']
    b += [b'0', _a(f['IMODE'], 1), b'0001', b'0001', _n(0 if f['NCOLS'] > 8192 else f['NCOLS'], 4),
          _n(0 if f['NROWS'] > 8192 else f['NROWS'], 4), _n(f['NBPP'], 2), _n(f['IDLVL'], 3), _n(f['IALVL'], 3),
          _a(f['ILOC'], 10), b'1.0 ', b'00000', b'00000']
    return b''.join(b)


def _des_bytes(xml, kind, spec, f, when, corners):
    lpg = ''
    if corners is not None:
        c = [f'{lat:0=+12.8f}{lon:0=+13.8f}' for lat, lon in np.asarray(corners, np.float64)]
        lpg = ''.join(c + c[:1])
    user = dict(DESCRC='99999', DESSHFT='XML', DESSHDT=when, DESSHSI=_SOURCE.get(kind, ''), DESSHSV=spec[0],
                DESSHSD=spec[1], DESSHTN=spec[2], DESSHLPG=lpg)
    user.update({k: v for k, v in f.items() if k in ('DESSHRP', 'DESSHLPT', 'DESSHLI', 'DESSHLIN', 'DESSHABS')})
    return b''.join([b'DE', _a('XML_DATA_CONTENT', 25), b'01', _security(f), b'0773']
                    + [_a(user.get(k, ''), n) for k, n in _XML_DES])


def _clevel(size, dim):
    by_size = 3 if size < 50 << 20 else 5 if size < 1 << 30 else 6 if size < 2 << 30 else 7 if size < 10 << 30 else 9
    by_dim = 3 if dim <= 2048 else 5 if dim <= 8192 else 6 if dim <= 65536 else 7
    return max(by_size, by_dim)


def _file_header_bytes(f, images, des, clevel=3, fl=0):
    """images, des: [(subheader length, data length)]."""
    tail = [_n(len(images), 3)] + [_n(a, 6) + _n(b, 10) for a, b in images] + [b'000', b'000', b'000', _n(len(des), 3)] \
        + [_n(a, 4) + _n(b, 9) for a, b in des] + [b'000', b'00000', b'00000']
    hl = 360 + sum(len(t) for t in tail)
    return b''.join([b'NITF02.10', _n(clevel, 2), b'BF01', _a(f.get('OSTAID', 'Unknown'), 10),
                     _a(_time(f.get('FDT'), _time(datetime.now(timezone.utc))), 14), _a(f.get('FTITLE', ''), 80),
                     _security(f), b'00000', b'00000', b'0', b'\0\0\0', _a(f.get('ONAME', ''), 24),
                     _a(f.get('OPHONE', ''), 18), _n(fl, 12), _n(hl, 6)] + tail)


def _write_pixels(fd, offset, a, dtype, mm=None):
    """Rows of a [rows, cols(, bands)] as big-endian `dtype` at `offset` of the open file, in row blocks on all
    cores: through the file's memory map mm when there is one, else converted in scratch blocks and written with
    os.pwrite. (Buffered writes to one file take turns in the kernel whatever the number of threads; page faults
    on a shared map do not, and the map measured 2.5 GB/s against 1.7 GB/s into the page cache.)"""
    from .patches import _rows, _rows_with
    rows = a.shape[0]
    row_bytes = int(np.prod(a.shape[1:], dtype=np.int64)) * dtype.itemsize
    chunk = _chunk(row_bytes)
    cast = 'unsafe' if a.dtype.kind == dtype.kind else 'same_kind'
    if mm is not None and rows:
        dst = np.frombuffer(mm, dtype, int(np.prod(a.shape, dtype=np.int64)), offset).reshape(a.shape)
        _rows(lambda sl: np.copyto(dst[sl], a[sl], casting=cast), rows, chunk)
        return

    def put(buf, pos):
        mv = memoryview(buf).cast('B')
        while len(mv):                              # a write may be partial (2 GiB at most on Linux)
            k = os.pwrite(fd, mv, pos)
            mv, pos = mv[k:], pos + k
    if dtype.itemsize == 1 and a.dtype == dtype and a.flags.c_contiguous:
        _rows(lambda sl: put(a[sl], offset + sl.start * row_bytes), rows, chunk)
        return

    def blk(sl, buf):
        b = buf[:sl.stop - sl.start]
        np.copyto(b, a[sl], casting=cast)
        put(b, offset + sl.start * row_bytes)
    _rows_with(blk, rows, lambda: np.empty((min(chunk, rows),) + a.shape[1:], dtype), chunk)


def _map(fd, total):
    """The file as a writable shared memory map with its blocks allocated (a map of a sparse file would fault
    when the storage fills), or None where the file system gives neither; the file then has length `total`."""
    import mmap
    if os.environ.get('FASTSAR_NITF_MMAP', '1') != '0' and hasattr(os, 'posix_fallocate'):
        try:
            os.posix_fallocate(fd, 0, total)
            return mmap.mmap(fd, total, access=mmap.ACCESS_WRITE)
        except (OSError, ValueError):
            pass
    os.ftruncate(fd, total)
    return None


def write(path, images, des, fields=None):
    """Write a NITF 2.1 file of image segments and XML data extension segments.

    images: [dict(data=array [rows, cols] or [rows, cols, bands] (a view suffices), dtype=the big-endian sample
    type written (a complex type writes two bands), bands=[(IREPBAND, ISUBCAT), ...], and the subheader fields IID1,
    PVTYPE, IREP, ABPP, NBPP, IMODE, IDLVL, IALVL, ILOC, and optionally IGEOLO, IDATIM, TGTID, IID2, ISORCE, ICAT,
    ICOM)]. des: [dict(xml=bytes, kind='SICD' or 'SIDD', spec=(DESSHSV, DESSHSD, DESSHTN), when=DESSHDT text,
    corners=[4, 2] or None)]. fields: file header fields (OSTAID, FDT, FTITLE, ONAME, OPHONE), the security group
    without prefix (CLAS, CLSY, CODE, CTLH, REL, ...), copied to every subheader, and the optional
    XML_DATA_CONTENT fields DESSHRP, DESSHLPT, DESSHLI, DESSHLIN, DESSHABS. -> file length in bytes."""
    f = dict(fields or {})
    f.setdefault('CLAS', 'U')
    subs, lens = [], []
    for im in images:
        a, dt = im['data'], np.dtype(im['dtype'])
        g = dict(f, **{k: v for k, v in im.items() if k not in ('data', 'dtype', 'bands')}, NROWS=a.shape[0],
                 NCOLS=a.shape[1])
        subs.append(_image_subheader_bytes(g, im['bands']))
        lens.append(int(np.prod(a.shape, dtype=np.int64)) * dt.itemsize)
        if lens[-1] > IS_SIZE_MAX:
            raise ValueError('an image segment exceeds the NITF limit of 9,999,999,998 bytes')
    dsubs = [_des_bytes(d['xml'], d['kind'], d['spec'], f, d.get('when', ''), d.get('corners')) for d in des]
    args = (f, [(len(s), n) for s, n in zip(subs, lens)], [(len(s), len(d['xml'])) for s, d in zip(dsubs, des)])
    hl = len(_file_header_bytes(*args))
    total = hl + sum(len(s) for s in subs) + sum(lens) + sum(len(s) + len(d['xml']) for s, d in zip(dsubs, des))
    dim = max([max(im['data'].shape[:2]) for im in images] or [0])
    head = _file_header_bytes(*args, clevel=_clevel(total, dim), fl=total)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        mm = _map(fd, total)
        try:
            os.pwrite(fd, head, 0)
            pos = hl
            for im, s, n in zip(images, subs, lens):
                os.pwrite(fd, s, pos)
                _write_pixels(fd, pos + len(s), im['data'], np.dtype(im['dtype']), mm)
                pos += len(s) + n
            for d, s in zip(des, dsubs):
                os.pwrite(fd, s + bytes(d['xml']), pos)
                pos += len(s) + len(d['xml'])
        finally:
            if mm is not None:
                mm.close()
    finally:
        os.close(fd)
    return total


def _xml_facts(xml):
    """What the headers take from a SICD or SIDD XML document: dict(kind, core, collector, start, created,
    classification, corners [4, 2] or None), each None where the document lacks it."""
    from ._xmlmeta import fromstring
    root = fromstring(xml, 'XML document')
    loc = lambda t: t.rsplit('}', 1)[-1]

    def find(path):
        e = root
        for name in path.split('/'):
            e = next((c for c in e if loc(c.tag) == name), None)
            if e is None:
                return None
        return e

    def text(path):
        e = find(path)
        return e.text.strip() if e is not None and e.text and e.text.strip() else None
    kind = loc(root.tag)
    out = dict(kind=kind, corners=None)
    if kind == 'SIDD':
        cl = find('ProductCreation/Classification')
        out.update(core=text('ProductCreation/ProductName'),
                   created=text('ProductCreation/ProcessorInformation/ProcessingDateTime'),
                   collector=text('ExploitationFeatures/Collection/Information/SensorName'),
                   start=text('ExploitationFeatures/Collection/Information/CollectionDateTime'),
                   classification=None if cl is None else next(
                       (v for k, v in cl.attrib.items() if loc(k) == 'classification'), None))
    else:
        out.update(core=text('CollectionInfo/CoreName'), collector=text('CollectionInfo/CollectorName'),
                   start=text('Timeline/CollectStart'), created=text('ImageCreation/DateTime'),
                   classification=text('CollectionInfo/Classification'))
    ic = find('GeoData/ImageCorners')
    if ic is not None:
        pts = {}
        for c in ic:
            ll = {loc(k.tag): float(k.text) for k in c}
            pts[int(str(c.get('index', len(pts) + 1)).split(':')[0])] = (ll['Lat'], ll['Lon'])
        if sorted(pts) == [1, 2, 3, 4]:
            out['corners'] = np.array([pts[k] for k in (1, 2, 3, 4)])
    return out


def _clas(text):
    """The NITF classification letter of a SICD Classification text or a SIDD classification attribute."""
    t = str(text or 'U').strip().upper()
    return 'T' if t.startswith(('TOP', 'TS')) else t[0] if t[:1] and t[0] in 'UCSR' else 'U'


def _stamp(text):
    """An XML date and time as the DESSHDT field: CCYY-MM-DDThh:mm:ssZ."""
    d = _time(text)
    return f'{d[:4]}-{d[4:6]}-{d[6:8]}T{d[8:10]}:{d[10:12]}:{d[12:14]}Z' if d else ''


def _layout(a, kind, bands, sub, corners, f, size_max, rows_max):
    """The image segment dicts `write` takes for one image cut by the SICD and SIDD rule."""
    rows, cols = a.shape[:2]
    dt = np.dtype(sub['dtype'])
    bounds = segments(rows, cols, int(np.prod(a.shape[2:], dtype=np.int64)) * dt.itemsize, size_max, rows_max)
    out = []
    for k, (r0, r1) in enumerate(bounds):
        if corners is None:
            geo = None
        elif callable(corners):
            geo = _dms(corners(r0, r1, 0, cols))
        else:
            geo = _dms(segment_corners(corners, rows, cols, r0, r1))
        iid1 = f'SIDD001{k + 1:03d}' if kind == 'SIDD' else f'SICD{0 if len(bounds) == 1 else k + 1:03d}'
        out.append(dict(sub, data=a[r0:r1], bands=bands, IID1=iid1, IDLVL=k + 1, IALVL=k, IGEOLO=geo,
                        ILOC=f'{bounds[k - 1][1] - bounds[k - 1][0] if k else 0:05d}00000',
                        **{n: f[n] for n in ('IDATIM', 'TGTID', 'IID2', 'ISORCE', 'ICOM') if n in f}))
    return out


def write_sicd(path, a, xml, *, corners=None, fields=None, spec=SICD_SPEC, size_max=None, rows_max=None):
    """Write a complex image and its SICD XML as a SICD NITF (PixelType RE32F_IM32F).

    a: complex [rows, cols], written as big-endian float32 I and Q interleaved by pixel. xml: the SICD XML (bytes),
    whose ImageData.NumRows and NumCols must be a's shape. corners: [4, 2] latitude and longitude (degrees) of the
    image's corners (first row first column, first row last column, last row last column, last row first column);
    default the XML's GeoData.ImageCorners; they give IGEOLO (interpolated for each segment of a large image) and
    the DES's DESSHLPG. fields: header fields that replace the defaults taken from the XML: FTITLE and IID2
    ('SICD: ' and the core name), ISORCE ('SICD: ' and the collector name), IDATIM (the collection start), FDT (the
    image creation time; a datetime or text), OSTAID ('Unknown'), TGTID, ICOM (a list of comments), and the security
    group CLAS (from the Classification text), CLSY ('US'), CODE, CTLH, REL, ... spec: (DESSHSV, DESSHSD, DESSHTN),
    the version, date and namespace of the SICD specification the XML follows. size_max, rows_max: the segment
    limits (bytes, rows), for tests. -> file length in bytes."""
    a = a if isinstance(a, np.ndarray) else np.asarray(a)
    if a.ndim != 2 or a.dtype.kind != 'c':
        raise ValueError(f'a SICD image is a complex array [rows, cols], not {a.dtype} {a.shape}')
    xml = bytes(xml)
    x = _xml_facts(xml)
    corners = x['corners'] if corners is None else corners
    f = dict(FTITLE=f"SICD: {x['core'] or 'Unknown'}", ISORCE=f"SICD: {x['collector'] or 'Unknown Collector'}",
             IDATIM=_time(x['start']), FDT=x['created'], CLAS=_clas(x['classification']), CLSY='US')
    f.update(fields or {})
    f.setdefault('IID2', f['FTITLE'])
    f['IDATIM'] = _time(f.get('IDATIM'))
    sub = dict(dtype='>c8', PVTYPE='R', IREP='NODISPLY', ABPP=32, NBPP=32, IMODE='P')
    images = _layout(a, 'SICD', [('', 'I'), ('', 'Q')], sub, corners, f, size_max, rows_max)
    des = [dict(xml=xml, kind='SICD', spec=spec, when=_stamp(f.get('FDT') or datetime.now(timezone.utc)),
                corners=None if corners is None or callable(corners) else corners)]
    return write(path, images, des, f)


def write_sidd(path, a, xmls, *, corners=None, fields=None, size_max=None, rows_max=None):
    """Write a display image and its XML documents as a SIDD NITF.

    a: uint8 [rows, cols] (MONO8I) or [rows, cols, 3] (RGB24I). xmls: [(xml bytes, kind, spec), ...] in file order,
    the SIDD first; kind 'SIDD' or 'SICD', spec = (DESSHSV, DESSHSD, DESSHTN) of the specification the document
    follows. corners: [4, 2] latitude and longitude (degrees) of the image's corners, ordered as for write_sicd, or
    a function (r0, r1, c0, c1) -> [4, 2] giving the outer corners of the pixel block [r0, r1) x [c0, c1), called
    for each image segment; default the SIDD XML's GeoData.ImageCorners. fields: header fields that replace the
    defaults: FTITLE and IID2 ('SIDD: Unknown'), ISORCE (the SIDD's sensor name), IDATIM (its collection time), FDT
    (its processing time), OSTAID ('Unknown'), TGTID, ICOM, and the security group CLAS (the SIDD's classification
    attribute), CLSY ('US'), ... size_max, rows_max: the segment limits, for tests. -> file length in bytes."""
    a = a if isinstance(a, np.ndarray) else np.asarray(a)
    if a.dtype != np.uint8 or not (a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 3)):
        raise ValueError('a SIDD image is uint8 [rows, cols] (MONO8I) or [rows, cols, 3] (RGB24I), not '
                         f'{a.dtype} {a.shape}')
    facts = [_xml_facts(bytes(x[0])) for x in xmls]
    x = next((v for v in facts if v['kind'] == 'SIDD'), None)
    if x is None:
        raise ValueError('write_sidd needs a SIDD XML document')
    corners = x['corners'] if corners is None else corners
    f = dict(FTITLE='SIDD: Unknown', ISORCE=x['collector'] or 'Unknown', IDATIM=_time(x['start']), FDT=x['created'],
             CLAS=_clas(x['classification']), CLSY='US')
    f.update(fields or {})
    f.setdefault('IID2', f['FTITLE'])
    f['IDATIM'] = _time(f.get('IDATIM'))
    mono = a.ndim == 2
    sub = dict(dtype='u1', PVTYPE='INT', IREP='MONO' if mono else 'RGB', ABPP=8, NBPP=8, IMODE='B' if mono else 'P')
    bands = [('M', '')] if mono else [('R', ''), ('G', ''), ('B', '')]
    images = _layout(a, 'SIDD', bands, sub, corners, f, size_max, rows_max)
    des = [dict(xml=bytes(xb), kind=kind, spec=spec,
                when=_stamp(v['created'] or f.get('FDT') or datetime.now(timezone.utc)),
                corners=v['corners']) for (xb, kind, spec), v in zip(xmls, facts)]
    return write(path, images, des, f)
