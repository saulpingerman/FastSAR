"""CPHD files read and written without a third-party reader: the file header, the XML metadata, the per-vector
parameters, the support arrays and the signal arrays' places in the file.

    r = open_cphd('scene.cphd')
    r.cphd_version, r.cphd_header.SIGNAL_BLOCK_BYTE_OFFSET
    m = r.cphd_meta                          # fastsar._xmlmeta.Node over the XML
    m.Data.Channels[0].NumVectors, m.Channel.Parameters[0].Polarization.TxPol, m.SceneCoordinates.IARP.ECF.get_array()
    r.read_pvp_array(0)['TxPos']             # structured array of one channel's per-vector parameters
    r.read_pvp_variable('SC0', 0)            # one parameter, None when the file does not carry it
    r.signal_map(0)                          # memory map [vectors, samples, 2] of the samples as stored
    r.read_chip((0, 16), (0, 256), index=0)  # complex64, scaled by AmpSF
    r.read_support_array('AGP1')

What is implemented, from the CPHD design documents (NGA.STND.0068-1, versions 1.0.1 and 1.1.0) and the
pre-standard CPHD 0.3 design description:

  File header (1.x section 2.2; 0.3 likewise): the first line `CPHD/<version>`, then `KEY := VALUE` lines, ended by
  the section terminator (form feed, line feed). 1.x gives XML_BLOCK_*, PVP_BLOCK_*, SIGNAL_BLOCK_* and optionally
  SUPPORT_BLOCK_* sizes and byte offsets; 0.3 gives XML_*, VB_* and CPHD_* sizes and offsets.
  XML block (1.x section 3 and the XML description): parsed as it stands, any element reachable by name. The
  Spec below adds the plural names and list semantics that the rest of FastSAR reads (Data.Channels,
  Channel.Parameters, Antenna.AntPattern, SupportArray.AntGainPhase, SceneCoordinates.ImageAreaCornerPoints, and
  for 0.3 Data.ArraySize, Global.ImageArea.Corner, RadarCollection.RcvChannels, CollectionInfo.Parameters).
  PVP block (1.x section 2.4, binary formats of table 10-2 of 1.0.1): every element of the XML's PVP block that has
  Offset, Size and Format children is a parameter at Offset 8-byte words into each vector's NumBytesPVP bytes: the
  standard ones, the TxAntenna and RcvAntenna groups of 1.1.0 (TxACX, TxACY, TxEB, ...) and AddedPVP (named by its
  Name child). Formats: U1 to U8, I1 to I8, F4, F8, CI2 to CI16, CF8, CF16, S<n>, and multiple-component strings
  such as `X=F8;Y=F8;Z=F8;`. Channel c's array starts PVPArrayByteOffset into the block.
  0.3 vector-based parameters: the fields of VectorParameters in document order, 8 or 24 bytes each, with the
  FxParameters or TOAParameters fields last; channel c's vectors start NumBytesVBP times the vectors of the channels
  before it into the block, and its samples after those channels' samples.
  Signal block (1.x section 2.5): SignalArrayFormat CI2, CI4 or CF8, big-endian, channel c at SignalArrayByteOffset.
  A file with Data.SignalCompressionID holds a compressed signal block in a producer-defined format: its metadata
  and per-vector parameters read as usual, and sample reads raise a ValueError (there is no decoder to fall back on).
  Support block (1.x section 2.3): array `identifier` at its ArrayByteOffset, NumRows x NumCols elements in the
  ElementFormat its SupportArray entry gives.

write_cphd writes a CPHD 1.0.1 (or 1.1.0) file from XML and arrays: header, XML, PVP block and CF8, CI4 or CI2
signal arrays, big-endian, with the header's sizes and offsets computed here."""
import os
import types
import xml.etree.ElementTree as ET

import numpy as np

from . import _xmlmeta

TERMINATOR = b'\f\n'

_TEXT = ('Identifier CollectorName IlluminatorName CoreName CollectType ModeType ModeID Classification ReleaseInfo '
         'CountryCode DomainType SignalArrayFormat SignalCompressionID SampleType TxPol RcvPol RefChId TxAPCId TxAPATId '
         'RcvAPCId RcvAPATId ACFId AntGPId CODId DwellId DTAId TxWFId RcvId IAZArrayId AddedSupportArrayId Format '
         'ElementFormat Name EarthModel TxRcvPolarization TxPolarization SideOfTrack CollectionStart CollectStart '
         'RcvStartTime Application DateTime Site Profile XUnits YUnits ZUnits LFMEclipseMode SRPType Desc Parameter '
         'ElementFormat SegmentIdentifier TxWFId RcvId')
_INTS = ('NumVectors NumSamples SignalArrayByteOffset PVPArrayByteOffset CompressedSignalSize NumBytesPVP NumBytesVBP '
         'NumCPHDChannels NumSupportArrays SGN PhaseSGN RefFreqIndex Offset Size NumRows NumCols BytesPerElement '
         'ArrayByteOffset FirstLine NumLines FirstSample RefVectorIndex NumSRPs SRP_Index NumCODTimes NumDwellTimes '
         'NumTxWFs NumRcvs NumACFs NumAPCs NumAntPats NumSegments StartLine StartSample EndLine EndSample')
_lst = lambda path, tag, kind='list': (path, tag, kind)
SPEC = _xmlmeta.Spec(
    text=_TEXT.split(), ints=_INTS.split(),
    lists={('Antenna', 'AntCoordFrame'), ('Antenna', 'AntPhaseCenter'), ('Antenna', 'AntPattern'), ('PVP', 'AddedPVP'),
           ('SupportArray', 'IAZArray'), ('SupportArray', 'AntGainPhase'), ('SupportArray', 'DwellTimeArray'),
           ('SupportArray', 'AddedSupportArray'), ('Data', 'ArraySize'), ('Dwell', 'CODTime'), ('Dwell', 'DwellTime'),
           ('AntPattern', 'GainPhaseArray'), ('NoiseLevel', 'FxNoiseProfile'), ('TxRcv', 'TxWFParameters'),
           ('TxRcv', 'RcvParameters'), ('ProductInfo', 'CreationInfo')},
    aliases={('Data', 'Channels'): _lst('', 'Channel'), ('Data', 'SupportArrays'): _lst('', 'SupportArray'),
             ('Channel', 'Parameters'): _lst('', 'Parameters'),
             ('SceneCoordinates', 'ImageAreaCornerPoints'): _lst('ImageAreaCornerPoints', 'IACP'),
             ('Dwell', 'CODTimes'): _lst('', 'CODTime'), ('Dwell', 'DwellTimes'): _lst('', 'DwellTime'),
             ('ImageArea', 'Corner'): _lst('Corner', 'ACP'), ('ImageArea', 'Polygon'): _lst('Polygon', 'Vertex'),
             ('RadarCollection', 'RcvChannels'): _lst('RcvChannels', 'ChanParameters'),
             ('CollectionInfo', 'Parameters'): _lst('', 'Parameter', 'dict'),
             ('ProductInfo', 'Parameters'): _lst('', 'Parameter', 'dict')})

_SCALAR = {'U1': 'u1', 'U2': 'u2', 'U4': 'u4', 'U8': 'u8', 'I1': 'i1', 'I2': 'i2', 'I4': 'i4', 'I8': 'i8', 'F4': 'f4',
           'F8': 'f8', 'CF8': 'c8', 'CF16': 'c16'}
_PAIRS = {'CI2': 'i1', 'CI4': 'i2', 'CI8': 'i4', 'CI16': 'i8'}
_SIGNAL = {'CI2': '>i1', 'CI4': '>i2', 'CF8': '>f4'}
_SIGNAL03 = {'RE08I_IM08I': '>i1', 'RE16I_IM16I': '>i2', 'RE32F_IM32F': '>f4'}


def _one_format(s, order):
    s = s.strip()
    if s.startswith('S') and s[1:].isdigit():
        if not 0 < int(s[1:]) <= 1 << 16:
            raise ValueError(f'CPHD binary format {s!r}: a string of that length cannot be a parameter')
        return np.dtype(s)
    if s in _PAIRS:
        return np.dtype([('real', order + _PAIRS[s]), ('imag', order + _PAIRS[s])])
    if s not in _SCALAR:
        raise ValueError(f'CPHD binary format {s!r} is not one of the formats of the standard')
    return np.dtype(order + _SCALAR[s])


def format_dtype(fmt, order='>'):
    """The NumPy dtype of a CPHD binary format string ('F8', 'CI4', 'S24', 'X=F8;Y=F8;Z=F8;', 'Gain=F4;Phase=F4;')
    in byte order `order` ('>' as stored, '=' native). Components named X, Y, Z or DCX, DCY of one type make a
    sub-array; other multiple-component formats a structured dtype with the components' names."""
    fmt = fmt.strip()
    if '=' not in fmt:
        return _one_format(fmt.rstrip(';'), order)
    parts = [p.split('=') for p in fmt.split(';') if p.strip()]
    if any(len(p) != 2 or not p[0].strip() for p in parts):
        raise ValueError(f'CPHD binary format {fmt!r} is not NAME=TYPE; pairs')
    names, types = [p[0].strip() for p in parts], [_one_format(p[1], order) for p in parts]
    if tuple(names) in (('X', 'Y', 'Z'), ('DCX', 'DCY')) and len(set(types)) == 1:
        return np.dtype((types[0], len(names)))
    return np.dtype(list(zip(names, types)))


def _local(e):
    return e.tag.rsplit('}', 1)[-1]


def pvp_dtype(pvp, nbytes, order='>'):
    """The structured dtype of one vector of a CPHD 1.x file, from the XML's PVP element (an ElementTree element,
    with or without namespaces) and Data.NumBytesPVP. Fields in the XML's order, each at its own offset."""
    names, formats, offsets, spans = [], [], [], []

    def walk(e):
        kids = {_local(c): c for c in e}
        if {'Offset', 'Size', 'Format'} <= set(kids):
            name = kids['Name'].text.strip() if 'Name' in kids and kids['Name'].text else _local(e)
            try:
                off, size = int(kids['Offset'].text), int(kids['Size'].text)
            except (TypeError, ValueError):
                raise ValueError(f'CPHD per-vector parameter {name}: Offset and Size must be whole numbers') from None
            dt = format_dtype(kids['Format'].text or '', order)
            if off < 0 or size < 1 or dt.itemsize > 8 * size or 8 * (off + size) > int(nbytes):
                raise ValueError(f'CPHD per-vector parameter {name}: Offset {off}, Size {size} (8-byte words) and '
                                 f'Format {kids["Format"].text!r} ({dt.itemsize} bytes) do not fit a vector of '
                                 f'{nbytes} bytes')
            names.append(name)
            formats.append(dt)
            offsets.append(8 * off)
            spans.append((off, off + size, name))
        else:
            for c in e:
                walk(c)
    for c in pvp:
        walk(c)
    if not names or len(set(names)) != len(names):
        raise ValueError('the CPHD PVP block of the XML defines no parameters, or one twice')
    spans.sort()
    for (_, end, a), (start, _, b) in zip(spans, spans[1:]):
        if start < end:
            raise ValueError(f'CPHD per-vector parameters {a} and {b} overlap (word {start} lies inside {a})')
    return np.dtype(dict(names=names, formats=formats, offsets=offsets, itemsize=int(nbytes)))


def vbp_dtype(vp, nbytes, order='>'):
    """The structured dtype of one vector of a CPHD 0.3 file from the XML's VectorParameters element: each field 8
    bytes (float64) or 24 (three float64), the fields of FxParameters or TOAParameters 8 bytes each."""
    fields = []
    for c in vp:
        if len(c):
            fields += [(_local(k), order + 'f8') for k in c]
        elif (c.text or '').strip() == '24':
            fields.append((_local(c), order + 'f8', (3,)))
        elif (c.text or '').strip() == '8':
            fields.append((_local(c), order + 'f8'))
        else:
            raise ValueError(f'CPHD 0.3 vector parameter {_local(c)} has size {c.text!r}: 8 or 24 bytes')
    dt = np.dtype(fields)
    if dt.itemsize != int(nbytes):
        raise ValueError(f'CPHD 0.3 vector parameters take {dt.itemsize} bytes, NumBytesVBP says {nbytes}')
    return dt


def read_header(f):
    """The file header of an open binary file -> (version string, {key: int or str}, the header's length in bytes
    with its terminator)."""
    f.seek(0)
    buf = f.read(4096)
    while TERMINATOR not in buf:
        more = f.read(4096)
        if not buf.startswith(b'CPHD/'[:len(buf)]) or not buf:
            raise ValueError(f'not a CPHD file: it starts {buf[:16]!r}')
        if not more or len(buf) > (1 << 20):
            raise ValueError('not a CPHD file: no header section terminator (form feed, line feed)')
        buf += more
    head = buf[:buf.index(TERMINATOR)]
    lines = head.decode('utf-8', 'replace').split('\n')
    if not lines[0].startswith('CPHD/'):
        raise ValueError(f'not a CPHD file: the first line is {lines[0][:32]!r}')
    kv = {}
    for line in lines[1:]:
        if ' := ' in line:
            k, v = line.split(' := ', 1)
            kv[k.strip()] = int(v) if v.strip().lstrip('-').isdigit() else v.strip()
    return lines[0][5:].strip(), kv, len(head) + len(TERMINATOR)


class Reader:
    """An open CPHD 1.0.x, 1.1.0 or 0.3 file; see the module docstring. Channels are addressed by index or by
    identifier. Arrays come back in native byte order. The members are those of the reader FastSAR used before
    (cphd_meta, cphd_header, cphd_details, read_pvp_array, read_pvp_variable, read_chip, read_support_array), so
    that code taking a reader takes either."""

    def __init__(self, path):
        self.file_name = os.fspath(path)
        with open(self.file_name, 'rb') as f, _xmlmeta.reading(self.file_name, 'CPHD file'):
            size = os.fstat(f.fileno()).st_size
            self.cphd_version, kv, hlen = read_header(f)
            self.v03 = self.cphd_version.startswith('0.3')
            if not self.v03 and not self.cphd_version.startswith('1.'):
                raise ValueError(f'CPHD version {self.cphd_version!r} is not read (0.3, 1.0.1 and 1.1.0 are)')
            self._blocks = self._check_blocks(kv, hlen, size)
            off, n = self._blocks['XML']
            f.seek(off)
            self._xml = f.read(n)
            f.seek(off + n)
            if not self.v03 and f.read(2) != TERMINATOR:
                raise ValueError(f'no section terminator after the XML block ({n} bytes at byte {off}): its size or '
                                 'offset in the header is wrong')
            self.cphd_header = types.SimpleNamespace(**kv)
            self.cphd_details = types.SimpleNamespace(file_name=self.file_name, get_cphd_bytes=lambda: self._xml,
                                                      cphd_version=self.cphd_version, cphd_header=self.cphd_header)
            self._tree = _xmlmeta.fromstring(self._xml, 'CPHD XML block')
            if _local(self._tree) != 'CPHD':
                raise ValueError(f'the XML block holds a {_local(self._tree)} document, not CPHD')
            self.cphd_meta = _xmlmeta.parse(self._tree, SPEC)
            d = self.cphd_meta.Data
            if d is None:
                raise ValueError('the CPHD XML has no Data block')
            self._channels = (d.ArraySize if self.v03 else d.Channels) or []
            self._ids = [str(i) for i in range(len(self._channels))] if self.v03 else \
                [str(c.Identifier) for c in self._channels]
            self._pvp = {}
            self._check_layout()

    def _check_blocks(self, kv, hlen, size):
        """The header's blocks {name: (offset, size)}: every key present and a whole number, every block of positive
        size after the header and within the file, the blocks in the standard's order without overlap."""
        names = (('XML', 'XML_BYTE_OFFSET', 'XML_DATA_SIZE'), ('PVP', 'VB_BYTE_OFFSET', 'VB_DATA_SIZE'),
                 ('SIGNAL', 'CPHD_BYTE_OFFSET', 'CPHD_DATA_SIZE')) if self.v03 else \
                (('XML', 'XML_BLOCK_BYTE_OFFSET', 'XML_BLOCK_SIZE'),
                 ('SUPPORT', 'SUPPORT_BLOCK_BYTE_OFFSET', 'SUPPORT_BLOCK_SIZE'),
                 ('PVP', 'PVP_BLOCK_BYTE_OFFSET', 'PVP_BLOCK_SIZE'),
                 ('SIGNAL', 'SIGNAL_BLOCK_BYTE_OFFSET', 'SIGNAL_BLOCK_SIZE'))
        blocks, end, last = {}, hlen, 'header'
        for name, ko, ks in names:
            if name == 'SUPPORT' and ko not in kv and ks not in kv:
                continue
            for k in (ko, ks):
                if k not in kv:
                    raise ValueError(f'the CPHD {self.cphd_version} header has no {k}')
                if not isinstance(kv[k], int):
                    raise ValueError(f'CPHD header {k} is not a whole number: {kv[k]!r}')
            off, n = kv[ko], kv[ks]
            if n <= 0 or off < 0 or off + n > size:
                raise ValueError(f'CPHD header: the {name} block ({ks} {n} bytes at {ko} {off}) is not within the '
                                 f'file ({size} bytes): truncated file, or a wrong header')
            if off < end:
                raise ValueError(f'CPHD header: the {name} block starts at byte {off}, inside the {last} '
                                 f'(which ends at byte {end})')
            blocks[name], end, last = (off, n), off + n, f'{name} block'
        return blocks

    def _check_layout(self):
        """The XML's array sizes against the header's blocks: the per-vector parameters and the samples of every
        channel within their blocks, and the blocks no larger or smaller than the arrays make them."""
        d, ch = self.cphd_meta.Data, self._channels
        if not ch:
            raise ValueError('the CPHD XML lists no channels')

        def count(c, k, name):
            v = getattr(c, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v != int(v) or v < (1 if name.startswith('Num') else 0):
                raise ValueError(f'CPHD channel {k}: {name} is {v!r}')
            return int(v)
        nv = [count(c, k, 'NumVectors') for k, c in enumerate(ch)]
        ns = [count(c, k, 'NumSamples') for k, c in enumerate(ch)]
        nb = d.NumBytesVBP if self.v03 else d.NumBytesPVP
        if not isinstance(nb, (int, float)) or nb != int(nb) or nb < 8 or int(nb) % 8:
            raise ValueError(f'CPHD Data.{"NumBytesVBP" if self.v03 else "NumBytesPVP"} is {nb!r}: a positive multiple of 8')
        nb = int(nb)
        self.pvp_dtype()                                     # the parameters' formats and places within a vector
        psize, ssize = self._blocks['PVP'][1], self._blocks['SIGNAL'][1]
        if self.v03:
            if str(d.SampleType) not in _SIGNAL03:
                raise ValueError(f'CPHD 0.3 Data.SampleType {d.SampleType!r}: one of {", ".join(_SIGNAL03)}')
            isz = 2 * np.dtype(_SIGNAL03[str(d.SampleType)]).itemsize
            pv, sg = [(sum(nv[:k]) * nb, nv[k] * nb) for k in range(len(ch))], \
                [(sum(a * b for a, b in zip(nv[:k], ns[:k])) * isz, nv[k] * ns[k] * isz) for k in range(len(ch))]
        else:
            pv = [(count(c, k, 'PVPArrayByteOffset'), nv[k] * nb) for k, c in enumerate(ch)]
            sg = None
            if self.compression is None:
                if str(d.SignalArrayFormat) not in _SIGNAL:
                    raise ValueError(f'CPHD Data.SignalArrayFormat {d.SignalArrayFormat!r}: CI2, CI4 or CF8')
                isz = 2 * np.dtype(_SIGNAL[str(d.SignalArrayFormat)]).itemsize
                sg = [(count(c, k, 'SignalArrayByteOffset'), nv[k] * ns[k] * isz) for k, c in enumerate(ch)]
        for what, arrays, size in (('per-vector parameters', pv, psize), ('samples', sg, ssize)):
            if arrays is None:
                continue
            for k, (off, n) in enumerate(arrays):
                if off + n > size:
                    raise ValueError(f'CPHD channel {k}: its {what} ({nv[k]} vectors'
                                     + (f' x {ns[k]} samples' if what == 'samples' else '')
                                     + f', {n} bytes at offset {off}) run past their block of {size} bytes')
            if sum(n for _, n in arrays) != size:
                raise ValueError(f'CPHD: the {what} of the channels take {sum(n for _, n in arrays)} bytes, their '
                                 f'block in the header has {size}')

    def _index(self, index):
        if isinstance(index, str):
            if index not in self._ids:
                raise ValueError(f'{self.file_name}: channel {index!r} not among {self._ids}')
            return self._ids.index(index)
        if not 0 <= int(index) < len(self._channels):
            raise ValueError(f'{self.file_name}: channel {index} out of range: the file has {len(self._channels)} '
                             'channels')
        return int(index)

    def _child(self, name):
        return next((c for c in self._tree if _local(c) == name), None)

    def pvp_dtype(self, order='>'):
        """The dtype of one vector's parameters as stored (order '>') or as returned ('=')."""
        d = self.cphd_meta.Data
        if self.v03:
            return vbp_dtype(self._child('VectorParameters'), d.NumBytesVBP, order)
        return pvp_dtype(self._child('PVP'), d.NumBytesPVP, order)

    def pvp_map(self, index):
        """Read-only memory map [vectors] of one channel's per-vector parameters as stored (big-endian)."""
        c = self._index(index)
        d, h = self.cphd_meta.Data, self.cphd_header
        if self.v03:
            off = h.VB_BYTE_OFFSET + d.NumBytesVBP * sum(int(a.NumVectors) for a in self._channels[:c])
        else:
            off = h.PVP_BLOCK_BYTE_OFFSET + int(self._channels[c].PVPArrayByteOffset)
        return np.memmap(self.file_name, dtype=self.pvp_dtype(), mode='r', offset=off,
                         shape=(int(self._channels[c].NumVectors),))

    def _pvp_native(self, index):
        c = self._index(index)
        if c not in self._pvp:
            self._pvp[c] = np.array(self.pvp_map(c)).astype(self.pvp_dtype('='))
        return self._pvp[c]

    def read_pvp_array(self, index=0, the_range=None):
        """One channel's per-vector parameters [vectors] as a structured array (a copy)."""
        a = self._pvp_native(index)
        return a.copy() if the_range is None else a[slice(*the_range) if isinstance(the_range, tuple) else the_range].copy()

    def read_pvp_variable(self, name, index=0, the_range=None):
        """One per-vector parameter of a channel [vectors] or [vectors, n] (a copy), None when the file has none."""
        a = self._pvp_native(index)
        if name not in a.dtype.names:
            return None
        v = np.array(a[name])
        return v if the_range is None else v[slice(*the_range) if isinstance(the_range, tuple) else the_range]

    @property
    def compression(self):
        """Data.SignalCompressionID, or None for a file whose signal arrays are plain samples."""
        v = None if self.v03 else self.cphd_meta.Data.SignalCompressionID
        return v if v not in (None, '') else None

    def signal_map(self, index=0):
        """Read-only memory map [vectors, samples, 2] of one channel's real and imaginary parts as stored
        (big-endian int8, int16 or float32). ValueError for a compressed signal block."""
        c = self._index(index)
        d, h, ch = self.cphd_meta.Data, self.cphd_header, self._channels
        if self.compression is not None:
            raise ValueError(f'{self.file_name}: the signal block is compressed (SignalCompressionID '
                             f'{self.compression!r}), in a format its producer defines; FastSAR reads plain samples only')
        if self.v03:
            raw = _SIGNAL03[str(d.SampleType)]                 # (checked on opening)
            before = sum(int(a.NumVectors) * int(a.NumSamples) for a in ch[:c])
            off = h.CPHD_BYTE_OFFSET + before * 2 * np.dtype(raw).itemsize
        else:
            if str(d.SignalArrayFormat) not in _SIGNAL:
                raise ValueError(f'{self.file_name}: SignalArrayFormat {d.SignalArrayFormat!r}: CI2, CI4 or CF8')
            raw = _SIGNAL[str(d.SignalArrayFormat)]
            off = h.SIGNAL_BLOCK_BYTE_OFFSET + int(ch[c].SignalArrayByteOffset)
        return np.memmap(self.file_name, dtype=raw, mode='r', offset=off,
                         shape=(int(ch[c].NumVectors), int(ch[c].NumSamples), 2))

    def read_chip(self, pulses=None, samples=None, index=0):
        """Samples [pulses, samples] of a channel as complex64, each vector scaled by its AmpSF when the file has
        one. pulses, samples: (first, last + 1) or None for all."""
        sig = self.signal_map(index)
        p0, p1 = (0, sig.shape[0]) if pulses is None else (int(pulses[0]), int(pulses[1]))
        k0, k1 = (0, sig.shape[1]) if samples is None else (int(samples[0]), int(samples[1]))
        b = sig[p0:p1, k0:k1]
        out = np.empty(b.shape[:2], np.complex64)
        out.real, out.imag = b[..., 0], b[..., 1]
        amp = self._pvp_native(index)
        if 'AmpSF' in amp.dtype.names:
            out *= amp['AmpSF'][p0:p1, None].astype(np.float32)
        return out

    def read_support_array(self, identifier):
        """A support array [NumRows, NumCols] by identifier (or index), in the ElementFormat of its SupportArray
        entry (a structured array for formats such as Gain=F4;Phase=F4;)."""
        m = self.cphd_meta
        sizes = m.Data.SupportArrays or []
        if isinstance(identifier, (int, np.integer)):
            identifier = sizes[int(identifier)].Identifier
        sz = next((s for s in sizes if s.Identifier == str(identifier)), None)
        if sz is None:
            raise ValueError(f'{self.file_name}: support array {identifier!r} not among '
                             f'{[s.Identifier for s in sizes]}')
        fmt = None
        sa = self._child('SupportArray')
        for e in (sa if sa is not None else ()):
            kids = {_local(k): k for k in e}
            if 'Identifier' in kids and kids['Identifier'].text.strip() == str(identifier) and 'ElementFormat' in kids:
                fmt = kids['ElementFormat'].text
        if fmt is None:
            raise ValueError(f'{self.file_name}: support array {identifier!r} has no ElementFormat in the '
                             'SupportArray block')
        dt = format_dtype(fmt)
        if dt.itemsize != int(sz.BytesPerElement):
            raise ValueError(f'{self.file_name}: support array {identifier!r}: format {fmt!r} takes {dt.itemsize} '
                             'bytes per element, '
                             f'BytesPerElement says {sz.BytesPerElement}')
        if 'SUPPORT' not in self._blocks:
            raise ValueError(f'{self.file_name}: the header has no support block for support array {identifier!r}')
        off, n = int(sz.ArrayByteOffset), int(sz.NumRows) * int(sz.NumCols) * dt.itemsize
        if off < 0 or n <= 0 or off + n > self._blocks['SUPPORT'][1]:
            raise ValueError(f'{self.file_name}: support array {identifier!r} ({n} bytes at offset {off}) is not '
                             f"within the support block of {self._blocks['SUPPORT'][1]} bytes")
        a = np.memmap(self.file_name, dtype=dt, mode='r', offset=self._blocks['SUPPORT'][0] + off,
                      shape=(int(sz.NumRows), int(sz.NumCols)))
        return np.array(a).astype(format_dtype(fmt, '='))


def open_cphd(path):
    """Open a CPHD file -> Reader. The one place FastSAR opens a CPHD by path."""
    return Reader(path)


def start_time(text):
    """A CPHD collection start (xs:dateTime, UTC) as 'YYYY-MM-DDThh:mm:ss.ffffff', or the text itself when it does
    not parse."""
    try:
        return str(np.datetime64(str(text).strip().rstrip('Z'), 'us'))
    except ValueError:
        return str(text)


def write_cphd(path, xml, pvp, signal, support=None):
    """Write a CPHD 1.x file. xml: the metadata (str or bytes) with its Data block already giving, per channel,
    NumVectors, NumSamples and the byte offsets (signal arrays and PVP arrays in channel order, each following the
    one before), and SignalArrayFormat CF8, CI4 or CI2; the version is the one of its namespace. pvp: one
    structured array [vectors] per channel with the fields of the XML's PVP block (any byte order; fields are
    matched by name). signal: one complex array [vectors, samples] per channel, written as stored (no AmpSF
    scaling; integer formats are rounded). support: the support block's bytes, if any. The header's sizes, offsets,
    classification and release information are derived here."""
    xml = xml.encode() if isinstance(xml, str) else bytes(xml)
    root = ET.fromstring(xml)
    ns = root.tag[1:].split('}')[0] if root.tag.startswith('{') else ''
    version = ns.rsplit('/', 1)[-1] if '/cphd/' in ns else '1.0.1'
    find = lambda path: root.find('/'.join(f'{{{ns}}}{p}' if ns else p for p in path.split('/')))
    data = find('Data')
    nbytes = int(find('Data/NumBytesPVP').text)
    dt = pvp_dtype(find('PVP'), nbytes)
    raw = _SIGNAL[find('Data/SignalArrayFormat').text.strip()]
    chans = [c for c in data if _local(c) == 'Channel']
    if not len(chans) == len(pvp) == len(signal):
        raise ValueError(f'{len(chans)} channels in the XML, {len(pvp)} PVP arrays, {len(signal)} signal arrays')
    blocks = []
    for v in pvp:
        b = np.zeros(len(v), dt)
        for n in dt.names:
            b[n] = v[n]
        blocks.append(b)
    support = b'' if support is None else bytes(support)
    pvp_size = sum(b.nbytes for b in blocks)
    sig_size = sum(int(np.prod(np.shape(s))) * 2 * np.dtype(raw).itemsize for s in signal)
    text = lambda p, default: (find(p).text.strip() if find(p) is not None and find(p).text else default)

    def header(xo, so, po, go):
        lines = [f'XML_BLOCK_SIZE := {len(xml)}', f'XML_BLOCK_BYTE_OFFSET := {xo}']
        if support:
            lines += [f'SUPPORT_BLOCK_SIZE := {len(support)}', f'SUPPORT_BLOCK_BYTE_OFFSET := {so}']
        lines += [f'PVP_BLOCK_SIZE := {pvp_size}', f'PVP_BLOCK_BYTE_OFFSET := {po}', f'SIGNAL_BLOCK_SIZE := {sig_size}',
                  f'SIGNAL_BLOCK_BYTE_OFFSET := {go}',
                  f'CLASSIFICATION := {text("CollectionID/Classification", "UNCLASSIFIED")}',
                  f'RELEASE_INFO := {text("CollectionID/ReleaseInfo", "UNRESTRICTED")}']
        return f'CPHD/{version}\n'.encode() + ''.join(s + '\n' for s in lines).encode() + TERMINATOR
    xo = len(header(0, 0, 0, 0))
    for _ in range(4):                          # the offsets' digits lengthen the header they follow
        so = xo + len(xml) + len(TERMINATOR)
        po = so + len(support)
        go = po + pvp_size
        h = header(xo, so, po, go)
        if len(h) == xo:
            break
        xo = len(h)
    with open(path, 'wb') as f:
        f.write(h + xml + TERMINATOR + support)
        for b in blocks:
            f.write(b.tobytes())
        for s in signal:
            s = np.asarray(s)
            for p0 in range(0, s.shape[0], 1024):
                blk = s[p0:p0 + 1024]
                out = np.empty(blk.shape + (2,), raw)
                if raw == '>f4':
                    out[..., 0], out[..., 1] = blk.real, blk.imag
                else:
                    out[..., 0], out[..., 1] = np.rint(blk.real), np.rint(blk.imag)
                f.write(out.tobytes())
