"""CPHD 0.3 (the pre-standard NGA format) through read_cphd and form_cphd, on a simulated spotlight collection written
twice: as CPHD 1.0.1 by sim.write_cphd and as CPHD 0.3 by write_cphd03 below, with the same pulses, frequency grid,
samples and image-area corners. The two files must read to the same collection and form the same image. Also: a 0.3
file whose frequencies are offsets from an unstated reference frequency (RefFreqIndex set) is refused, 0.3's
TropoSRP is reported but not applied (CPHD 0.3 does not define its units), and the channel can be chosen by
polarization. Needs sarpy (which reads 0.3 but has no 0.3 writer, hence the writer here)."""
import os

import numpy as np
import pytest

import fastsar
from fastsar import io, sim

pytestmark = pytest.mark.needs('sarpy')

lat0, lon0, h0, heading = 40.7934, -77.8600, 351.0, 30.0
TERM = b'\f\n'


def write_cphd03(path, v1path, ref_freq_index=0, tropo=None, pols=('V:V',), sgn=None, nchan=1, sarcastic=False):
    """A CPHD 0.3 file holding the collection of the CPHD 1.x file v1path: per-vector parameters TxTime, TxPos,
    RcvTime, RcvPos, SRPPos, AmpSF (and TropoSRP when given) with the frequency grid as Fx0, Fx_SS, Fx1, Fx2 (Fx1 and
    Fx2 the band edges, distinct from Fx0), big-endian as 0.3 stores them; the samples as big-endian float32 pairs
    (RE32F_IM32F); the image area as the 1.x file's corner points at the reference point's height; pols in
    RadarCollection.RcvChannels. nchan channels hold the same collection, channel c with its samples divided by
    c + 1 and AmpSF = c + 1 (so every channel reads the same). sgn: the PhaseSGN written (the samples conjugated
    when it differs from the 1.x file's). sarcastic: as the SARCASTIC simulator writes 0.3, the polarization only as a
    CollectionInfo parameter (no RadarCollection) and the image-area corners all at latitude, longitude, height 0."""
    from sarpy.io.phase_history.converter import open_phase_history
    r = open_phase_history(v1path)
    m = r.cphd_meta
    P, K = m.Data.Channels[0].NumVectors, m.Data.Channels[0].NumSamples
    pv = lambda n: r.read_pvp_variable(n, 0)
    S = r.read_chip((0, P), (0, K), index=0)
    sc = m.SceneCoordinates
    hae = float(sc.IARP.LLH.HAE)
    g = lambda x: repr(float(x))
    corners = ''.join(f'<ACP index="{k + 1}"><Lat>{g(0 if sarcastic else c.Lat)}</Lat><Lon>{g(0 if sarcastic else c.Lon)}</Lon>'
                      f'<HAE>{g(0 if sarcastic else hae)}</HAE></ACP>' for k, c in enumerate(sc.ImageAreaCornerPoints))
    names = [('TxTime', 8), ('TxPos', 24), ('RcvTime', 8), ('RcvPos', 24), ('SRPPos', 24), ('AmpSF', 8)] + ([('TropoSRP', 8)] if tropo is not None else [])
    sgn1 = int(m.Global.SGN)
    sgn = sgn1 if sgn is None else sgn
    if sgn != sgn1:
        S = np.conj(S)
    fx = (pv('SC0'), pv('SCSS'), pv('SC0') - 0.25 * pv('SCSS'), pv('SC0') + (K - 0.75) * pv('SCSS'))
    nbytes = sum(b for _, b in names) + 32
    xml = f"""<CPHD xmlns="urn:CPHD:0.3">
<CollectionInfo><CollectorName>FastSAR simulator</CollectorName><CoreName>SIMULATED03</CoreName><CollectType>MONOSTATIC</CollectType>
<RadarMode><ModeType>SPOTLIGHT</ModeType></RadarMode><Classification>UNCLASSIFIED</Classification>{'<Parameter name="Polarisation">VV</Parameter>' if sarcastic else ''}</CollectionInfo>
<Data><SampleType>RE32F_IM32F</SampleType><NumCPHDChannels>{nchan}</NumCPHDChannels><NumBytesVBP>{nbytes}</NumBytesVBP>
{f'<ArraySize><NumVectors>{P}</NumVectors><NumSamples>{K}</NumSamples></ArraySize>' * nchan}</Data>
<Global><DomainType>FX</DomainType><PhaseSGN>{sgn}</PhaseSGN><RefFreqIndex>{ref_freq_index}</RefFreqIndex>
<CollectStart>2026-01-01T00:00:00.000000Z</CollectStart><CollectDuration>{g(pv('TxTime')[-1] - pv('TxTime')[0])}</CollectDuration>
<TxTime1>{g(pv('TxTime')[0])}</TxTime1><TxTime2>{g(pv('TxTime')[-1])}</TxTime2><ImageArea><Corner>{corners}</Corner></ImageArea></Global>
<Channel>{f'<Parameters><SRP_Index>0</SRP_Index><NomTOARateSF>1.0</NomTOARateSF><FxCtrNom>{g(0.5 * (fx[2][0] + fx[3][0]))}</FxCtrNom><BWSavedNom>{g(fx[3][0] - fx[2][0])}</BWSavedNom><TOASavedNom>{g(0.9 / fx[1][0])}</TOASavedNom></Parameters>' * nchan}</Channel>
<SRP><SRPType>STEPPED</SRPType><NumSRPs>1</NumSRPs></SRP>
{'' if sarcastic else '<RadarCollection>'}{'' if sarcastic else '<TxFrequency>'}<Min>{g(fx[2][0])}</Min><Max>{g(fx[3][0])}</Max>{'' if sarcastic else '</TxFrequency>'}{'' if sarcastic else f'<TxPolarization>{pols[0].split(":")[0]}</TxPolarization>'}
{'' if sarcastic else f'<RcvChannels size="{len(pols)}">' + ''.join(f'<ChanParameters index="{i + 1}"><TxRcvPolarization>{q}</TxRcvPolarization></ChanParameters>' for i, q in enumerate(pols)) + '</RcvChannels></RadarCollection>'}
<VectorParameters>{''.join(f'<{n}>{b}</{n}>' for n, b in names)}<FxParameters><Fx0>8</Fx0><Fx_SS>8</Fx_SS><Fx1>8</Fx1><Fx2>8</Fx2></FxParameters></VectorParameters>
</CPHD>""".encode()
    dt = np.dtype([(n, '>f8') if b == 8 else (n, '>f8', (3,)) for n, b in names] + [(n, '>f8') for n in ('Fx0', 'Fx_SS', 'Fx1', 'Fx2')])
    vbs, sigs = [], []
    for c in range(nchan):
        vb = np.zeros(P, dt)
        for n, _ in names:
            vb[n] = tropo if n == 'TropoSRP' else (c + 1.0 if n == 'AmpSF' else pv(n))
        for n, a in zip(('Fx0', 'Fx_SS', 'Fx1', 'Fx2'), fx):
            vb[n] = a
        sig = np.empty((P, K, 2), '>f4')
        sig[..., 0], sig[..., 1] = S.real / (c + 1.0), S.imag / (c + 1.0)
        vbs.append(vb); sigs.append(sig)
    vball, sigall = b''.join(v.tobytes() for v in vbs), b''.join(x.tobytes() for x in sigs)

    def header(xo, vo, co):
        return (b'CPHD/0.3\n' + f'XML_DATA_SIZE := {len(xml)}\nXML_BYTE_OFFSET := {xo}\nVB_DATA_SIZE := {len(vball)}\n'
                f'VB_BYTE_OFFSET := {vo}\nCPHD_DATA_SIZE := {len(sigall)}\nCPHD_BYTE_OFFSET := {co}\n'
                'CLASSIFICATION := UNCLASSIFIED\nRELEASE_INFO := UNRESTRICTED\n'.encode() + TERM)
    xo = len(header(10 ** 9, 10 ** 9, 10 ** 9)) + 16           # room for the offsets' digits, padded below
    vo = xo + len(xml) + len(TERM)
    co = vo + len(vball)
    h = header(xo, vo, co)
    with open(path, 'wb') as f:
        f.write(h + b'\0' * (xo - len(h)) + xml + TERM + vball + sigall)


@pytest.fixture(scope='module')
def files(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp('cphd03'))
    col = sim.make_collect(res=0.5, scene=60.0, r0=6e3, graze_deg=35.0)
    tg = np.array([[0.0, 0.0, 0.0], [10.0, -12.0, 0.0], [-12.0, 8.0, 0.0]])
    S = sim.simulate_brute(col, tg, np.ones(len(tg))).astype(np.complex64)
    p1, p3 = os.path.join(tmp, 'sim_101.cphd'), os.path.join(tmp, 'sim_03.cphd')
    sim.write_cphd(p1, col, S, lat0, lon0, h0, heading=heading)
    write_cphd03(p3, p1)
    return tmp, p1, p3


def test_reads_like_1x(files):
    _, p1, p3 = files
    c1, m1 = io.read_cphd(p1, meta=True)
    c3, m3 = io.read_cphd(p3, meta=True)
    assert m3['cphd_version'] == '0.3' and m1['cphd_version'].startswith('1.')
    assert 'CPHD 0.3 file' in m3['notes']
    assert c3['S'].shape == c1['S'].shape
    np.testing.assert_array_equal(c3['S'], c1['S'])
    np.testing.assert_allclose(c3['ant'], c1['ant'], rtol=0, atol=1e-6)
    assert c3['fmin'] == c1['fmin'] and c3['df'] == c1['df']
    np.testing.assert_allclose(m3['ref'], m1['ref'], rtol=0, atol=1e-6)
    assert m3['polarization'] == 'VV' and m3['channel'] == '0'
    assert m3['mode'] == 'SPOTLIGHT' and m3['core_name'] == 'SIMULATED03'
    assert m3['collector'] == m1['collector'] and str(m3['start']).startswith('2026-01-01')
    np.testing.assert_array_equal(m3['tx_time'], m1['tx_time'])
    np.testing.assert_array_equal(m3['rcv_time'], m1['rcv_time'])
    assert m3['pulses'] == m1['pulses']


def test_forms_like_1x(files):
    _, p1, p3 = files
    o1 = fastsar.form_cphd(p1, backend='cpu', spacing=0.4, height=h0)
    o3 = fastsar.form_cphd(p3, backend='cpu', spacing=0.4, height=h0)
    assert o3['image'].shape == o1['image'].shape
    a, b = o3['image'].astype(np.complex128), o1['image'].astype(np.complex128)
    err = 10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))
    print(f'0.3 against 1.0.1 image: {err:.1f} dB')
    assert err < -100


def test_channel_by_polarization(files):
    _, _, p3 = files
    c, m = io.read_cphd(p3, channel='VV', meta=True)
    assert m['channel_index'] == 0
    with pytest.raises(ValueError, match='not among'):
        io.read_cphd(p3, channel='HH')


def test_ref_freq_index_refused(files):
    tmp, p1, _ = files
    p = os.path.join(tmp, 'sim_03_reffreq.cphd')
    write_cphd03(p, p1, ref_freq_index=1)
    with pytest.raises(ValueError, match='RefFreqIndex'):
        io.read_cphd(p)


def test_tropo_reported_not_applied(files):
    tmp, p1, p3 = files
    p = os.path.join(tmp, 'sim_03_tropo.cphd')
    write_cphd03(p, p1, tropo=2.3)
    c, m = io.read_cphd(p, meta=True)
    c0 = io.read_cphd(p3)
    np.testing.assert_array_equal(c['S'], c0['S'])
    assert any('TropoSRP' in n and 'not applied' in n for n in m['notes'])


def test_second_channel(files):
    """Channel 1 of a two-channel 0.3 file (samples halved, AmpSF 2) reads like channel 0, positions included; sarpy's
    own 0.3 reader would give channel 1 the first channel's parameters shifted by one pulse."""
    tmp, p1, _ = files
    p = os.path.join(tmp, 'sim_03_2ch.cphd')
    write_cphd03(p, p1, nchan=2, pols=('V:V', 'V:H'))
    c0, m0 = io.read_cphd(p, channel=0, meta=True)
    c1, m1 = io.read_cphd(p, channel='VH', meta=True)
    assert m1['channel_index'] == 1 and m1['polarization'] == 'VH' and m0['polarization'] == 'VV'
    np.testing.assert_array_equal(c1['ant'], c0['ant'])
    np.testing.assert_array_equal(m1['tx_time'], m0['tx_time'])
    np.testing.assert_allclose(c1['S'], c0['S'], rtol=1e-6, atol=0)
    assert c1['fmin'] == c0['fmin'] and c1['df'] == c0['df']


def test_phase_sign_plus(files):
    """PhaseSGN = +1 with conjugated samples reads like the original."""
    tmp, p1, p3 = files
    p = os.path.join(tmp, 'sim_03_sgn.cphd')
    write_cphd03(p, p1, sgn=+1)
    np.testing.assert_array_equal(io.read_cphd(p)['S'], io.read_cphd(p3)['S'])


def test_polarization_counts_disagree(files):
    """Two receive channels listed for one array: the polarization is left unknown rather than guessed."""
    tmp, p1, _ = files
    p = os.path.join(tmp, 'sim_03_pols.cphd')
    write_cphd03(p, p1, pols=('V:V', 'V:H'))
    _, m = io.read_cphd(p, meta=True)
    assert m['polarization'] is None


def test_troposphere_true_refused(files):
    _, _, p3 = files
    with pytest.raises(ValueError, match='TropoSRP'):
        io.read_cphd(p3, troposphere=True)


def test_sarcastic_style(files):
    """The polarization from a CollectionInfo parameter when there is no RadarCollection; placeholder corners (all at
    0, 0, 0) treated as no image area, so form_cphd asks for an extent and forms with one."""
    tmp, p1, p3 = files
    p = os.path.join(tmp, 'sim_03_sarcastic.cphd')
    write_cphd03(p, p1, sarcastic=True)
    c, m = io.read_cphd(p, meta=True)
    assert m['polarization'] == 'VV'
    np.testing.assert_array_equal(c['S'], io.read_cphd(p3)['S'])
    with pytest.raises(ValueError, match='no planar image area'):
        fastsar.form_cphd(p, backend='cpu')
    o = fastsar.form_cphd(p, backend='cpu', extent=(30.0, 30.0), height=h0)
    assert np.isfinite(o['image']).all()


@pytest.mark.skipif(not os.environ.get('FASTSAR_CPHD03_FILE'), reason='FASTSAR_CPHD03_FILE (a real CPHD 0.3 file) not set')
def test_real_file():
    """A real CPHD 0.3 file, for example one of the SARCASTIC v2.0 T-72 data domes (doi 10.5522/04/c.5892392, CC BY
    4.0): it reads, and forms a finite image with a strong target over a 12 m extent."""
    f = os.environ['FASTSAR_CPHD03_FILE']
    c, m = io.read_cphd(f, meta=True)
    assert m['cphd_version'].startswith('0.3') and c['S'].size and np.isfinite(c['S']).all()
    o = fastsar.form_cphd(f, backend='cpu', extent=(12.0, 12.0))
    a = np.abs(o['image'])
    assert np.isfinite(a).all() and 20 * np.log10(a.max() / np.median(a)) > 30
