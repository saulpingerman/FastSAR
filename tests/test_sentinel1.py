"""Sentinel-1 Level-0 packets: the user-data decoder against the specification's worked examples (with the
values of its printed tables), a bypass packet, the Huffman tables' completeness, the header parser on a
synthetic packet, and the sub-commutated orbit record."""
import numpy as np
import pytest

from fastsar import sentinel1 as s1


def bits_to_bytes(bits):
    bits = bits + '0' * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def align(b):
    return b + '0' * (-len(b) % 16)


def one_packet(user, baq_mode, nq):
    hdr = dict(quads=np.array([nq]), user_offset=np.array([0]), user_length=np.array([len(user)]), baq_mode=np.array([baq_mode]))
    return s1.decode_user_data(user, hdr, [0])


def test_huffman_tables():
    """Every tree is a complete prefix code (Kraft sum 1) with the expected number of magnitudes."""
    for brc, codes in enumerate(s1.HUFFMAN):
        assert len(codes) == [4, 5, 7, 10, 16][brc]
        assert sum(2.0 ** -len(c) for c in codes) == pytest.approx(1.0)
        assert not any(a != b and b.startswith(a) for a in codes for b in codes)


def test_fdbaq_example():
    """Section 4.4's example: BRC 2, THIDX 239, HCode 011 1110 (sign 0, magnitude 5): NRL(BRC 2, 5) times SF(239)."""
    ie = align('010' + '0' + '111110'); io = align('1' + '110'); qe = align(format(239, '08b') + '0' + '0'); qo = align('0' + '10')
    out, bad = one_packet(bits_to_bytes(ie + io + qe + qo), 12, 1)
    assert bad == 0
    assert out[0, 0].real == pytest.approx(2.5369 * 237.19, abs=0.01)
    assert out[0, 0].imag == pytest.approx(0.2305 * 237.19, abs=0.01)
    assert out[0, 1] == pytest.approx(-(1.1528 * 237.19) + 1j * (0.6916 * 237.19), abs=0.01)


def test_baq_example():
    """Section 4.3's example: 3-bit BAQ, THIDX 130, SCode 110 (sign 1, magnitude 2): -NRL(3-bit, 2) times SF(130)."""
    ie = align('110'); io = align('001'); qe = align(format(130, '08b') + '011'); qo = align('010')
    out, bad = one_packet(bits_to_bytes(ie + io + qe + qo), 3, 1)
    assert bad == 0
    assert out[0, 0].real == pytest.approx(-1.3655 * 100.58, abs=0.01)
    # simple reconstruction below the threshold: THIDX 2, magnitude 3 of 3-bit BAQ is A3[2]
    qe = align(format(2, '08b') + '011')
    out, _ = one_packet(bits_to_bytes(align('011') + io + qe + qo), 3, 1)
    assert out[0, 0].real == pytest.approx(3.12) and out[0, 0].imag == pytest.approx(3.12)


def test_bypass_packet():
    vals = [[5, -17, 300], [2, 4, -6], [-1, 0, 511], [100, -100, 7]]
    user = ''.join(align(''.join(('1' if v < 0 else '0') + format(abs(v), '09b') for v in ch)) for ch in vals)
    out, bad = one_packet(bits_to_bytes(user), 0, 3)
    assert bad == 0 and np.allclose(out[0], [5 - 1j, 2 + 100j, -17 + 0j, 4 - 100j, 300 + 511j, -6 + 7j])


def test_truncated_stream():
    """A user data field shorter than its codes is reported, not read past."""
    user = bits_to_bytes(align('010' + '0' + '111110'))
    out, bad = one_packet(user, 12, 8)
    assert bad == 1 and out.shape == (1, 16)


def packet_bytes(coarse=1475594816, fine=38437, rank=9, pri_code=20066, swst_code=4971, swl_code=9677, nq=12835, baq=12,
                 rgdec=1, txprr_code=0x8000 | 2000, txpsf_code=0x0000 | 19120, txpl_code=1706, signal_type=0, swath=36,
                 subcom_index=1, subcom_word=0x1234, pol=3, rx_channel=1, user=b''):
    h = bytearray(68)
    pdl = 62 + len(user) - 1
    h[0:2] = (0x0400 | 1052).to_bytes(2, 'big'); h[2:4] = (0xC000).to_bytes(2, 'big'); h[4:6] = pdl.to_bytes(2, 'big')
    h[6:10] = coarse.to_bytes(4, 'big'); h[10:12] = fine.to_bytes(2, 'big'); h[12:16] = (0x352EF853).to_bytes(4, 'big')
    h[20] = 11; h[21] = (0 << 4) | rx_channel; h[26] = subcom_index; h[27:29] = subcom_word.to_bytes(2, 'big')
    h[37] = baq; h[38] = 31; h[40] = rgdec; h[41] = 10
    h[42:44] = txprr_code.to_bytes(2, 'big'); h[44:46] = txpsf_code.to_bytes(2, 'big'); h[46:49] = txpl_code.to_bytes(3, 'big')
    h[49] = rank; h[50:53] = pri_code.to_bytes(3, 'big'); h[53:56] = swst_code.to_bytes(3, 'big'); h[56:59] = swl_code.to_bytes(3, 'big')
    h[59] = (0 << 7) | (pol << 4); h[62] = 0; h[63] = (signal_type << 4); h[64] = swath; h[65:67] = nq.to_bytes(2, 'big')
    return bytes(h) + user


def test_parse_packets():
    p = packet_bytes(user=bytes(4)) + packet_bytes(coarse=1475594817, signal_type=1, nq=0, user=bytes(4))
    h = s1.parse_packets(p)
    assert len(h['offset']) == 2 and h['apid'][0] == 1052 and h['sync'][0] == 0x352EF853
    assert h['time'][0] == pytest.approx(1475594816 + 38437 / 65536)
    assert h['rank'][0] == 9 and h['pri'][0] == pytest.approx(20066 / s1.F_REF) and h['swst'][0] == pytest.approx(4971 / s1.F_REF)
    assert h['tx_ramp_rate'][0] == pytest.approx(2000 * s1.F_REF ** 2 / 2 ** 21)                   # polarity bit 1: positive
    assert h['tx_start_frequency'][0] == pytest.approx(h['tx_ramp_rate'][0] / (4 * s1.F_REF) - 19120 * s1.F_REF / 2 ** 14)
    assert h['tx_pulse_length'][0] == pytest.approx(1706 / s1.F_REF) and h['quads'][0] == 12835 and h['quads'][1] == 0
    assert h['signal_type'].tolist() == [0, 1] and h['swath'][0] == 36 and h['polarisation'][0] == 3 and h['rx_channel'][0] == 1
    assert h['user_offset'][0] == 68 and h['user_length'][0] == 4 and h['offset'][1] == 72
    assert s1.sampling_frequency(1) == pytest.approx(2 / 3 * 4 * s1.F_REF)


def test_orbit_from_packets():
    """A position and velocity record spread over 22 packets' sub-commutated words comes back as one state vector."""
    pos = np.array([5083906.25, 4427022.69, -2160997.18]); vel = np.array([2762.09, 267.29, 7067.02]); t = 1475594816.0 + 0.25
    raw = pos.astype('>f8').tobytes() + vel.astype('>f4').tobytes() + bytes([0]) + int(t).to_bytes(4, 'big') + int(0.25 * 2 ** 24).to_bytes(3, 'big')
    words = [int.from_bytes(raw[2 * i:2 * i + 2], 'big') for i in range(22)]
    p = b''.join(packet_bytes(subcom_index=i + 1, subcom_word=w, user=bytes(4)) for i, w in enumerate(words))
    h = s1.parse_packets(p)
    ot, op, ov = s1.orbit_from_packets(h)
    assert ot.shape == (1,) and ot[0] == pytest.approx(t) and np.allclose(op[0], pos) and np.allclose(ov[0], vel, rtol=1e-6)
