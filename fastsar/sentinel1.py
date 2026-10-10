"""Sentinel-1 Level-0: the instrument source packets of a measurement file (S1-IF-ASD-PL-0007, issue 13).

parse_packets reads the primary and secondary headers of every space packet in a measurement .dat file into
arrays (times, radar configuration, swath, signal type, number of quads, user-data offsets). decode_user_data
turns the user data of selected packets into complex samples: the four channels IE, IO, QE, QO (even and odd
samples' in-phase and quadrature parts), each a sequence of BAQ blocks of 128 codes; in FDBAQ (format D) each
IE block starts with a 3-bit bit-rate code and each QE block with an 8-bit threshold index, the codes are a sign
bit and a Huffman-coded magnitude (one tree per bit-rate code), and a sample is reconstructed either simply
(the magnitude, or a parameter A or B at the largest magnitude) or normally (a normalised reconstruction level
times the sigma factor of the threshold index); in BAQ (format C) the codes are fixed width; in bypass and
decimation-only data (formats A and B) they are 10 bits. The decoder is C++ (compiled on first use like the CPU
kernels). orbit_from_packets assembles the sub-commutated position and velocity records (one every 64 packets)
into state vectors on GPS time, the time base of the packet datation.

The radar timing conventions used by io.read_sentinel1: the data of a packet are the echo of the pulse
transmitted RANK pulse repetition intervals before the packet's own, and sample 0 of the window lies
RANK * PRI + SWST + 320 / (8 f_ref) after that pulse (the decimation filter's suppressed transient); the chirp
starts at TXPSF relative to the carrier and sweeps at TXPRR for TXPL.
"""
import ctypes
import os

import numpy as np

C = 299792458.0
F_REF = 37.53472224e6                       # Hz, the instrument's reference frequency
F_CARRIER = 5.405e9                         # Hz
T_SUPPRESSED = 320.0 / (8.0 * F_REF)        # s, decimation filter transient at the start of the sampling window
DECIMATION = {0: (3, 4), 1: (2, 3), 3: (5, 9), 4: (4, 9), 5: (3, 8), 6: (1, 3), 7: (1, 6), 8: (3, 7), 9: (5, 16), 10: (3, 26), 11: (4, 11)}
SIGNAL_TYPES = {0: 'echo', 1: 'noise', 8: 'tx cal', 9: 'rx cal', 10: 'epdn cal', 11: 'ta cal', 12: 'apdn cal', 15: 'txh cal iso'}

# Table 5.2-1: simple reconstruction parameters A (BAQ) and B (FDBAQ) by threshold index
A_BAQ = {3: [3.0, 3.0, 3.12, 3.55], 4: [7.0, 7.0, 7.0, 7.17, 7.4, 7.76],
         5: [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.44, 15.56, 16.11, 16.38, 16.65]}
B_FDBAQ = [[3.0, 3.0, 3.16, 3.53], [4.0, 4.0, 4.08, 4.37], [6.0, 6.0, 6.0, 6.15, 6.5, 6.88],
           [9.0, 9.0, 9.0, 9.0, 9.36, 9.5, 10.1], [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.22, 15.5, 16.05]]
# Table 5.2-2: normalised reconstruction levels by magnitude code
NRL_BAQ = {3: [0.2490, 0.7681, 1.3655, 2.1864], 4: [0.1290, 0.3900, 0.6601, 0.9471, 1.2623, 1.6261, 2.0793, 2.7467],
           5: [0.0660, 0.1985, 0.3320, 0.4677, 0.6061, 0.7487, 0.8964, 1.0510, 1.2143, 1.3896, 1.5800, 1.7914, 2.0329, 2.3234, 2.6971, 3.2692]}
NRL_FDBAQ = [[0.3637, 1.0915, 1.8208, 2.6406], [0.3042, 0.9127, 1.5216, 2.1313, 2.8426],
             [0.2305, 0.6916, 1.1528, 1.6140, 2.0754, 2.5369, 3.1191],
             [0.1702, 0.5107, 0.8511, 1.1916, 1.5321, 1.8726, 2.2131, 2.5536, 2.8942, 3.3744],
             [0.1130, 0.3389, 0.5649, 0.7908, 1.0167, 1.2428, 1.4687, 1.6947, 1.9206, 2.1466, 2.3725, 2.5985, 2.8244, 3.0504, 3.2764, 3.6623]]
# Table 5.2-3: sigma factors by threshold index (the printed table to two decimals: 0.6267 steps to index 99,
# 62.98 then 1.2533 steps, 255.99 at the end)
SIGMA = np.array([round(i * 0.6267, 2) if i <= 99 else min(255.99, round(62.98 + (i - 100) * 1.2533, 2)) for i in range(256)])
# the Huffman codes of the five bit-rate codes (Fig. 4-7 to 4-11; checked against a public decoder on real data),
# magnitude code by code word
HUFFMAN = [['0', '10', '110', '111'], ['0', '10', '110', '1110', '1111'],
           ['0', '10', '110', '1110', '11110', '111110', '111111'],
           ['00', '01', '10', '110', '1110', '11110', '111110', '1111110', '11111110', '11111111'],
           ['00', '010', '011', '100', '101', '1100', '1101', '1110', '11110', '111110', '11111100', '11111101', '111111100', '111111101',
            '111111110', '111111111']]
SIMPLE_THRESHOLD_FDBAQ = [3, 3, 5, 6, 8]            # simple reconstruction for THIDX up to this value, by BRC
SIMPLE_THRESHOLD_BAQ = {3: 3, 4: 5, 5: 10}

_SRC = r'''
#include <cmath>
#include <cstdint>
#include <cstring>
struct Bits { const uint8_t* p; long n; long pos; bool bad;
  Bits(const uint8_t* p_, long n_) : p(p_), n(n_), pos(0), bad(false) {}
  inline unsigned bit() { if (pos >= 8 * n) { bad = true; return 0; } unsigned b = (p[pos >> 3] >> (7 - (pos & 7))) & 1; ++pos; return b; }
  inline unsigned bits(int k) { unsigned v = 0; for (int i = 0; i < k; ++i) v = (v << 1) | bit(); return v; }
  inline void align16() { pos = (pos + 15) & ~15L; }
};
// Huffman tries: node i has children kid[i][0], kid[i][1]; a leaf stores -(mcode + 1)
static int kid[5][64][2]; static int built = 0;
static const char* codes[5][16] = {
  {"0", "10", "110", "111"},
  {"0", "10", "110", "1110", "1111"},
  {"0", "10", "110", "1110", "11110", "111110", "111111"},
  {"00", "01", "10", "110", "1110", "11110", "111110", "1111110", "11111110", "11111111"},
  {"00", "010", "011", "100", "101", "1100", "1101", "1110", "11110", "111110", "11111100", "11111101", "111111100", "111111101", "111111110", "111111111"}};
static const int ncodes[5] = {4, 5, 7, 10, 16};
static void build() {
  if (built) return;
  for (int t = 0; t < 5; ++t) {
    int nn = 1; memset(kid[t], 0, sizeof(kid[t]));
    for (int m = 0; m < ncodes[t]; ++m) {
      int node = 0; const char* c = codes[t][m];
      for (int i = 0; c[i]; ++i) {
        int b = c[i] - '0';
        if (c[i + 1] == 0) { kid[t][node][b] = -(m + 1); }
        else { if (kid[t][node][b] == 0) kid[t][node][b] = nn++; node = kid[t][node][b]; }
      }
    }
  }
  built = 1;
}
static inline int huff(Bits& r, int brc) {
  int node = 0;
  for (int i = 0; i < 16; ++i) { int nx = kid[brc][node][r.bit()]; if (nx < 0) return -nx - 1; node = nx; if (r.bad) return 0; }
  r.bad = true; return 0;
}
static const double A3[4] = {3.0, 3.0, 3.12, 3.55}, A4[6] = {7.0, 7.0, 7.0, 7.17, 7.4, 7.76},
  A5[11] = {15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.44, 15.56, 16.11, 16.38, 16.65};
static const double B0[4] = {3.0, 3.0, 3.16, 3.53}, B1[4] = {4.0, 4.0, 4.08, 4.37}, B2[6] = {6.0, 6.0, 6.0, 6.15, 6.5, 6.88},
  B3[7] = {9.0, 9.0, 9.0, 9.0, 9.36, 9.5, 10.1}, B4[9] = {15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.22, 15.5, 16.05};
static const double* BT[5] = {B0, B1, B2, B3, B4}; static const int bthr[5] = {3, 3, 5, 6, 8}, bmax[5] = {3, 4, 6, 9, 15};
static const double N3[4] = {0.2490, 0.7681, 1.3655, 2.1864}, N4[8] = {0.1290, 0.3900, 0.6601, 0.9471, 1.2623, 1.6261, 2.0793, 2.7467},
  N5[16] = {0.0660, 0.1985, 0.3320, 0.4677, 0.6061, 0.7487, 0.8964, 1.0510, 1.2143, 1.3896, 1.5800, 1.7914, 2.0329, 2.3234, 2.6971, 3.2692};
static const double F0[4] = {0.3637, 1.0915, 1.8208, 2.6406}, F1[5] = {0.3042, 0.9127, 1.5216, 2.1313, 2.8426},
  F2[7] = {0.2305, 0.6916, 1.1528, 1.6140, 2.0754, 2.5369, 3.1191},
  F3[10] = {0.1702, 0.5107, 0.8511, 1.1916, 1.5321, 1.8726, 2.2131, 2.5536, 2.8942, 3.3744},
  F4[16] = {0.1130, 0.3389, 0.5649, 0.7908, 1.0167, 1.2428, 1.4687, 1.6947, 1.9206, 2.1466, 2.3725, 2.5985, 2.8244, 3.0504, 3.2764, 3.6623};
static const double* FT[5] = {F0, F1, F2, F3, F4};
static inline double sigma(int thidx) { double v = thidx <= 99 ? thidx * 0.6267 : 62.98 + (thidx - 100) * 1.2533; if (v > 255.99) v = 255.99; return std::round(v * 100.0) / 100.0; }
static inline double recon_fdbaq(int brc, int thidx, int sign, int m) {
  double v;
  if (thidx <= bthr[brc]) v = (m < bmax[brc]) ? (double)m : BT[brc][thidx];
  else v = FT[brc][m] * sigma(thidx);
  return sign ? -v : v;
}
static inline double recon_baq(int nbits, int thidx, int sign, int m) {
  double v; int thr = nbits == 3 ? 3 : (nbits == 4 ? 5 : 10); int mx = (1 << (nbits - 1)) - 1;
  if (thidx <= thr) v = (m < mx) ? (double)m : (nbits == 3 ? A3[thidx] : (nbits == 4 ? A4[thidx] : A5[thidx]));
  else v = (nbits == 3 ? N3[m] : (nbits == 4 ? N4[m] : N5[m])) * sigma(thidx);
  return sign ? -v : v;
}
// one packet: user data p[n], nq quads, baq mode; writes 2 nq complex samples (re, im interleaved) to out;
// returns 0, or 1 if the bit stream ran out (the samples are then partial)
extern "C" int decode_packet(const uint8_t* p, long n, int nq, int baqmod, float* out) {
  build();
  Bits r(p, n);
  int nb = (nq + 127) / 128;
  if (nb > 512 || nq > 65536) return 2;
  int brc[512]; int thidx[512];
  // first pass: the sign and magnitude code of every sample per channel (0 IE, 1 IO, 2 QE, 3 QO); the threshold
  // index of a block is only known after the QE channel, so values are reconstructed afterwards
  static thread_local uint8_t sgn[4][65536]; static thread_local uint16_t mag[4][65536];
  for (int ch = 0; ch < 4; ++ch) {
    int k = 0;
    for (int b = 0; b < nb; ++b) {
      int cnt = (b == nb - 1) ? (nq - 128 * b) : 128;
      if (baqmod >= 12) {                                   // FDBAQ
        if (ch == 0) { brc[b] = r.bits(3); if (brc[b] > 4) { r.bad = true; brc[b] = 0; } }
        if (ch == 2) thidx[b] = r.bits(8);
        for (int i = 0; i < cnt; ++i) { sgn[ch][k] = r.bit(); mag[ch][k] = huff(r, brc[b]); ++k; }
      } else if (baqmod >= 3 && baqmod <= 5) {              // BAQ: sign + (nbits - 1) magnitude bits
        if (ch == 2) thidx[b] = r.bits(8);
        for (int i = 0; i < cnt; ++i) { sgn[ch][k] = r.bit(); mag[ch][k] = r.bits(baqmod - 1); ++k; }
      } else {                                              // bypass / decimation only: 10-bit sign + magnitude
        for (int i = 0; i < cnt; ++i) { sgn[ch][k] = r.bit(); mag[ch][k] = r.bits(9); ++k; }
      }
    }
    r.align16();
  }
  for (int ch = 0; ch < 4; ++ch) {
    for (int i = 0; i < nq; ++i) {
      int b = i >> 7; double v;
      if (baqmod >= 12) v = recon_fdbaq(brc[b], thidx[b], sgn[ch][i], mag[ch][i]);
      else if (baqmod >= 3 && baqmod <= 5) v = recon_baq(baqmod, thidx[b], sgn[ch][i], mag[ch][i]);
      else v = sgn[ch][i] ? -(double)mag[ch][i] : (double)mag[ch][i];
      // even sample i: IE (ch 0) real, QE (ch 2) imaginary; odd sample i: IO (ch 1) real, QO (ch 3) imaginary
      int o = (ch & 1) ? 4 * i + 2 : 4 * i;
      out[o + (ch >> 1)] = (float)v;
    }
  }
  return r.bad ? 1 : 0;
}
'''
_lib = None


def _decoder():
    global _lib
    if _lib is None:
        from ._build import shared_object, default_flags
        L = ctypes.CDLL(shared_object('s1_decode', _SRC, default_flags() + ['-std=c++17']))
        L.decode_packet.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        L.decode_packet.restype = ctypes.c_int
        _lib = L
    return _lib


def parse_packets(data):
    """The headers of every space packet in a Level-0 measurement file (bytes or a memory map): a dict of arrays
    over packets (offset and length of the packet, user-data offset and length, coarse and fine time as GPS seconds
    in `time`, data take id, ecc number, test mode, rx channel id, instrument configuration id, sub-commutation word
    index and word, space packet count, pri count, error flag, baq mode, baq block length, range decimation, rx gain
    in dB, tx ramp rate in Hz/s, tx pulse start frequency in Hz, tx pulse length in s, rank, pri, swst and swl in s,
    polarisation code, elevation and azimuth beam addresses, calibration mode, tx pulse number, signal type, swap
    flag, swath number, number of quads)."""
    buf = np.frombuffer(data, np.uint8)
    n = len(buf)
    offsets = []
    pos = 0
    while pos + 6 <= n:
        pdl = int(buf[pos + 4]) << 8 | int(buf[pos + 5])
        length = 6 + pdl + 1
        if pos + length > n:
            break
        offsets.append(pos)
        pos += length
    off = np.array(offsets, np.int64)
    P = len(off)
    h = np.stack([buf[off + k] for k in range(68)], 1).astype(np.int64)      # the 6 + 62 header octets of every packet
    be = lambda a, b: sum(h[:, k] << (8 * (b - k)) for k in range(a, b + 1))
    pdl = be(4, 5)
    out = dict(offset=off, length=6 + pdl + 1, user_offset=off + 68, user_length=pdl + 1 - 62,
               version=h[:, 0] >> 5, packet_type=(h[:, 0] >> 4) & 1, secondary_header=(h[:, 0] >> 3) & 1, apid=be(0, 1) & 0x7FF,
               sequence_flags=h[:, 2] >> 6, sequence_count=be(2, 3) & 0x3FFF,
               coarse_time=be(6, 9), fine_time=be(10, 11), sync=be(12, 15), data_take_id=be(16, 19), ecc=h[:, 20],
               test_mode=(h[:, 21] >> 4) & 7, rx_channel=h[:, 21] & 0xF, icid=be(22, 25),
               subcom_index=h[:, 26], subcom_word=be(27, 28), packet_count=be(29, 32), pri_count=be(33, 36),
               error_flag=h[:, 37] >> 7, baq_mode=h[:, 37] & 0x1F, baq_block_length=8 * (h[:, 38] + 1), range_decimation=h[:, 40],
               rx_gain_db=-0.5 * h[:, 41], rank=h[:, 49] & 0x1F)
    out['time'] = out['coarse_time'] + out['fine_time'] / 65536.0
    txprr_code, txpsf_code = be(42, 43), be(44, 45)
    s_prr = np.where(txprr_code >> 15 == 1, 1.0, -1.0)
    out['tx_ramp_rate'] = s_prr * (txprr_code & 0x7FFF) * F_REF ** 2 / 2 ** 21
    s_psf = np.where(txpsf_code >> 15 == 1, 1.0, -1.0)
    out['tx_start_frequency'] = out['tx_ramp_rate'] / (4 * F_REF) + s_psf * (txpsf_code & 0x7FFF) * F_REF / 2 ** 14
    out['tx_pulse_length'] = be(46, 48) / F_REF
    out['pri'] = be(50, 52) / F_REF
    out['swst'] = be(53, 55) / F_REF
    out['swl'] = be(56, 58) / F_REF
    out['ssb_flag'] = h[:, 59] >> 7
    out['polarisation'] = (h[:, 59] >> 4) & 7
    out['elevation_beam'] = h[:, 60] >> 4
    out['azimuth_beam'] = ((h[:, 60] & 0xF) << 6) | (h[:, 61] >> 2)
    out['cal_mode'] = h[:, 62] >> 6
    out['tx_pulse_number'] = h[:, 62] & 0x1F
    out['signal_type'] = h[:, 63] >> 4
    out['swap'] = h[:, 63] & 1
    out['swath'] = h[:, 64]
    out['quads'] = be(65, 66)
    return out


def sampling_frequency(range_decimation):
    """Hz, after the decimation filter of the given code (Table 3.2-14: L / M times 4 f_ref)."""
    L, M = DECIMATION[int(range_decimation)]
    return L / M * 4.0 * F_REF


def orbit_from_packets(hdr):
    """The sub-commutated position, velocity and time words of the packets, assembled into state vectors:
    (times [n] GPS seconds, positions [n, 3] ECEF m, velocities [n, 3] m/s). A record is 22 words (indices 1 to
    22) spread over 22 consecutive packets; records that are incomplete or repeat the previous time are dropped."""
    idx, word = np.asarray(hdr['subcom_index']), np.asarray(hdr['subcom_word'])
    starts = np.nonzero(idx == 1)[0]
    recs = []
    for s in starts:
        if s + 22 > len(idx) or not np.array_equal(idx[s:s + 22], np.arange(1, 23)):
            continue
        w = word[s:s + 22].astype(np.uint64)
        raw = b''.join(int(x).to_bytes(2, 'big') for x in w)
        pos = np.frombuffer(raw[0:24], '>f8')
        vel = np.frombuffer(raw[24:36], '>f4').astype(np.float64)
        # Table 3.2-7: the 64-bit CUC stamp holds 8 unused bits, 32 bits of seconds and 24 bits of fraction
        secs = int.from_bytes(raw[37:41], 'big')
        frac = int.from_bytes(raw[41:44], 'big') / 2 ** 24
        recs.append((secs + frac, pos, vel))
    if not recs:
        raise ValueError('no complete position and velocity record in the sub-commutated ancillary data')
    recs.sort(key=lambda r: r[0])
    t = np.array([r[0] for r in recs])
    keep = np.r_[True, np.diff(t) > 1e-6]
    return t[keep], np.stack([r[1] for r in recs])[keep], np.stack([r[2] for r in recs])[keep]


def decode_user_data(data, hdr, sel):
    """The complex samples of the packets sel (indices into hdr): complex64 [len(sel), 2 max quads], shorter
    packets zero padded; and the number of packets whose bit stream ran out."""
    L = _decoder()
    buf = np.frombuffer(data, np.uint8)
    sel = np.asarray(sel)
    nq = hdr['quads'][sel]
    width = int(2 * nq.max())
    out = np.zeros((len(sel), width), np.complex64)
    bad = 0
    for i, p in enumerate(sel):
        o, n = int(hdr['user_offset'][p]), int(hdr['user_length'][p])
        q = int(nq[i])
        if q == 0 or n <= 0:
            continue
        chunk = np.ascontiguousarray(buf[o:o + n])
        row = np.zeros(2 * q, np.complex64)
        rc = L.decode_packet(chunk.ctypes.data, n, q, int(hdr['baq_mode'][p]), row.ctypes.data)
        bad += rc != 0
        out[i, :2 * q] = row
    return out, bad


def chirp(tx_start_frequency, tx_ramp_rate, tx_pulse_length, fs):
    """The baseband replica of the transmitted pulse at the sampling rate fs: exp(j 2 pi (f0 t + k t^2 / 2)) for
    0 <= t < TXPL."""
    L = int(round(tx_pulse_length * fs))
    t = np.arange(L) / fs
    return np.exp(2j * np.pi * (tx_start_frequency * t + 0.5 * tx_ramp_rate * t ** 2)).astype(np.complex64)
