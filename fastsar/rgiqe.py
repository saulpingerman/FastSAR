"""Radiometric quality model of a CPHD collection and the processing areas that follow from it.

The radar generalized image quality equation (RGIQE) predicts an RNIIRS rating from the information density of an
image, bandwidth_area * log2(1 + SNR), with bandwidth_area = |Row.ImpRespBW * Col.ImpRespBW * cos(SlopeAng)| in
cycles^2 / m^2 (the ground-plane area of the image's spatial-frequency support). The four functions rniirs,
information_density, snr_to_rniirs and rniirs_to_snr reproduce sarpy 2.1.1 (sarpy.processing.sicd.rgiqe), including
its linear branch at low information density.

Radiometry models the signal-to-noise ratio of a uniformly weighted backprojection image over the ground. With
N = pulses x samples phase-history values, a 1 m^2 point at the scene reference point has the per-sample power PTRef
and adds coherently to N^2 PTRef, while thermal noise of per-sample power PN adds to N PN, so the noise-equivalent
sigma zero is

    NESZ(x) = PN * bandwidth_area / (N * PTRef * G(x)),

G the two-way antenna gain at the ground point x relative to the reference point, averaged over the aperture as
(mean pulse amplitude gain)^2 and including the (R_ref / R)^4 spreading loss. The image is not corrected for the
antenna pattern: its impulse response and its noise level are the same everywhere and the signal fades toward the
edges, so RNIIRS varies over the image through the SNR alone. area_mask and area_bounds turn the model into the
region to process: the tasked area, the area at least as good as the tasked one, the area above a given RNIIRS, or
the area out to black.

Two reference signals. An RNIIRS is a rating against a fixed reference scene, the RGIQE's: a sigma zero of 0 dB for a
co-polarized channel and -6.02 dB for a cross-polarized one, sarpy's convention, so that an RNIIRS computed here is
the one sarpy's PREDICTED_RNIIRS states for the same image. The edge of the 'black' area is a statement about the
ground itself: where terrain of an ordinary brightness, black_sigma0_db (-10 dB by default), no longer stands above
the noise and the ambiguities, that is, where the noise-equivalent sigma zero plus the ambiguity level reaches
black_sigma0_db. The two therefore differ by the 10 dB between the references: 'black' ends where the SNR of the
RGIQE's reference scene is about 10 dB, not 0 dB.

Calibration. The same levels tie the image's pixel power to RCS and sigma zero: calibration gives the content of the
SICD Radiometric block (the noise level and the four scale factor polynomials) for an image of a collection whose
file states PTRef, and reference_levels says what a file states.
"""
import os

import numpy as np


__all__ = ['rniirs', 'information_density', 'snr_to_rniirs', 'rniirs_to_snr', 'Radiometry', 'area_mask', 'area_bounds',
           'reference_levels', 'support_area', 'calibration']

C = 299792458.0
RNIIRS_FIT = (3.4761, 0.4357)          # RNIIRS = a0 + a1 log2(information density), sarpy's fit of 2022-02-01
_FLOOR_DB = 15.0                       # one-way depth below the peak to which an antenna gain polynomial is trusted
_NOISE_BLOCKS, _NOISE_BLOCK_PULSES = 4, 128
_CELL_SAMPLES = 1000                   # least number of power samples averaged into a cell of the range-Doppler map


def _transition():
    a0, a1 = RNIIRS_FIT
    t = np.exp(1 - np.log(2) * a0 / a1)
    return t, a1 / (t * np.log(2))


def rniirs(information_density):
    """RNIIRS of an information density (cycles^2 / m^2 times bits): a0 + a1 log2(d) above the transition density
    exp(1 - ln(2) a0 / a1), and the tangent line through the origin below it, which keeps the rating positive."""
    a0, a1 = RNIIRS_FIT
    t, slope = _transition()
    d = np.asarray(information_density, np.float64)
    out = np.where(d > t, a0 + a1 * np.log2(np.where(d > t, d, t)), slope * d)
    return float(out) if out.ndim == 0 else out


def information_density(rniirs):
    """The information density that gives an RNIIRS: the inverse of rniirs()."""
    a0, a1 = RNIIRS_FIT
    t, slope = _transition()
    r = np.asarray(rniirs, np.float64)
    out = np.where(r > slope * t, np.exp2((r - a0) / a1), r / slope)
    return float(out) if out.ndim == 0 else out


def snr_to_rniirs(bandwidth_area, snr):
    """RNIIRS of an image with the bandwidth area (cycles^2 / m^2) and the signal-to-noise power ratio snr."""
    return rniirs(np.asarray(bandwidth_area, np.float64) * np.log2(1 + np.asarray(snr, np.float64)))


def rniirs_to_snr(bandwidth_area, rniirs):
    """The signal-to-noise power ratio an image of the bandwidth area needs for an RNIIRS: the inverse of
    snr_to_rniirs()."""
    with np.errstate(over='ignore'):                                # an RNIIRS out of reach needs an infinite ratio
        out = np.exp2(np.asarray(information_density(rniirs), np.float64) / np.asarray(bandwidth_area, np.float64)) - 1
    return float(out) if out.ndim == 0 else out


def _find(seq, key):
    return next((a for a in (seq or []) if str(a.Identifier) == str(key)), None)


def _pvp(pv, name):
    """A per-vector parameter as a float array [P] or [P, n]."""
    a = pv[name]
    if a.dtype.names:
        a = np.stack([a[n] for n in a.dtype.names], -1)
    return np.asarray(a, np.float64)


def _poly1(p, t):
    return np.polynomial.polynomial.polyval(t, np.atleast_1d(np.asarray(p.get_array(), np.float64)))


def _xyz_poly(p, t):
    return np.stack([_poly1(getattr(p, c), t) for c in 'XYZ'], -1)


class _Pattern:
    """A one-way antenna gain (dB, up to a constant) over direction cosines, held as a table read bilinearly, with
    its edge values beyond the table. A gain polynomial is trusted from its peak outward until it has fallen
    _FLOOR_DB or turns upward again, and keeps its value at that boundary beyond it (polynomials fitted to a main
    lobe diverge outside the lobe)."""

    def __init__(self, table, x0, y0, xss, yss, clamp=None):
        self.table, self.clamp = np.ascontiguousarray(table, np.float64), clamp
        self.x0, self.y0, self.xss, self.yss = float(x0), float(y0), float(xss), float(yss)

    @classmethod
    def from_poly(cls, coef, n=513):
        c = np.atleast_2d(np.asarray(coef, np.float64))
        if c.size == 1 or not np.any(c.ravel()[1:]):
            return None                                             # a constant: no pattern
        pv2 = np.polynomial.polynomial.polyval2d
        phi = np.linspace(-np.pi, np.pi, 721)
        r = np.r_[0.0, np.logspace(-6, 0, 1201)]
        g = pv2(np.outer(np.cos(phi), r), np.outer(np.sin(phi), r), c)
        drop = np.maximum.accumulate(g, axis=1) - g
        rows, low = np.arange(len(phi)), drop >= _FLOOR_DB
        j = np.maximum(low.argmax(1), 1)                            # the floor, at the radius where it is crossed
        d0, d1 = drop[rows, j - 1], drop[rows, j]
        rt = np.where(low.any(1), r[j - 1] + (r[j] - r[j - 1]) * (_FLOOR_DB - d0) / np.maximum(d1 - d0, 1e-300), 1.0)
        up = np.zeros(g.shape, bool)
        up[:, 1:] = (g[:, 1:] > g[:, :-1] + 1e-9) & (drop[:, :-1] > 1e-9)       # rising again after a fall
        rt = np.where(up.any(1), np.minimum(rt, r[np.maximum(up.argmax(1) - 1, 0)]), rt)
        xm, ym = 1.02 * float((rt * np.abs(np.cos(phi))).max()), 1.02 * float((rt * np.abs(np.sin(phi))).max())
        x, y = np.linspace(-xm, xm, n), np.linspace(-ym, ym, n)
        X, Y = np.meshgrid(x, y, indexing='ij')
        rr = np.hypot(X, Y)
        lim = np.interp(np.arctan2(Y, X), phi, rt)
        s = np.minimum(1.0, lim / np.maximum(rr, 1e-30))
        return cls(pv2(X * s, Y * s, c), -xm, -ym, x[1] - x[0], y[1] - y[0], clamp=rr > lim)

    @classmethod
    def from_array(cls, gain, x0, y0, xss, yss):
        g = np.array(gain, np.float64)
        if g.ndim != 2 or min(g.shape) < 2 or np.ptp(g[np.isfinite(g)]) == 0:
            return None
        g[~np.isfinite(g)] = g[np.isfinite(g)].min()
        return cls(g, x0, y0, xss, yss)

    def __call__(self, dx, dy, clamped=False):
        from scipy.ndimage import map_coordinates
        ij = np.stack([(dx - self.x0) / self.xss, (dy - self.y0) / self.yss])
        g = map_coordinates(self.table, ij, order=1, mode='nearest')
        if not clamped:
            return g
        n1, n2 = self.table.shape
        out = (ij[0] < 0) | (ij[0] > n1 - 1) | (ij[1] < 0) | (ij[1] > n2 - 1)
        if self.clamp is not None:
            k = np.clip(np.rint(ij), 0, [[n1 - 1], [n2 - 1]]).astype(np.intp)
            out |= self.clamp[k[0], k[1]]
        return g, out


def _parameters(m, meta, ch):
    """The Channel.Parameters entry of the channel that meta describes (by its identifier, else by index), or None."""
    pars = getattr(getattr(m, 'Channel', None), 'Parameters', None)
    par = _find(pars, meta.get('channel')) if meta.get('channel') is not None else None
    if par is None:
        par = pars[ch] if pars is not None and len(pars) > ch else None
    return par


def reference_levels(cphd_path, meta, channel=0):
    """(PTRef, PNRef / BNRef) of the channel of a CPHD file that meta (read_cphd's) describes: the per-sample power
    of a 1 m^2 point target at the scene reference point, and the noise power per sample. None for a level the file
    does not state. A file with PTRef supports the radiometric calibration of its image (calibration); with PNRef
    as well, its noise level needs no measurement."""
    if hasattr(cphd_path, 'cphd_meta'):
        r = cphd_path
    else:
        from .io import _open_cphd
        r = _open_cphd(str(cphd_path))
    par = _parameters(r.cphd_meta, meta, int(meta.get('channel_index', channel)))
    pt = getattr(getattr(par, 'TgtRefLevel', None), 'PTRef', None)
    nl = getattr(par, 'NoiseLevel', None)
    pn = getattr(nl, 'PNRef', None)
    return (None if pt is None else float(pt),
            None if pn is None else float(pn) / float(getattr(nl, 'BNRef', None) or 1.0))


def _sicd_meta(sicd):
    if isinstance(sicd, (str, bytes, os.PathLike)):
        from . import _sicdxml
        return _sicdxml.read(sicd)
    s = getattr(sicd, 'sicd_meta', sicd)                 # a reader's metadata, or the metadata themselves
    return s[0] if isinstance(s, (tuple, list)) else s


def _sicd_levels(sicd, notes):
    """(NESZ at the scene center point as a power ratio or None, PREDICTED_RNIIRS or None) of a SICD."""
    s = _sicd_meta(sicd)
    nesz, pred = None, None
    rad = getattr(s, 'Radiometric', None)
    nl = getattr(rad, 'NoiseLevel', None)
    sf = getattr(rad, 'SigmaZeroSFPoly', None)
    if nl is None or sf is None or getattr(nl, 'NoisePoly', None) is None:
        notes.append('The SICD has no Radiometric block with a noise level and a sigma zero scale factor.')
    elif str(nl.NoiseLevelType).upper() != 'ABSOLUTE':
        notes.append('The SICD noise level is not of type ABSOLUTE and was not used.')
    else:
        nesz = float(10.0 ** (0.1 * float(nl.NoisePoly[0, 0])) * float(sf[0, 0]))
        if not 1e-6 <= nesz <= 10.0:
            notes.append(f'The SICD Radiometric block gives a noise-equivalent sigma zero of {10 * np.log10(nesz):.1f} dB '
                         'at the scene center point, outside the plausible range of -60 to +10 dB, and was not used.')
            nesz = None
    par = getattr(getattr(s, 'CollectionInfo', None), 'Parameters', None)
    try:
        v = None if par is None else par.get('PREDICTED_RNIIRS', None)
        pred = None if v is None else float(v)
    except (TypeError, ValueError, AttributeError):
        pred = None
    return nesz, pred


class Radiometry:
    """Signal-to-noise model of the image a CPHD collection forms.

    cphd_path: the CPHD file (or an open reader of it: fastsar's own or sarpy's). meta: the dict read_cphd or
    read_collection returned for the same channel; ground points are given in its local frame (io.ecf_to_local,
    io.local_to_ecf). pulses: a slice (or (first, last + 1)) of meta's pulses when the image uses only some of them.
    channel: the channel index when meta does not carry one.

    The noise level at the scene reference point comes from the first of: nesz_db (dB, given by the caller); sicd, a
    SICD of the same collection (path, reader or metadata) with a Radiometric block; the file's PTRef with PNRef;
    PTRef with the noise measured from the data; and, for a file with no reference level, the clutter-to-noise ratio
    measured from the data together with an assumed mean scene sigma zero, scene_sigma0_db. level_source names the
    one used ('argument', 'sicd', 'cphd', 'cphd+measured', 'uncalibrated') and notes lists what was done in plain
    sentences. sigma0_db is the reference signal of the SNR; the default follows sarpy's RGIQE, 0 dB for a
    co-polarized channel and -6.02 dB for a cross-polarized one.

    The antenna gain is evaluated on at most eval_pulses pulses spread over the aperture (fewer for large point
    sets, never under 8), from the gain arrays of the file when it has them and from its gain polynomials otherwise.
    """

    def __init__(self, cphd_path, meta, *, channel=0, pulses=None, nesz_db=None, sicd=None, sigma0_db=None,
                 scene_sigma0_db=-10.0, eval_pulses=64):
        if hasattr(cphd_path, 'cphd_meta'):
            r = cphd_path
        else:
            from .io import _open_cphd
            r = _open_cphd(str(cphd_path))
        m = r.cphd_meta
        self.notes, self.meta = [], meta
        ch = int(meta.get('channel_index', channel))
        lo, hi = meta.get('pulses', (0, len(meta['tx'])))
        if pulses is None:
            pulses = slice(None)
        elif not isinstance(pulses, slice):
            pulses = slice(*pulses)
        fsel = np.arange(lo, hi)[pulses]                            # the file's pulse numbers the image uses
        tx, rcv = np.asarray(meta['tx'], np.float64)[pulses], np.asarray(meta['rcv'], np.float64)[pulses]
        srp = np.asarray(meta['srp_pulses'], np.float64)[pulses]
        t_tx = np.asarray(meta['tx_time'], np.float64)[pulses]
        t_rcv = np.asarray(meta['rcv_time'], np.float64)[pulses] if 'rcv_time' in meta else t_tx
        P = len(fsel)
        if P < 2:
            raise ValueError('the radiometric model needs at least two pulses')
        self.pulses_used = P
        self.ref_point = (np.asarray(meta['srp'], np.float64) - meta['origin']) @ np.asarray(meta['R']).T
        par = _parameters(m, meta, ch)
        pv = r.read_pvp_array(ch)
        names = pv.dtype.names
        v1 = par is not None and 'TOA1' in names                    # a CPHD 1.x channel
        fc = getattr(par, 'FxC', None)
        if fc is None:
            f1 = 'FX1' if 'FX1' in names else 'Fx1'
            fc = 0.5 * float(np.mean(_pvp(pv, f1)[fsel]) + np.mean(_pvp(pv, f1[:2] + '2')[fsel]))
        self.wavelength = C / float(fc)
        self.prf = (P - 1) / float(t_tx[-1] - t_tx[0]) if t_tx[-1] > t_tx[0] else None
        # mid-aperture geometry: the antenna phase center, its velocity, the ground range and cross range directions
        apc = 0.5 * (tx + rcv)
        mid = P // 2
        self._apc = apc[mid]
        self._vel = np.gradient(apc, t_tx, axis=0)[mid] if self.prf else np.zeros(3)
        los = self.ref_point - self._apc
        rg = np.array([los[0], los[1], 0.0]) / max(np.hypot(los[0], los[1]), 1e-9)
        self.range_dir, self.cross_dir = rg, np.array([-rg[1], rg[0], 0.0])
        self.graze = float(np.arcsin(-los[2] / np.linalg.norm(los)))
        # the reference signal: sarpy's default, 1 for a co-polarized and 0.25 for a cross-polarized channel
        pol = getattr(par, 'Polarization', None)
        txp, rcp = getattr(pol, 'TxPol', None), getattr(pol, 'RcvPol', None)
        if txp is None or rcp is None:
            p2 = str(meta.get('polarization') or '')
            txp, rcp = (p2[0], p2[1]) if len(p2) == 2 else (None, None)
        cross = txp is not None and str(txp) != str(rcp)
        self.sigma0_db = float(sigma0_db) if sigma0_db is not None else (-6.02 if cross else 0.0)
        self._sigma_ref = 10.0 ** (0.1 * self.sigma0_db) if sigma0_db is not None else (0.25 if cross else 1.0)
        # the pulses the gain is evaluated on, with their geometry and delay windows
        self.eval_pulses = int(max(1, eval_pulses))
        idx = np.unique(np.rint(np.linspace(0, P - 1, min(self.eval_pulses, P))).astype(int))
        fi = fsel[idx]
        e = dict(tx=tx[idx], rcv=rcv[idx], srp=srp[idx])
        e['rt'], e['rr'] = np.linalg.norm(srp[idx] - tx[idx], axis=1), np.linalg.norm(srp[idx] - rcv[idx], axis=1)
        e['mono'] = bool(np.abs(tx[idx] - rcv[idx]).max() <= 1e-9 * e['rt'].min())
        inf = np.full(len(idx), np.inf)
        e['toa'] = (_pvp(pv, 'TOA1')[fi], _pvp(pv, 'TOA2')[fi]) if v1 else (-inf, inf)
        self.has_extended = v1 and 'TOAE1' in names and 'TOAE2' in names
        e['toae'] = (_pvp(pv, 'TOAE1')[fi], _pvp(pv, 'TOAE2')[fi]) if self.has_extended else e['toa']
        if not v1:
            self.notes.append('The file gives no delay window per pulse (TOA1, TOA2), so the range swath is not limited.')
        self._e = e
        self._ant = self._antenna(r, m, par, pv, fi, t_tx[idx], t_rcv[idx]) if v1 else None
        self.has_antenna = self._ant is not None
        if self._ant is None:
            self.notes.append('The file has no antenna description that could be used, so the antenna gain is taken '
                              'as 0 dB everywhere and only the spreading loss shapes the signal.')
        e['gref'] = np.zeros(len(idx))
        if self._ant is not None:
            for i in range(len(idx)):
                e['gref'][i] = sum(float(self._oneway(k, i, e['srp'][i][None], np.array([e['rt' if k == 'tx' else 'rr'][i]]))[0][0])
                                   for k in ('tx', 'rcv'))
        # the noise level
        self.noise_per_sample = self.clutter_per_sample = self.area_eff = self.measurement = None
        self.pt_ref = self.sicd_rniirs = self._nesz_fixed = None
        self.scene_sigma0_db = float(scene_sigma0_db)
        tgt, nl = getattr(par, 'TgtRefLevel', None), getattr(par, 'NoiseLevel', None)
        pt = getattr(tgt, 'PTRef', None)
        sicd_nesz = None
        if sicd is not None:
            sicd_nesz, self.sicd_rniirs = _sicd_levels(sicd, self.notes)
        if nesz_db is not None:
            self.level_source, self._nesz_fixed = 'argument', 10.0 ** (0.1 * float(nesz_db))
            self.notes.append(f'The noise-equivalent sigma zero at the reference point is the given {float(nesz_db):.2f} dB.')
        elif sicd_nesz is not None:
            self.level_source, self._nesz_fixed = 'sicd', sicd_nesz
            self.notes.append(f'The noise-equivalent sigma zero at the reference point, {10 * np.log10(sicd_nesz):.2f} dB, '
                              'is the noise level of the SICD at its scene center point.')
        elif pt is not None and getattr(nl, 'PNRef', None) is not None:
            bn = float(getattr(nl, 'BNRef', None) or 1.0)
            self.level_source, self.pt_ref, self.noise_per_sample = 'cphd', float(pt), float(nl.PNRef) / bn
            self.notes.append('The noise level is PNRef / BNRef of the file (the noise power per sample at the delay '
                              'of the scene) and the signal level is its PTRef.')
        else:
            if str(getattr(getattr(m, 'Global', None), 'DomainType', 'FX')) != 'FX' or not v1:
                raise ValueError('the noise level can be measured only from a frequency-domain CPHD 1.x file; '
                                 'give nesz_db or a SICD with a Radiometric block')
            K = int(m.Data.Channels[ch].NumSamples)
            self.measurement = _measure_noise(r, ch, pv, fsel, K)
            self.noise_per_sample, self.clutter_per_sample = self.measurement['noise'], self.measurement['clutter']
            self.notes.append(self.measurement['note'])
            if self._ant is not None and self.clutter_per_sample > 0:
                self.area_eff = self._effective_area()
            if pt is not None:
                self.level_source, self.pt_ref = 'cphd+measured', float(pt)
                self.notes.append('The signal level is PTRef of the file; it has no PNRef, so the noise power per '
                                  'sample was measured from the data.')
                if self.area_eff:
                    s0 = 10 * np.log10(self.clutter_per_sample / (self.pt_ref * self.area_eff))
                    self.notes.append(f'The echo power implies a mean scene sigma zero of {s0:.1f} dB.')
            else:
                if not self.area_eff:
                    raise ValueError('the file has no reference level, and the clutter level cannot be turned into a '
                                     'sigma zero without an antenna pattern and measurable clutter; give nesz_db')
                self.level_source = 'uncalibrated'
                self.notes.append('The file is not radiometrically calibrated (no PTRef). The noise-equivalent sigma '
                                  'zero is the assumed mean scene sigma zero of '
                                  f'{self.scene_sigma0_db:.1f} dB divided by the clutter-to-noise ratio measured '
                                  'from the data, and is only as good as that assumption.')

    def _antenna(self, r, m, par, pv, fi, t_tx, t_rcv):
        """The transmit and receive patterns and their per-pulse axes (local frame) and steering, or None."""
        ant, pa = getattr(m, 'Antenna', None), getattr(par, 'Antenna', None)
        if ant is None or pa is None:
            return None
        names, R = pv.dtype.names, np.asarray(self.meta['R'], np.float64)
        sup = getattr(getattr(m, 'SupportArray', None), 'AntGainPhase', None)
        out, src = {}, set()
        for k, K, t in (('tx', 'Tx', t_tx), ('rcv', 'Rcv', t_rcv)):
            pat = _find(ant.AntPattern, getattr(pa, K + 'APATId', None))
            apc = _find(ant.AntPhaseCenter, getattr(pa, K + 'APCId', None))
            acf = _find(ant.AntCoordFrame, getattr(apc, 'ACFId', None))
            pvp_axes = K + 'ACX' in names and K + 'ACY' in names
            if pat is None or (acf is None and not pvp_axes):
                return None
            if pvp_axes:
                ax, ay = _pvp(pv, K + 'ACX')[fi], _pvp(pv, K + 'ACY')[fi]
                src.add('per-pulse antenna axes')
            else:
                ax, ay = _xyz_poly(acf.XAxisPoly, t), _xyz_poly(acf.YAxisPoly, t)
                src.add('antenna axis polynomials')
            ax, ay = ax @ R.T, ay @ R.T
            ax, ay = ax / np.linalg.norm(ax, axis=1, keepdims=True), ay / np.linalg.norm(ay, axis=1, keepdims=True)
            ebp = getattr(pat, 'EB', None)
            if K + 'EB' in names:
                eb = _pvp(pv, K + 'EB')[fi].reshape(len(fi), 2)
            elif ebp is not None and getattr(ebp, 'DCXPoly', None) is not None:
                eb = np.stack([_poly1(ebp.DCXPoly, t), _poly1(ebp.DCYPoly, t)], -1) * np.ones((len(fi), 1))
            else:
                eb = np.zeros((len(fi), 2))
            parts = {}
            for part, tags in (('array', ('Array',)), ('elem', ('Elem', 'Element'))):
                gp = next((g for g in (getattr(pat, tag, None) for tag in tags) if g is not None), None)
                p, gid = None, getattr(gp, 'AntGPId', None)
                sa = _find(sup, gid) if gid else None
                if sa is not None:
                    try:
                        a = r.read_support_array(gid)
                        p = _Pattern.from_array(a['Gain'] if a.dtype.names else a[..., 0], sa.X0, sa.Y0, sa.XSS, sa.YSS)
                    except Exception:                              # an unreadable array: fall back on the polynomial
                        p = None
                    if p is not None:
                        src.add('gain arrays')
                if p is None and getattr(gp, 'GainPoly', None) is not None:
                    p = _Pattern.from_poly(gp.GainPoly.get_array())
                    if p is not None:
                        src.add('gain polynomials (held at their value 15 dB one-way below the peak beyond it)')
                parts[part] = p
            out[k] = dict(parts, ax=ax, ay=ay, eb=eb)
        if all(out[k][p] is None for k in out for p in ('array', 'elem')):
            return None
        same = lambda a, b: (a is None and b is None) or (a is not None and b is not None and a.table.shape == b.table.shape
                                                           and np.array_equal(a.table, b.table)
                                                           and (a.x0, a.y0, a.xss, a.yss) == (b.x0, b.y0, b.xss, b.yss))
        out['same'] = bool(self._e['mono'] and all(same(out['tx'][p], out['rcv'][p]) for p in ('array', 'elem'))
                           and all(np.allclose(out['tx'][q], out['rcv'][q], rtol=0, atol=1e-9) for q in ('ax', 'ay', 'eb')))
        self.notes.append('The antenna gain comes from the ' + ' and '.join(sorted(src)) + ' of the file.')
        return out

    def _oneway(self, k, i, X, rng, clamped=False):
        """One-way gain (dB) of the transmit ('tx') or receive ('rcv') antenna at evaluation pulse i toward the
        points X [n, 3] at the ranges rng [n] from it, and where it lies beyond the trusted pattern.

        CPHD 1.x (and SICD) convention: the direction cosines DCX, DCY are the components along the antenna X and Y
        axes of the unit vector from the phase center to the point, the mechanical boresight is X x Y (toward the
        scene), the array pattern is a function of (DCX - EB_DCX, DCY - EB_DCY) with EB the direction cosines of the
        electrical boresight, and the element pattern a function of (DCX, DCY). A gain array holds DCX along its
        rows (X0 + row XSS) and DCY along its columns."""
        a, pos = self._ant[k], self._e[k][i]
        dx, dy = (X @ a['ax'][i] - pos @ a['ax'][i]) / rng, (X @ a['ay'][i] - pos @ a['ay'][i]) / rng
        g, cl = 0.0, np.zeros(len(X), bool) if clamped else None
        for p, sx, sy in ((a['array'], a['eb'][i, 0], a['eb'][i, 1]), (a['elem'], 0.0, 0.0)):
            if p is not None:
                v = p(dx - sx, dy - sy, clamped)
                if clamped:
                    cl |= v[1]
                g = g + (v[0] if clamped else v)
        return g, cl

    def _evaluate(self, points, clamp=False):
        """Two-way gain relative to the reference point at points [..., 3]: 'gain' the square of the pulse-mean
        amplitude gain, 'power' the pulse-mean power gain, 'margin' and 'margin_ext' the least time (s) by which
        the echo lies inside the full-bandwidth and the extended delay window over the pulses evaluated (negative
        outside), 'clamped' where the pattern is not known (clamp=True)."""
        pts = np.asarray(points, np.float64)
        X = pts.reshape(-1, 3)
        ok = np.isfinite(X).all(1)
        if not ok.all():
            X = np.where(ok[:, None], X, self.ref_point)
        n, e = len(X), self._e
        m = len(e['rt'])
        take = np.unique(np.rint(np.linspace(0, m - 1, int(np.clip(4e6 / max(n, 1), min(8, m), m)))).astype(int))
        x2 = np.einsum('ij,ij->i', X, X)
        amp, pw, cl = np.zeros(n), np.zeros(n), np.zeros(n, bool)
        mg, mge = np.full(n, 1.0), np.full(n, 1.0)
        for i in take:
            rt = np.sqrt(np.maximum(x2 - 2 * (X @ e['tx'][i]) + e['tx'][i] @ e['tx'][i], 1e-12))
            rr = rt if e['mono'] else np.sqrt(np.maximum(x2 - 2 * (X @ e['rcv'][i]) + e['rcv'][i] @ e['rcv'][i], 1e-12))
            a = (e['rt'][i] * e['rr'][i]) / (rt * rr)
            if self._ant is not None:
                want = clamp and i == take[len(take) // 2]
                g, c1 = self._oneway('tx', i, X, rt, want)
                g2, c2 = (g, c1) if self._ant['same'] else self._oneway('rcv', i, X, rr, want)
                a *= 10.0 ** (0.05 * (g + g2 - e['gref'][i]))
                if want:
                    cl = c1 | c2
            amp += a
            pw += a * a
            dt = (rt + rr - e['rt'][i] - e['rr'][i]) / C
            mg = np.minimum(mg, np.minimum(dt - e['toa'][0][i], e['toa'][1][i] - dt))
            mge = np.minimum(mge, np.minimum(dt - e['toae'][0][i], e['toae'][1][i] - dt))
        shape = pts.shape[:-1]
        gain, power = np.where(ok, (amp / len(take)) ** 2, 0.0), np.where(ok, pw / len(take), 0.0)
        mg, mge = np.where(ok, mg, -1.0), np.where(ok, mge, -1.0)
        return dict(gain=gain.reshape(shape), power=power.reshape(shape), margin=mg.reshape(shape),
                    margin_ext=mge.reshape(shape), clamped=cl.reshape(shape))

    def gain_db(self, points):
        """Two-way antenna gain (dB) at the ground points [..., 3] (local frame) relative to the scene reference
        point: the square of the amplitude gain averaged over the pulses, with the (R_ref / R)^4 spreading loss."""
        return 10 * np.log10(np.maximum(self._evaluate(points)['gain'], 1e-30))

    def _nesz_ref(self, bandwidth_area, samples):
        if self._nesz_fixed is not None:
            return self._nesz_fixed
        k = float(bandwidth_area) / float(samples)
        if self.pt_ref is not None:
            return self.noise_per_sample * k / self.pt_ref
        return 10.0 ** (0.1 * self.scene_sigma0_db) * k * self.area_eff * self.noise_per_sample / self.clutter_per_sample

    def nesz_db(self, points, bandwidth_area, samples):
        """Noise-equivalent sigma zero (dB) at the ground points of the uniformly weighted image formed from
        `samples` phase-history values (pulses x samples per pulse) with the given bandwidth area."""
        return 10 * np.log10(self._nesz_ref(bandwidth_area, samples)) - self.gain_db(points)

    def cnr_db(self, bandwidth_area, samples):
        """Clutter-to-noise ratio (dB) the image has at the reference point for the mean clutter level measured
        from the data, or None when the noise was not measured or there is no antenna pattern."""
        if self.measurement is None or not self.area_eff or self.clutter_per_sample <= 0:
            return None
        return float(10 * np.log10(self.clutter_per_sample * float(samples)
                                   / (self.area_eff * float(bandwidth_area) * self.noise_per_sample)))

    def _ambiguity(self, points, k):
        """The ground points at the range and height of `points` whose Doppler frequency at mid-aperture differs by
        k times the pulse repetition frequency, on the same side of the track; NaN where there is none."""
        X = np.asarray(points, np.float64)
        d = X - self._apc
        rng, rho = np.linalg.norm(d, axis=-1), np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-9)
        vh, pv = max(np.hypot(self._vel[0], self._vel[1]), 1e-12), np.arctan2(self._vel[1], self._vel[0])
        psi = np.arctan2(d[..., 1], d[..., 0]) - pv
        c = np.cos(psi) + k * self.wavelength * self.prf * rng / (2 * rho * vh)
        pk = np.where(np.sin(psi) < 0, -1.0, 1.0) * np.arccos(np.clip(c, -1, 1)) + pv
        out = np.stack([self._apc[0] + rho * np.cos(pk), self._apc[1] + rho * np.sin(pk), X[..., 2]], -1)
        out[np.abs(c) > 1] = np.nan
        return out

    def _fields(self, points, bandwidth_area, samples, ambiguities=True, sigma0_db=None):
        """The SNR before the delay window is applied, the two window margins and the unknown-pattern flag.
        sigma0_db: the scene's sigma zero in place of the reference one."""
        ev = self._evaluate(points, clamp=True)
        sigma = self._sigma_ref if sigma0_db is None else 10.0 ** (0.1 * float(sigma0_db))
        den = self._nesz_ref(bandwidth_area, samples) / sigma
        if ambiguities and self.prf:
            for k in (-2, -1, 1, 2):
                den = den + self._evaluate(self._ambiguity(points, k))['power']
        return ev['gain'] / den, ev['margin'], ev['margin_ext'], ev['clamped']

    def snr(self, points, bandwidth_area, samples, ambiguities=True, *, extended=False, sigma0_db=None):
        """Signal-to-noise power ratio at the ground points for a scene of the reference sigma zero (self.sigma0_db),
        or of sigma0_db (dB) when given. With ambiguities, the echoes of the first two azimuth ambiguities on either
        side (a uniform scene of the same sigma zero is assumed) count as noise, which makes the ratio one of signal
        to noise and clutter: SNR = G(x) / (NESZ_ref / sigma + sum_k G(x_k)). It is at least 1 exactly where the
        noise-equivalent sigma zero plus the ambiguity level, NESZ(x) + sigma sum_k G(x_k) / G(x), is at most sigma.
        Points outside the full-bandwidth delay window TOA1..TOA2 of the pulses get 0; extended=True admits the
        extended window TOAE1..TOAE2 instead."""
        snr, mg, mge, _ = self._fields(points, bandwidth_area, samples, ambiguities, sigma0_db)
        return np.where((mge if extended else mg) >= 0, snr, 0.0)

    def rniirs(self, points, bandwidth_area, samples, ambiguities=True, *, extended=False):
        """RNIIRS the RGIQE predicts at the ground points: snr_to_rniirs(bandwidth_area, snr(...))."""
        return snr_to_rniirs(bandwidth_area, self.snr(points, bandwidth_area, samples, ambiguities, extended=extended))

    def _effective_area(self):
        """Gain-weighted ground area (m^2) inside the full-bandwidth delay window: the area whose echoes, at the
        gain of the reference point, would give the clutter power of one pulse."""
        e = self._e
        mid = len(e['rt']) // 2
        t1, t2 = float(e['toa'][0][mid]), float(e['toa'][1][mid])
        if not np.isfinite(t1 + t2) or t2 <= t1:
            return None
        cg = max(np.cos(self.graze), 0.05)
        g1, g2 = 0.5 * C * t1 / cg, 0.5 * C * t2 / cg
        w = g2 - g1
        rg = np.linspace(g1 - 0.25 * w, g2 + 0.25 * w, 96)
        dr, dc = rg[1] - rg[0], w / 64
        half, last = w, None
        while True:
            cr = np.arange(-half, half + 0.5 * dc, dc)
            pts = self.ref_point + rg[:, None, None] * self.range_dir + cr[None, :, None] * self.cross_dir
            ev = self._evaluate(pts, clamp=True)
            area = float(ev['power'][(ev['margin'] >= 0) & ~ev['clamped']].sum() * dr * dc)
            if (last is not None and area - last <= 0.01 * area) or half >= 16 * w:
                return area
            last, half = area, 2 * half


def _phase_sign(m):
    """The phase sign of the file's samples as fastsar.io reads them: Global.SGN, except for Capella collections,
    which declare +1 while their phase follows -1 (see io.read_cphd). -1: a later echo has the phase
    exp(-j 2 pi f delay)."""
    try:
        sgn = int(m.Global.SGN)
    except (AttributeError, TypeError, ValueError):
        return -1
    collector = str(getattr(getattr(m, 'CollectionID', None), 'CollectorName', '') or '')
    return -1 if collector.lower().startswith('capella') else sgn


def _line_fit(tau, prof, lo, hi):
    """Robust straight line through the range profile between the delays lo and hi: medians of 3 to 24 groups of
    bins, least squares, and one pass that drops groups standing well above the line (echoes). None when the
    interval holds too few bins."""
    sel = np.flatnonzero((tau >= lo) & (tau <= hi))
    if len(sel) < 48:
        return None
    sel = sel[np.argsort(tau[sel])]
    groups = np.array_split(sel, int(np.clip(len(sel) // 64, 3, 24)))
    x, y = np.array([tau[g].mean() for g in groups]), np.array([np.median(prof[g]) for g in groups])
    c = np.polyfit(x, y, 1)
    if len(x) >= 6:
        res = y - np.polyval(c, x)
        keep = res <= 2.5 * 1.4826 * np.median(np.abs(res - np.median(res))) + np.median(res)
        if 4 <= keep.sum() < len(x):
            c = np.polyfit(x[keep], y[keep], 1)
    return c


def _measure_noise(r, ch, pv, fsel, K):
    """Noise and clutter power per phase-history sample of the pulses fsel, from a few blocks of consecutive pulses.

    Each block is compressed in range (a unitary transform over the K samples, so white noise keeps its per-sample
    power in every delay bin) and the powers are averaged into a range profile. Thermal noise is taken, where the
    file saves an extended delay swath well beyond the scene (TOAE1, TOAE2), from straight lines fitted to the
    profile on either side of the scene and continued under it: after compression against a pulse of finite length
    the noise floor falls linearly toward the ends of the saved swath. Otherwise, and as an upper bound in every
    case, it is the least cell of a coarse, smoothed range-Doppler power map of the scene swath, which sees the
    noise wherever the pulse repetition frequency exceeds the Doppler band of the beam. Clutter is the rest of the
    power in the scene swath TOA1..TOA2, expressed per sample."""
    from scipy import fft as sfft
    names = pv.dtype.names
    sgn = _phase_sign(r.cphd_meta)                                  # the delay axis mirrors with the wrong sign
    first, span = int(fsel[0]), int(fsel[-1]) - int(fsel[0]) + 1
    nb = int(min(_NOISE_BLOCK_PULSES, span))
    starts = np.unique(np.rint(np.linspace(first, first + span - nb, max(1, min(_NOISE_BLOCKS, span // nb)))).astype(int))
    rows = np.concatenate([np.arange(p, p + nb) for p in starts])
    scss = float(np.mean(_pvp(pv, 'SCSS')[rows]))
    tau = np.fft.fftfreq(K, scss)
    t1, t2 = float(np.mean(_pvp(pv, 'TOA1')[rows])), float(np.mean(_pvp(pv, 'TOA2')[rows]))
    sw = (tau >= t1) & (tau <= t2)
    if sw.sum() < 8:
        sw = np.ones(K, bool)
    nsw, total = int(sw.sum()), len(rows) * int(sw.sum())
    nd, nr = max(min(32, nb // 4), 1), max(min(16, nsw // 8), 1)
    while nd * nr * _CELL_SAMPLES > total and nr > 1:
        nr //= 2
    while nd * nr * _CELL_SAMPLES > total and nd > 4:
        nd //= 2
    w = np.hanning(nb + 2)[1:-1]
    prof, cells = np.zeros(K), np.zeros((nd, nr))
    for p0 in starts:
        s = np.asarray(r.read_chip((int(p0), int(p0) + nb), (0, K), index=ch)).reshape(nb, K)
        rc = sfft.ifft(s, axis=1, workers=-1) * np.sqrt(K) if sgn < 0 else sfft.fft(s, axis=1, workers=-1) / np.sqrt(K)
        del s
        prof += (rc.real.astype(np.float64) ** 2 + rc.imag.astype(np.float64) ** 2).sum(0)
        rd = sfft.fft(rc[:, sw] * w[:, None].astype(rc.real.dtype), axis=0, workers=-1)
        pw = (rd.real ** 2 + rd.imag ** 2) / float(np.sum(w * w))
        del rc, rd
        pw = pw[:nd * (nb // nd), :nr * (nsw // nr)].reshape(nd, nb // nd, nr, nsw // nr)
        cells += pw.mean((1, 3), dtype=np.float64)
        del pw
    prof /= len(rows)
    cells /= len(starts)
    if nd >= 6:
        cells = (np.roll(cells, 1, 0) + cells + np.roll(cells, -1, 0)) / 3
    rd_min = float(cells.min())
    model, method, ext = None, 'range-Doppler minimum', None
    if 'TOAE1' in names and 'TOAE2' in names:
        e1, e2 = float(np.mean(_pvp(pv, 'TOAE1')[rows])), float(np.mean(_pvp(pv, 'TOAE2')[rows]))
        e1, e2, width = max(e1, float(tau.min())), min(e2, float(tau.max())), t2 - t1
        lines = [c for c in (_line_fit(tau, prof, e1 + 0.15 * (t1 - e1), t1 - width) if t1 - width - e1 > 0.6 * width else None,
                             _line_fit(tau, prof, t2 + width, e2 - 0.15 * (e2 - t2)) if e2 - t2 - width > 0.6 * width else None)
                 if c is not None]
        if lines:
            m = np.min([np.polyval(c, tau[sw]) for c in lines], axis=0)
            ext = float(m.mean())
            if np.isfinite(ext) and 0 < ext <= 1.05 * rd_min and m.min() > 0:
                model, method = m, 'extended swath'
    if model is None:
        model = np.full(nsw, rd_min)
    noise = float(model.mean())
    clutter = float(np.maximum(prof[sw] - model, 0).sum() / K)
    how = ('straight lines fitted to the range-compressed power in the extended delay swath on either side of the '
           'scene and continued under it' if method == 'extended swath' else
           'the least cell of a smoothed %d x %d range-Doppler power map of the scene swath, an upper bound' % (nd, nr))
    note = (f'The noise power per sample ({noise:.4g}) was measured from {len(rows)} pulses in {len(starts)} blocks as {how}. '
            f'The clutter power per sample from the scene swath is {clutter:.4g}.')
    return dict(noise=noise, clutter=clutter, method=method, rd_min=rd_min, extended=ext, pulses=len(rows),
                swath=(t1, t2), profile_mean=float(prof.mean()), note=note)


def _ring(tasked):
    """The corners of the tasked area [4, 3] ordered around their centroid."""
    q = np.asarray(tasked, np.float64).reshape(-1, 3)
    d = q[:, :2] - q[:, :2].mean(0)
    return q[np.argsort(np.arctan2(d[:, 1], d[:, 0]))]


def _in_polygon(xy, poly):
    """Even-odd test of the points xy [..., 2] against the polygon poly [n, 2]."""
    x, y = xy[..., 0], xy[..., 1]
    inside = np.zeros(x.shape, bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, 0)):
        if y1 != y2:
            inside ^= ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    return inside


def _weights(nodes, i):
    """Left node and weight of the right node for linear interpolation at the indices i between sorted nodes."""
    if len(nodes) == 1:
        return np.zeros(len(i), np.intp), np.zeros(len(i))
    k = np.clip(np.searchsorted(nodes, i, side='right') - 1, 0, len(nodes) - 2)
    return k, (i - nodes[k]) / (nodes[k + 1] - nodes[k])


def area_mask(rad, points, kind, bandwidth_area, samples, tasked=None, tasked_rniirs=None, *, ambiguities=True,
              black_sigma0_db=-10.0):
    """The part of a ground grid to process, as a boolean mask [n1, n2], and a dict describing it.

    rad: a Radiometry. points: a regular ground grid [n1, n2, 3] in the local frame (the grid is needed, not a list
    of points: the area is a connected region of it). bandwidth_area, samples and ambiguities describe the image as
    in Radiometry.snr. tasked: the four corners [4, 3] of the tasked area (cphd._image_area). kind selects the area:

      'tasked'          the tasked polygon;
      'tasked_quality'  where the predicted RNIIRS is at least the tasked one: tasked_rniirs when given, else the
                        PREDICTED_RNIIRS parameter of the SICD given to rad, else the lowest RNIIRS the model
                        predicts inside the tasked polygon;
      a number          where the predicted RNIIRS is at least that number;
      'black'           where a scene of sigma zero black_sigma0_db (dB) has an SNR of at least 1, ambiguities
                        counted as noise: where the noise-equivalent sigma zero plus the ambiguity level is at most
                        black_sigma0_db. The range swath is opened to the extended delay window TOAE1..TOAE2 of
                        the file when it has one.

    The RNIIRS of the two middle kinds is the RGIQE's, for its own reference scene (rad.sigma0_db: 0 dB for a
    co-polarized channel, -6.02 dB for a cross-polarized one), so that it is the rating sarpy's PREDICTED_RNIIRS
    states; black_sigma0_db does not enter it. The black edge refers to the ground instead: with the default of
    -10 dB it lies where terrain of that brightness sinks into the noise and the ambiguities, which keeps most of
    what an image out to an SNR of 1 of the reference scene would show at less area.

    Apart from 'tasked', the mask is the union of the tasked polygon with the connected region that meets the
    criterion and contains the scene reference point or touches the polygon, so it never falls short of the tasked
    area. Points where the antenna pattern is not known do not meet any criterion. A lower RNIIRS threshold gives a
    larger area, and so does a lower black_sigma0_db. 'black' contains the area of every RNIIRS threshold that needs
    a reference SNR of at least rad.sigma0_db - black_sigma0_db (10 dB for a co-polarized channel at the default);
    a threshold that needs less, a low RNIIRS at a large bandwidth area or a tasked quality met at a low SNR, gives
    an area that reaches beyond 'black' across the beam. Grids of more than about 300,000 points are evaluated on
    a lattice of at most 257 nodes per axis and interpolated.

    The dict holds kind, criterion ('polygon', 'rniirs' or 'snr'), threshold, snr_threshold, black_sigma0_db (for
    'black', else None), tasked_rniirs and tasked_rniirs_source, center_rniirs, center_snr_db and nesz_ref_db (model
    values at the reference point, for the reference scene), level_source, border (whether the mask reaches the
    first and last row and the first and last column of the grid), notes and level.

    level is a float32 array of the mask's shape with level >= 0 exactly where the mask is set, for drawing the
    boundary more finely than the grid: interpolate it bilinearly and keep what is >= 0. Inside the mask it is the
    margin by which the criterion is met (0 where a point of the tasked polygon does not meet it), outside the
    margin by which it is missed, at most -0.001 (also for points that meet the criterion in a region not connected
    to the scene). The margin is the lesser of the predicted RNIIRS minus the threshold (for 'black' the SNR in dB
    of a scene of black_sigma0_db) and the time in microseconds by which the echo lies inside the delay window, so
    only its sign and its zero crossing mean the same thing everywhere. For 'tasked' it is +1 inside the polygon and
    -1 outside."""
    from scipy import ndimage
    pts = np.asarray(points, np.float64)
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError('points must be a ground grid of shape [n1, n2, 3]')
    n1, n2 = pts.shape[:2]
    ring = None if tasked is None else _ring(tasked)
    notes = []
    c_snr = float(rad.snr(rad.ref_point[None], bandwidth_area, samples, ambiguities)[0])
    info = dict(kind=kind, criterion='polygon', threshold=None, snr_threshold=None, black_sigma0_db=None,
                tasked_rniirs=None,
                tasked_rniirs_source=None, center_rniirs=float(snr_to_rniirs(bandwidth_area, c_snr)),
                center_snr_db=float(10 * np.log10(max(c_snr, 1e-30))),
                nesz_ref_db=float(10 * np.log10(rad._nesz_ref(bandwidth_area, samples))),
                level_source=rad.level_source, border=(False,) * 4, notes=notes)
    if tasked_rniirs is not None:
        info['tasked_rniirs'], info['tasked_rniirs_source'] = float(tasked_rniirs), 'argument'
    elif rad.sicd_rniirs is not None:
        info['tasked_rniirs'], info['tasked_rniirs_source'] = float(rad.sicd_rniirs), 'sicd'
    if isinstance(kind, str) and kind not in ('tasked', 'tasked_quality', 'black'):
        try:
            kind = float(kind)
        except ValueError:
            raise ValueError("kind must be 'tasked', 'tasked_quality', 'black' or an RNIIRS value") from None

    def polygon(rows=slice(None)):
        return _in_polygon(pts[rows, :, :2], ring[:, :2])

    def border(mask):
        return bool(mask[0].any()), bool(mask[-1].any()), bool(mask[:, 0].any()), bool(mask[:, -1].any())

    if isinstance(kind, str) and kind == 'tasked':
        if ring is None:
            raise ValueError("kind 'tasked' needs the tasked area")
        mask = polygon()
        info['border'], info['level'] = border(mask), np.where(mask, 1.0, -1.0).astype(np.float32)
        return mask, info
    extended, scene = False, None
    if isinstance(kind, str) and kind == 'black':
        if not np.isfinite(float(black_sigma0_db)):
            raise ValueError(f'black_sigma0_db must be a sigma zero in dB, got {black_sigma0_db!r}')
        info['criterion'], thr = 'snr', 1.0
        info['threshold'], info['black_sigma0_db'] = 1.0, float(black_sigma0_db)
        scene = float(black_sigma0_db)                              # the SNR of this scene, not of the RGIQE's
        extended = bool(rad.has_extended)
        if extended:
            notes.append('The range swath is opened to the extended delay window TOAE1..TOAE2 of the file, in which '
                         'the echoes do not have the full bandwidth everywhere, so the resolution degrades toward its ends.')
    else:
        if isinstance(kind, str):                                   # 'tasked_quality'
            if info['tasked_rniirs'] is None:
                if ring is None:
                    raise ValueError("kind 'tasked_quality' needs the tasked area, tasked_rniirs or a SICD that "
                                     'gives PREDICTED_RNIIRS')
                u = np.linspace(0, 1, 17)
                lat = ((1 - u)[:, None, None] * ((1 - u)[None, :, None] * ring[0] + u[None, :, None] * ring[1])
                       + u[:, None, None] * ((1 - u)[None, :, None] * ring[3] + u[None, :, None] * ring[2]))
                s_t, m_t, _, c_t = rad._fields(lat, bandwidth_area, samples, ambiguities)
                if c_t.all():
                    raise ValueError('the antenna pattern is not known anywhere in the tasked area; give tasked_rniirs')
                info['tasked_rniirs'] = float(snr_to_rniirs(bandwidth_area, s_t[~c_t].min()))
                info['tasked_rniirs_source'] = 'model'
                notes.append('No tasked RNIIRS was given and the SICD parameters hold none, so the tasked quality is '
                             f"the lowest RNIIRS the model predicts inside the tasked area, {info['tasked_rniirs']:.2f}.")
                if (m_t < 0).any():
                    notes.append('Part of the tasked area lies outside the delay window TOA1..TOA2 in which the '
                                 'echoes have the full bandwidth; the tasked quality ignores that loss.')
            r_min = info['tasked_rniirs']
        else:
            r_min = float(kind)
        info['criterion'], info['threshold'] = 'rniirs', r_min
        thr = float(rniirs_to_snr(bandwidth_area, r_min))
        if info['center_rniirs'] < r_min:
            notes.append(f"The model predicts RNIIRS {info['center_rniirs']:.2f} at the reference point, below the "
                         f'threshold of {r_min:.2f}, so no area beyond the tasked one qualifies.')
    info['snr_threshold'] = thr
    # the criterion on the grid, or on a lattice of its nodes
    direct = n1 * n2 <= 300000
    i1 = np.arange(n1) if direct else np.unique(np.rint(np.linspace(0, n1 - 1, min(n1, 257))).astype(int))
    i2 = np.arange(n2) if direct else np.unique(np.rint(np.linspace(0, n2 - 1, min(n2, 257))).astype(int))
    lat = pts if direct else pts[i1][:, i2]
    snr, mg, mge, cl = rad._fields(lat, bandwidth_area, samples, ambiguities, scene)
    mg = mge if extended else mg
    snr = np.where(cl, 0.0, snr)
    ok = (snr >= thr) & (mg >= 0)
    if info['criterion'] == 'snr':
        crit = 10 * np.log10(np.maximum(snr, 1e-6))                 # thr is 1
    else:
        crit = snr_to_rniirs(bandwidth_area, snr) - info['threshold']
    lev = np.minimum(crit, 1e6 * np.maximum(mg, -1e-4))
    if cl.any() and (ok & ndimage.binary_dilation(cl)).any():
        notes.append('The area is limited in places by the extent to which the antenna pattern is known.')
    lab, _ = ndimage.label(ok)
    seed = np.zeros(ok.shape, bool)
    d = lat[..., :2] - rad.ref_point[:2]
    j = np.unravel_index(np.argmin((d * d).sum(-1)), ok.shape)
    step = max(np.linalg.norm(lat[min(1, len(i1) - 1), 0] - lat[0, 0]), np.linalg.norm(lat[0, min(1, len(i2) - 1)] - lat[0, 0]))
    if np.hypot(*d[j]) <= 1.5 * step:
        seed[j] = True
    if ring is not None:
        seed |= _in_polygon(lat[..., :2], ring[:, :2])
    region = np.isin(lab, np.unique(lab[seed & ok])) & ok
    level = np.empty((n1, n2), np.float32)
    if direct:
        mask = region
        if ring is not None:
            mask = mask | polygon()
        level[:] = np.where(mask, np.maximum(lev, 0.0), np.minimum(lev, -1e-3))
    else:
        mask = np.zeros((n1, n2), bool)
        k2, w2 = _weights(i2, np.arange(n2))
        reg = region.astype(np.float64)
        for a in range(0, n1, 512):
            rows = slice(a, min(a + 512, n1))
            k1, w1 = _weights(i1, np.arange(rows.start, rows.stop))

            def up(f):
                g = f[k1] * (1 - w1)[:, None] + f[np.minimum(k1 + 1, len(i1) - 1)] * w1[:, None]
                return g[:, k2] * (1 - w2) + g[:, np.minimum(k2 + 1, len(i2) - 1)] * w2
            m = (up(snr) >= thr) & (up(mg) >= 0) & (up(reg) > 0)
            mask[rows] = m | polygon(rows) if ring is not None else m
            lv = up(lev)
            level[rows] = np.where(mask[rows], np.maximum(lv, 0.0), np.minimum(lv, -1e-3))
    info['border'], info['level'] = border(mask), level
    return mask, info


def area_bounds(rad, kind, center, e1, e2, bandwidth_area, samples, tasked=None, tasked_rniirs=None, step=50.0,
                limit=30e3, *, ambiguities=True, black_sigma0_db=-10.0):
    """Bounding rectangle (lo1, hi1, lo2, hi2), in meters from `center` along the horizontal unit vectors e1 and e2
    (local frame), of the area area_mask selects; the other arguments are area_mask's. The mask is evaluated on a
    grid of spacing `step` (m) through the center that grows on every side the mask reaches until it no longer does
    or the side is `limit` (m) from the center; the result is padded by one step and always contains the tasked
    area."""
    center, e1, e2 = (np.asarray(v, np.float64) for v in (center, e1, e2))
    box = None
    if tasked is not None:
        d = np.asarray(tasked, np.float64).reshape(-1, 3) - center
        box = np.array([(d @ e1).min(), (d @ e1).max(), (d @ e2).min(), (d @ e2).max()])
    if isinstance(kind, str) and kind == 'tasked':
        if box is None:
            raise ValueError("kind 'tasked' needs the tasked area")
        return tuple(float(v) for v in box)
    sign = np.array([-1.0, 1.0, -1.0, 1.0])
    ext = np.full(4, 2000.0) if box is None else np.maximum(sign * box, 0) + 2 * step      # distance of each side
    ext = np.minimum(ext, limit)
    while True:
        a = np.arange(-np.ceil(ext[0] / step), np.ceil(ext[1] / step) + 1) * step
        b = np.arange(-np.ceil(ext[2] / step), np.ceil(ext[3] / step) + 1) * step
        pts = center + a[:, None, None] * e1 + b[None, :, None] * e2
        mask, info = area_mask(rad, pts, kind, bandwidth_area, samples, tasked, tasked_rniirs, ambiguities=ambiguities,
                               black_sigma0_db=black_sigma0_db)
        grow = np.array(info['border']) & (ext < limit)
        if not grow.any():
            break
        ext = np.where(grow, np.minimum(np.maximum(1.5 * ext, ext + 4 * step), limit), ext)
    if not mask.any():
        return tuple(float(v) for v in (box if box is not None else np.zeros(4)))
    r1, r2 = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    out = np.array([a[r1[0]] - step, a[r1[-1]] + step, b[r2[0]] - step, b[r2[-1]] + step])
    out = np.clip(out, -limit, limit)
    if box is not None:
        out = np.array([min(out[0], box[0]), max(out[1], box[1]), min(out[2], box[2]), max(out[3], box[3])])
    return tuple(float(v) for v in out)


def support_area(ant, band, point):
    """Ground-plane area (cycles^2 / m^2) of the spatial-frequency support of a spotlight image at the ground point
    `point` [3] (local frame): pulse p at frequency f lies at (2 f / c) times the horizontal part of the unit vector
    from the antenna (ant [P, 3]) to the point, and the support is the region these sweep over the pulses and over
    the band (first, last frequency in Hz). This is the area the samples fill, as the calibration needs it; the
    product of the bandwidths of its bounding rectangle is larger."""
    u = np.asarray(point, np.float64) - np.asarray(ant, np.float64)
    u = (u / np.linalg.norm(u, axis=1, keepdims=True))[:, :2]
    ring = np.concatenate([2 * band[0] / C * u, 2 * band[1] / C * u[::-1]])
    x, y = ring[:, 0], ring[:, 1]
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def _fit_2d(u, v, f, order):
    """Least squares in the relative error of a polynomial of the given order in each of u and v (in [-1, 1]) to the
    positive values f: coefficients [order + 1, order + 1] in powers of u and v."""
    cheb = np.polynomial.chebyshev
    V = cheb.chebvander2d(u, v, [order, order])
    c = np.linalg.lstsq(V / f[:, None], np.ones(len(f)), rcond=None)[0].reshape(order + 1, order + 1)
    T = np.zeros((order + 1, order + 1))                           # Chebyshev to power basis, by columns
    for k in range(order + 1):
        p = cheb.cheb2poly(np.eye(order + 1)[k])
        T[:len(p), k] = p
    return T @ c @ T.T


def calibration(rad, sicd, samples, bandwidth_area, coherent=None, noise=None, *, tasked=None, region=None,
                tolerance_db=0.1, max_order=8):
    """The content of the SICD Radiometric block of an image of the collection, from the collection's own levels.

    rad: a Radiometry with the file's PTRef (level_source 'cphd': with its PNRef; 'cphd+measured': with the noise
    measured from the phase history; a measurement that finds no noise leaves the noise level out). sicd: the
    image's SICD metadata (fastsar._sicdxml.SICD), whose projection gives each pixel's ground point. samples: the
    number of phase-history values the image sums (pulses x samples per pulse). bandwidth_area: the ground-plane
    area of their spatial-frequency support (support_area). coherent, noise: the sum of the aperture weights over
    the samples and the sum of their squares (both `samples` for an unweighted image, the default).

    The formers sum the weighted samples without normalization. With PTRef the per-sample power of a 1 m^2 point at
    the scene reference point, PN the noise power per sample and G(x) the two-way gain at the ground point x
    relative to the reference point (Radiometry.gain_db: the image is not corrected for the antenna pattern), the
    image has

        peak power of a point of 1 m^2             coherent^2 PTRef G(x)
        mean power of clutter of sigma zero 1      samples noise PTRef G(x) / bandwidth_area
        mean power of thermal noise                noise PN

    so pixel power times 1 / (coherent^2 PTRef G(x)) is RCS in m^2, and times bandwidth_area / (samples noise
    PTRef G(x)) it is sigma zero. Beta zero is sigma zero / cos(SlopeAng) and gamma zero sigma zero / sin(GrazeAng),
    SICD's relations, with the angles of the scene center point. The four factors are properties of the ground
    point, the same for a ground-plane and a slant-plane image of it. Each is written as a polynomial in SICD's
    image coordinates (meters from the scene center point along rows and along columns), a constant times one fit
    of 1 / G(x) by least squares in the relative error on a lattice over the image; the order, the same in both
    coordinates, is the lowest for which the fit is within tolerance_db of the model on a finer lattice, up to
    max_order (order 8 follows a gain that falls some 30 dB over the image; where no order reaches the tolerance, the
    one with the least error is taken and residual_db says what it is). tasked: the corners [4, 3] of the tasked area,
    for the residual there. region: a function of lattice points [n1, n2, 3] (local) -> the mask of those to fit
    and to judge: the pixels that hold data (default: all of them; a slant-plane image is zero beyond the ground
    that was formed, and the polynomials mean nothing there). Points where the antenna pattern is not known are
    left out.

    -> dict(source (rad.level_source), noise_db (thermal noise power of a pixel, dB: NoisePoly, a constant; None
    without a noise power), rcs, sigma0, beta0, gamma0 (coefficients [n + 1, n + 1], exponent of the row coordinate
    first), order, residual_db (largest error of the fit over the region), tasked_residual_db (over the tasked
    area, or None), gain_span_db (the range of G over the region), nesz_db (noise-equivalent sigma zero at the
    scene center point, or None), scp, urow, ucol (the frame the polynomials are in, ECF))."""
    from .io import ecf_to_local
    if getattr(rad, 'pt_ref', None) is None:
        raise ValueError('the calibration needs the PTRef of the CPHD file '
                         f"(level_source 'cphd' or 'cphd+measured', not {rad.level_source!r})")
    n = float(samples)
    coherent, noise = (n if v is None else float(v) for v in (coherent, noise))
    rows, cols = int(sicd.ImageData.NumRows), int(sicd.ImageData.NumCols)
    r0, c0 = float(sicd.ImageData.SCPPixel.Row), float(sicd.ImageData.SCPPixel.Col)
    ssr, ssc = float(sicd.Grid.Row.SS), float(sicd.Grid.Col.SS)
    hx, hy = (max(abs(a), abs(b), 1.0) for a, b in ((-r0 * ssr, (rows - 1 - r0) * ssr),
                                                    (-c0 * ssc, (cols - 1 - c0) * ssc)))

    def lattice(m):
        """m x m pixels over the image: coordinates in [-1, 1], 1 / G, and which of them count."""
        r, c = np.meshgrid(np.linspace(0, rows - 1, m), np.linspace(0, cols - 1, m), indexing='ij')
        pts = ecf_to_local(sicd.project_image_to_ground(np.stack([r, c], -1)), rad.meta)
        ev = rad._evaluate(pts, clamp=True)
        ok = np.isfinite(pts).all(-1) & ~ev['clamped'] & (ev['gain'] > 0)
        if region is not None:
            ok &= np.asarray(region(pts), bool)
        inside = None if tasked is None else _in_polygon(pts[..., :2], _ring(tasked)[:, :2])
        return ((r - r0) * ssr / hx)[ok], ((c - c0) * ssc / hy)[ok], 1.0 / ev['gain'][ok], \
            None if inside is None else inside[ok]

    u, v, f, _ = lattice(33)
    uu, vv, ff, inside = lattice(64)                               # the fit is judged between its own nodes
    if len(f) < 4 or len(ff) < 4:
        raise ValueError('the antenna pattern is known at too few points of the image for a fit of the scale factors')
    best = None
    for order in range(0, int(max_order) + 1):
        if (order + 1) ** 2 > len(f):
            break
        coef = _fit_2d(u, v, f, order)
        p = np.polynomial.polynomial.polyval2d(uu, vv, coef)
        err = np.abs(10 * np.log10(np.where(p > 0, p, 1e-300) / ff))
        res = float(err.max())
        if best is None or res < best[0]:
            best = (res, order, coef, err)
        if res <= tolerance_db:
            break
    res, order, coef, err = best
    k = np.arange(order + 1)
    inv_gain = coef / (hx ** k[:, None] * hy ** k[None, :])         # 1 / G in meters from the scene center point
    rcs = 1.0 / (coherent ** 2 * rad.pt_ref)
    sigma = float(bandwidth_area) / (n * noise * rad.pt_ref)
    slope, graze = np.radians(float(sicd.SCPCOA.SlopeAng)), np.radians(float(sicd.SCPCOA.GrazeAng))
    pn = rad.noise_per_sample
    noise_db = float(10 * np.log10(noise * pn)) if pn is not None and pn > 0 else None
    return dict(source=rad.level_source, noise_db=noise_db, rcs=rcs * inv_gain, sigma0=sigma * inv_gain,
                beta0=sigma / np.cos(slope) * inv_gain, gamma0=sigma / np.sin(graze) * inv_gain, order=int(order),
                residual_db=res,
                tasked_residual_db=None if inside is None or not inside.any() else float(err[inside].max()),
                gain_span_db=float(10 * np.log10(ff.max() / ff.min())),
                nesz_db=None if noise_db is None else float(noise_db + 10 * np.log10(sigma * inv_gain[0, 0])),
                scp=np.asarray(sicd.GeoData.SCP.ECF.get_array(), np.float64),
                urow=np.asarray(sicd.Grid.Row.UVectECF.get_array(), np.float64),
                ucol=np.asarray(sicd.Grid.Col.UVectECF.get_array(), np.float64))
