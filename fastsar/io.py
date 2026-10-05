"""Read a spotlight collection from CPHD (frequency domain, fixed scene reference point) and, optionally, the image
grid of the matching SICD, into the arrays form_image takes. Needs sarpy.

    col = fastsar.io.read_cphd('x_CPHD.cphd', sicd='x_SICD.nitf')
    img = fastsar.form_image(**col)

The local frame has x along track, y along ground range away from the radar and z up, with its origin at the scene
reference point. With a SICD the output grid is the vendor's own (pixel counts, spacings and image-plane axes);
without one, pass nx, ny, spx, spy, e1, e2 to form_image yourself.
"""
import numpy as np


def read_cphd(cphd, sicd=None):
    """-> dict(S, ant, fmin, df[, nx, ny, spx, spy, e1, e2]) ready for form_image(**d)."""
    from sarpy.io.phase_history.converter import open_phase_history
    r = open_phase_history(cphd)
    m = r.cphd_meta
    ch = m.Data.Channels[0]
    P, K = ch.NumVectors, ch.NumSamples
    if m.Global.DomainType != 'FX' or not m.Channel.Parameters[0].SRPFixed:
        raise ValueError('needs a frequency-domain CPHD with a fixed scene reference point (spotlight)')
    tx, rcv, srp = (r.read_pvp_variable(n, 0) for n in ('TxPos', 'RcvPos', 'SRPPos'))
    sc0, scss = r.read_pvp_variable('SC0', 0), r.read_pvp_variable('SCSS', 0)
    f0, df = float(sc0.mean()), float(scss.mean())
    S = r.read_chip((0, P), (0, K), index=0).astype(np.complex64)
    if m.Global.SGN > 0:
        S = np.conj(S)
    apc = 0.5 * (tx + rcv) - srp[0]
    # pulses without a valid position or with all-zero samples at the ends of the aperture are trimmed; an invalid
    # interior position is interpolated from its neighbors
    bad = ~np.isfinite(apc).all(1) | ~np.isfinite(S).all(1) | (np.abs(S).sum(1) == 0)
    good = np.nonzero(~bad)[0]
    lo, hi = int(good[0]), int(good[-1]) + 1
    S, apc, bad = S[lo:hi], apc[lo:hi], bad[lo:hi]
    if bad.any():
        idx = np.arange(len(apc))
        for c in range(3):
            apc[bad, c] = np.interp(idx[bad], idx[~bad], apc[~bad, c])
        S[~np.isfinite(S)] = 0
    P = len(apc)
    up = srp[0] / np.linalg.norm(srp[0])
    mid = apc[P // 2]
    los_h = mid - (mid @ up) * up
    yhat = -los_h / np.linalg.norm(los_h)
    xhat = np.cross(yhat, up)
    R = np.stack([xhat, yhat, up])
    out = dict(S=S, ant=apc @ R.T, fmin=f0, df=df)
    if sicd is not None:
        from sarpy.io.complex.converter import open_complex
        sm = open_complex(sicd).sicd_meta
        rows, cols = int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)
        row_ss, col_ss = float(sm.Grid.Row.SS), float(sm.Grid.Col.SS)
        row_u = np.array(sm.Grid.Row.UVectECF.get_array()) @ R.T
        col_u = np.array(sm.Grid.Col.UVectECF.get_array()) @ R.T
        range_is_row = abs(row_u[1]) > abs(row_u[0])
        e2, e1 = (row_u, col_u) if range_is_row else (col_u, row_u)
        out.update(nx=cols if range_is_row else rows, ny=rows if range_is_row else cols,
                   spx=col_ss if range_is_row else row_ss, spy=row_ss if range_is_row else col_ss,
                   e1=e1 / np.linalg.norm(e1), e2=e2 / np.linalg.norm(e2))
    return out
