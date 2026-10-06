"""Form a crop of a Capella stripmap collection from its CPHD with the patch mosaic, sample it on the pixels of the
vendor's SICD (a range / zero-Doppler grid) and compare the two amplitude images.

    python examples/form_capella_stripmap.py scene_CPHD.cphd scene_SICD.ntf out [--size 512] [--backend cpu]

The scene reference point of a stripmap CPHD moves with the beam; read_cphd re-references the phase history to the
mid-aperture point, and each pulse's own point gives the beam center. The beam's extent along track is the azimuth
bandwidth the vendor processed (SICD Grid.Col.ImpRespBW).
"""
import argparse, time

import numpy as np

import fastsar
from fastsar import io, patches, products

C = 299792458.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cphd'); ap.add_argument('sicd'); ap.add_argument('out')
    ap.add_argument('--size', type=int, default=512); ap.add_argument('--backend', default='cpu')
    ap.add_argument('--spacing', type=float, default=0.35)
    a = ap.parse_args()
    from sarpy.io.complex.converter import open_complex
    from sarpy.io.phase_history.converter import open_phase_history
    col, meta = io.read_cphd(a.cphd, meta=True)
    S, ant, f0, df = col['S'], col['ant'], col['fmin'], col['df']
    P, K = S.shape
    srp = io.ecf_to_local(open_phase_history(a.cphd).read_pvp_variable('SRPPos', 0), meta)
    srp = srp[:P] if len(srp) >= P else srp
    rd = open_complex(a.sicd)
    sm = rd.sicd_meta
    dsin = float(sm.Grid.Col.ImpRespBW) * C / (f0 + K / 2 * df) / 2      # processed spread of sin(look angle)
    d = np.gradient(ant, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    s_c = ((srp - ant) * d).sum(1) / np.linalg.norm(srp - ant, axis=1)        # beam center of each pulse

    def beam(idx, pts):
        w = np.asarray(pts)[None] - ant[idx][:, None]
        return ((w * d[idx][:, None]).sum(-1) / np.linalg.norm(w, axis=-1) - s_c[idx][:, None]) / (dsin / 2)

    n = a.size
    r0, c0 = int(sm.ImageData.NumRows) // 2 - n // 2, int(sm.ImageData.NumCols) // 2 - n // 2
    rr, cc = np.meshgrid(np.arange(r0, r0 + n), np.arange(c0, c0 + n), indexing='ij')
    pts = io.sicd_points(sm, rr, cc, meta)
    e1, e2 = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    lo, hi = pts.reshape(-1, 3).min(0), pts.reshape(-1, 3).max(0)
    origin = np.array([lo[0] - 5, lo[1] - 5, pts[..., 2].mean()])
    nx, ny = int((hi[0] - lo[0] + 10) / a.spacing), int((hi[1] - lo[1] + 10) / a.spacing)
    t = time.perf_counter()
    img = patches.form_mosaic(dict(S=S, fmin=f0, df=df, ref=meta['ref'], band=(f0, f0 + K * df)), ant, origin, nx, ny,
                              a.spacing, a.spacing, e1, e2, beam=beam, backend=a.backend)
    print(f'{nx} x {ny} ground pixels formed in {time.perf_counter() - t:.0f} s')
    ours = products.sample(img, np.stack([(pts - origin) @ e1 / a.spacing, (pts - origin) @ e2 / a.spacing], -1))
    v = rd[r0:r0 + n, c0:c0 + n]
    print(f'amplitude correlation with the vendor image: {np.corrcoef(np.abs(ours).ravel(), np.abs(v).ravel())[0, 1]:.3f}')
    np.save(a.out + '.npy', ours)


if __name__ == '__main__':
    main()
