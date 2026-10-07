"""From a CPHD file to products: a complex SICD with its geometry, a map-projected amplitude GeoTIFF (dB, UTM) and a
PNG quicklook of it.

    python examples/chain.py scene_CPHD.cphd out [--sicd scene_SICD.ntf] [--dem dem.tif --geoid 31.5] [--autofocus]
    python examples/chain.py --simulate out          # writes a small simulated collection to out_sim.cphd first

Writes out.nitf (SICD), out_db.tif and out_db.png. --looks sets the multilooking (along track, across track) before
detection; --dem a DEM GeoTIFF for terrain correction, with --geoid the geoid's height above the ellipsoid at the
scene when the DEM gives heights above the geoid (without --dem the map lies at the image center's height).
"""
import argparse, time

import numpy as np

import fastsar
from fastsar import products, sim


def simulate(path):
    """A 100 m spotlight scene at 6 km (0.5 m resolution): a 3 x 3 grid of point targets over weak clutter."""
    rng = np.random.default_rng(0)
    col = sim.make_collect(res=0.5, scene=100.0, r0=6e3, graze_deg=35.0)
    g = np.linspace(-30.0, 30.0, 3)
    pts = np.stack([*np.meshgrid(g, g), np.zeros((3, 3))], -1).reshape(-1, 3)
    cl = np.concatenate([rng.uniform(-45, 45, (3000, 2)), np.zeros((3000, 1))], 1)
    pos = np.concatenate([pts, cl])
    amp = np.concatenate([np.full(9, 30.0), 0.5 * (rng.standard_normal(3000) + 1j * rng.standard_normal(3000))])
    S = sum(sim.simulate_brute(col, pos[i:i + 50], amp[i:i + 50]) for i in range(0, len(pos), 50))
    sim.write_cphd(path, col, S.astype(np.complex64), lat=40.7934, lon=-77.86, height=351.0, heading=30.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cphd', nargs='?'); ap.add_argument('out')
    ap.add_argument('--simulate', action='store_true'); ap.add_argument('--sicd')
    ap.add_argument('--dem'); ap.add_argument('--geoid', type=float, default=0.0)
    ap.add_argument('--autofocus', action='store_true'); ap.add_argument('--backend', default='auto')
    ap.add_argument('--looks', type=int, nargs=2, default=(2, 2)); ap.add_argument('--spacing', type=float)
    a = ap.parse_args()
    if a.simulate:
        a.cphd = a.out + '_sim.cphd'
        simulate(a.cphd)
    t = time.perf_counter()
    out = fastsar.form_cphd(a.cphd, sicd=a.sicd, backend=a.backend, autofocus=a.autofocus)
    img = out['image']
    print(f'{out["mode"]} image {img.shape}, {out["spx"]:.3f} x {out["spy"]:.3f} m pixels, '
          f'{time.perf_counter() - t:.1f} s')
    for n in out['notes']:
        print('  ' + n)
    nx, ny = img.shape
    lat, lon, h = products.geolocate(out, [0, 0, nx - 1, nx - 1, nx / 2], [0, ny - 1, 0, ny - 1, ny / 2])
    print('corners and center (lat, lon, height on the image plane):')
    for v in zip(lat, lon, h):
        print('  %.6f %.6f %.1f' % v)

    products.write_sicd(a.out + '.nitf', out)
    dem = products.read_dem(a.dem, offset=a.geoid) if a.dem else None
    power = products.multilook(img, *a.looks)
    geo = products.geocode_image(out, power, spacing=a.spacing, height=dem)
    geo['data'] = products.to_db(geo['data'])
    products.write_geotiff(a.out + '_db.tif', **geo)
    print(f'wrote {a.out}.nitf and {a.out}_db.tif ({geo["crs"]}, {geo["data"].shape[1]} x {geo["data"].shape[0]})')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        db = geo['data']
        top = np.nanpercentile(db, 99.5)
        plt.imsave(a.out + '_db.png', np.nan_to_num(db, nan=top - 40), cmap='gray', vmin=top - 40, vmax=top)
        print(f'wrote {a.out}_db.png (north up)')
    except ImportError:
        print('no matplotlib: PNG skipped')


if __name__ == '__main__':
    main()
