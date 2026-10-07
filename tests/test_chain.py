"""The chain from a CPHD file to products, on a simulated spotlight collection written as CPHD (sim.write_cphd):
form_cphd, the amplitude convention, pixel <-> latitude/longitude both ways (on the plane, at a height, on a DEM),
map-projected GeoTIFFs (UTM and latitude/longitude, with and without the DEM), a SICD round trip whose own
projection is compared with FastSAR's, and form_cphd's autofocus. Needs sarpy; the GeoTIFF part needs rasterio."""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fastsar
from fastsar import sim, io, products, quality

lat0, lon0, h0, heading = 40.7934, -77.8600, 351.0, 30.0
col = sim.make_collect(res=0.5, scene=100.0, r0=6e3, graze_deg=35.0)
# targets on a sloped terrain z = 0.3 x + 0.2 y (simulator frame: x across track, y along track)
xy = np.array([[0.0, 0.0], [12.0, -15.0], [-15.0, 8.0], [10.0, 20.0], [-20.0, -18.0]])
tg = np.concatenate([xy, (0.3 * xy[:, 0] + 0.2 * xy[:, 1])[:, None]], 1)
truth = io.ecf_to_geodetic(sim.to_ecf(tg, lat0, lon0, h0, heading))
M = np.array([[np.cos(np.radians(heading)), np.sin(np.radians(heading)), 0.0],
              [-np.sin(np.radians(heading)), np.cos(np.radians(heading)), 0.0], [0.0, 0.0, 1.0]])


def dem(lat, lon):
    """The terrain's height above the ellipsoid at (lat, lon)."""
    s = (io.geodetic_to_ecf(lat, lon, h0) - io.geodetic_to_ecf(lat0, lon0, h0)) @ sim.enu_axes(lat0, lon0).T @ M
    return h0 + 0.3 * s[..., 0] + 0.2 * s[..., 1]


def ground_m(lat1, lon1, lat2, lon2):
    """Horizontal distance (m) between nearby points."""
    return np.hypot((lat1 - lat2) * 111132.0, (lon1 - lon2) * 111320.0 * np.cos(np.radians(lat1)))


def peaks(img, ij, spx, spy):
    """Upsampled peak positions [n, 2] and amplitudes near the predicted pixels ij."""
    out = []
    for i0, j0 in np.round(ij).astype(int):
        w = np.abs(img[i0 - 5:i0 + 6, j0 - 5:j0 + 6])
        a, b = np.unravel_index(np.argmax(w), w.shape)
        m = quality.point_target(img, (i0 - 5 + a, j0 - 5 + b), spx, spy, half=16)
        out.append((m['i'], m['j'], m['peak']))
    return np.array(out)


tmp = tempfile.mkdtemp()
S = sim.simulate_brute(col, tg, np.ones(len(tg))).astype(np.complex64)
path = os.path.join(tmp, 'sim.cphd')
sim.write_cphd(path, col, S, lat0, lon0, h0, heading=heading)
t0 = time.perf_counter()
out = fastsar.form_cphd(path, backend='cpu', spacing=0.4, height=h0)
img = out['image']
print(f'{col.Np} pulses, {col.K} samples; form_cphd: {out["mode"]}, image {img.shape}, '
      f'{time.perf_counter() - t0:.1f} s')

# amplitude: a unit point on the grid plane peaks at the sum of the window (Taylor windows of mean 1: Np K)
pk = peaks(img, products.locate(out, *truth), out['spx'], out['spy'])
print(f'peak of the unit target at the center / (Np K): {pk[0, 2] / (col.Np * col.K):.4f}')
assert abs(pk[0, 2] / (col.Np * col.K) - 1) < 0.01

# where the targets appear (locate, terrain heights) and where their peaks map back to (geolocate at their heights)
ij = products.locate(out, *truth)
d_img = np.hypot((pk[:, 0] - ij[:, 0]) * out['spx'], (pk[:, 1] - ij[:, 1]) * out['spy'])
g = products.geolocate(out, pk[:, 0], pk[:, 1], height=truth[2])
d_geo = ground_m(g[0], g[1], truth[0], truth[1])
g_dem = products.geolocate(out, pk[:, 0], pk[:, 1], height=dem)
d_dem = np.sqrt(ground_m(g_dem[0], g_dem[1], truth[0], truth[1]) ** 2 + (g_dem[2] - truth[2]) ** 2)
lay = products.geolocate(out, pk[:, 0], pk[:, 1])               # on the image plane (at h0)
print('target heights above the plane (m):', np.round(truth[2] - h0, 1).tolist())
print(f'peaks vs locate: {d_img.max() * 100:.2f} cm; geolocated peaks vs truth: at the true heights '
      f'{d_geo.max() * 100:.2f} cm, on the DEM {d_dem.max() * 100:.2f} cm; on the plane (no terrain correction) '
      f'up to {ground_m(lay[0], lay[1], truth[0], truth[1]).max():.1f} m')
assert d_img.max() < 0.03 and d_geo.max() < 0.03 and d_dem.max() < 0.03

# round trips over the whole image: pixel -> ground -> pixel
I, J = np.meshgrid(np.linspace(-0.5, img.shape[0] - 0.5, 9), np.linspace(-0.5, img.shape[1] - 0.5, 9), indexing='ij')
worst = 0.0
for h in (None, h0, h0 + 40.0, dem):
    gg = products.geolocate(out, I, J, height=h)
    back = products.locate(out, *gg)
    worst = max(worst, np.abs(back[..., 0] - I).max(), np.abs(back[..., 1] - J).max())
print(f'pixel -> latitude, longitude, height -> pixel: {worst:.1e} pixels')
assert worst < 1e-4

try:
    import rasterio
    from rasterio.warp import transform as warp
except ImportError:
    rasterio = None
if rasterio is not None:
    amp = np.abs(img)
    I, J = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    for crs, h in (('utm', dem), ('EPSG:4326', dem), ('utm', None)):
        f = os.path.join(tmp, 'map.tif')
        # the pixel coordinates i + 1j j, geocoded (bilinear interpolation keeps them exact): at each target's map
        # position the map must hold the pixel the target appears at
        products.write_geotiff(f, **products.geocode_image(out, I + 1j * J, crs=None if crs == 'utm' else crs, height=h))
        with rasterio.open(f) as src:
            A, tr, c = src.read(1), src.transform, src.crs
        x, y = warp('EPSG:4326', c, truth[1], truth[0])
        cr = np.array([~tr * p for p in zip(x, y)])                    # fractional (col, row) of pixel corners
        v = products.sample(A, cr[:, ::-1] - 0.5)
        ij_h = products.locate(out, truth[0], truth[1], truth[2] if h is not None else h0)
        d_map = np.abs(np.stack([v.real, v.imag], 1) - ij_h).max()
        # each target's brightest map pixel of the amplitude, refined by a parabola in each direction, against its
        # true position (DEM) or, without the DEM, the point at the map's height that has its range and Doppler
        exp = truth if h is not None else lay
        x, y = warp('EPSG:4326', c, exp[1], exp[0])
        cr = np.array([~tr * p for p in zip(x, y)])
        products.write_geotiff(f, **products.geocode_image(out, amp, crs=None if crs == 'utm' else crs, height=h))
        with rasterio.open(f) as src:
            A = src.read(1)
        errs = []
        for (cc, r), la, lo in zip(cr.astype(int), exp[0], exp[1]):
            w = np.nan_to_num(A[r - 4:r + 5, cc - 4:cc + 5])
            a_, b_ = np.unravel_index(np.argmax(w), w.shape)
            r, cc = r - 4 + a_, cc - 4 + b_
            par = lambda y0, y1, y2: 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            x1, y1 = tr * (cc + 0.5 + par(*A[r, cc - 1:cc + 2]), r + 0.5 + par(*A[r - 1:r + 2, cc]))
            lo1, la1 = warp(c, 'EPSG:4326', [x1], [y1])
            errs.append(ground_m(la1[0], lo1[0], la, lo))
        errs = np.array(errs)
        pix = abs(tr.a) * (111320.0 * np.cos(np.radians(lat0)) if c.is_geographic else 1.0)
        print(f'GeoTIFF {str(c):10s} {A.shape[1]} x {A.shape[0]}, {pix:.2f} m pixels, '
              f'{"DEM" if h is not None else "no DEM"}: pixel coordinates at the targets {d_map:.1e} pixels from '
              f'locate; amplitude peaks from the {"true" if h is not None else "laid-over"} positions '
              + ', '.join(f'{e:.2f}' for e in errs) + ' m')
        assert d_map < 1e-3 and errs.max() < 0.5 * max(out['spx'], out['spy']), (d_map, errs)
    # the terrain as a DEM GeoTIFF (2 m posts in latitude and longitude, heights 20 m below the ellipsoid's), read back
    d = 2.0 / 111132.0
    la_, lo_ = lat0 + 0.0012 - d * (np.arange(120) + 0.5), lon0 - 0.0016 + d * (np.arange(120) + 0.5)
    f = os.path.join(tmp, 'dem.tif')
    products.write_geotiff(f, dem(*np.meshgrid(la_, lo_, indexing='ij')) - 20.0, (d, 0.0, lon0 - 0.0016, 0.0, -d, lat0 + 0.0012))
    g_tif = products.geolocate(out, pk[:, 0], pk[:, 1], height=products.read_dem(f, offset=20.0))
    d_tif = np.sqrt(ground_m(g_tif[0], g_tif[1], truth[0], truth[1]) ** 2 + (g_tif[2] - truth[2]) ** 2)
    print(f'geolocated peaks on the DEM read from a GeoTIFF: {d_tif.max() * 100:.2f} cm from the truth')
    assert d_tif.max() < 0.03
    # complex GeoTIFF (nearest pixel: phases unchanged), and multilooked data
    geo = products.geocode_image(out, img, height=dem, order=0)
    products.write_geotiff(os.path.join(tmp, 'slc.tif'), **geo)
    with rasterio.open(os.path.join(tmp, 'slc.tif')) as src:
        B = src.read(1)
    ok = np.isfinite(geo['data'])
    assert B.dtype == np.complex64 and np.array_equal(B[ok], geo['data'][ok].astype(np.complex64))
    assert np.isin(B[ok], img).all()
    ml = products.geocode_image(out, products.multilook(img, 2, 2), height=dem)
    print(f'complex GeoTIFF: {ok.sum()} pixels, every one a pixel of the image; multilooked 2 x 2: {ml["data"].shape}')
else:
    print('GeoTIFF checks skipped: no rasterio')

# SICD: pixels kept, and its own projection model (sarpy) agrees with locate and geolocate
from sarpy.io.complex.converter import open_complex
f = os.path.join(tmp, 'out.nitf')
products.write_sicd(f, out)
rd = open_complex(f)
sm, data = rd.sicd_meta, rd[:, :]
flip = int(sm.ImageData.SCPPixel.Row) != img.shape[1] // 2
assert np.array_equal(data, (img.T[::-1] if flip else img.T))
rc = lambda i, j: np.stack([(img.shape[1] - 1 - j) if flip else j, i], -1)
e_sicd = np.array([sm.project_image_to_ground(rc(i, j)[None], projection_type='HAE', hae0=h)[0]
                   for i, j, h in zip(pk[:, 0], pk[:, 1], truth[2])])
d_sicd = np.linalg.norm(e_sicd - io.geodetic_to_ecf(*g), axis=1)
d_true = np.linalg.norm(e_sicd - sim.to_ecf(tg, lat0, lon0, h0, heading), axis=1)
back = np.array([sm.project_ground_to_image(io.geodetic_to_ecf(la, lo, h)[None], tolerance=1e-5)[0][0]
                 for la, lo, h in zip(*truth)])
d_back = np.abs(back - rc(ij[:, 0], ij[:, 1])).max()
print(f'SICD ({sm.ImageData.NumRows} x {sm.ImageData.NumCols}, rows {"along -e2" if flip else "along e2"}): pixels '
      f'identical; sarpy projection of the target peaks vs geolocate {d_sicd.max() * 1000:.3f} mm, vs truth '
      f'{d_true.max() * 100:.2f} cm; ground to image vs locate {d_back:.1e} pixels')
assert d_sicd.max() < 0.01 and d_true.max() < 0.03 and d_back < 1e-3
f2 = os.path.join(tmp, 'again.nitf')
products.write_sicd(f2, data, f, transpose=False)                 # the template path: that SICD's metadata
assert np.array_equal(open_complex(f2)[:, :], data)
try:
    import sarkit.verification as skv
    for name, kind, fn in (('simulated CPHD', skv.CphdConsistency, path), ('SICD', skv.SicdConsistency, f)):
        with open(fn, 'rb') as fh:
            con = kind.from_file(fh)
        con.check()
        fail = con.failures(omit_passed_sub=True)
        err = sorted(k for k, v in fail.items() if any(d['severity'] == 'Error' for d in v['details']))
        warn = sorted(set(fail) - set(err))
        print(f'sarkit consistency of the {name}: {len(err)} errors' + (f', warnings {warn}' if warn else ''))
        assert not err, err
except ImportError:
    print('sarkit consistency checks skipped: no sarkit')

# autofocus: a phase error per pulse, written into the CPHD, and removed by form_cphd(autofocus=True)
t = np.linspace(-1, 1, col.Np)
err = 6 * t ** 2 + 2 * t ** 3
path2 = os.path.join(tmp, 'blur.cphd')
sim.write_cphd(path2, col, S * np.exp(1j * err)[:, None].astype(np.complex64), lat0, lon0, h0, heading=heading)
blur = fastsar.form_cphd(path2, backend='cpu', spacing=0.4, height=h0)['image']
t0 = time.perf_counter()
af = fastsar.form_cphd(path2, backend='cpu', spacing=0.4, height=h0, autofocus=True)
ib, ia = (peaks(x, ij, out['spx'], out['spy'])[:, 2] / pk[:, 2] for x in (blur, af['image']))
res = fastsar.autofocus._detrend(af['phase_error'] - err, np.arange(col.Np, dtype=float))
print(f'autofocus ({time.perf_counter() - t0:.1f} s): peak / focused peak {ib.min():.2f} before, {ia.min():.3f} after; '
      f'residual phase error {res.std():.3f} rad rms')
assert ia.min() > 0.97 and res.std() < 0.1

# a moving beam (stripmap: the scene reference point follows the antenna), formed as a mosaic: geolocation holds,
# since for a straight track the range-Doppler circle does not depend on the reference pulse
C = 299792458.0
srp = np.zeros_like(col.ant)
srp[:, 1] = col.ant[:, 1]
tg2 = np.array([[0.0, 0.0, 0.0], [12.0, -15.0, 5.0], [-15.0, 18.0, -5.0], [10.0, 10.0, 0.0]])
dr = np.linalg.norm(tg2[None] - col.ant[:, None], axis=-1) - np.linalg.norm(srp - col.ant, axis=1)[:, None]
S2 = np.exp(-4j * np.pi / C * col.freqs[None, :, None] * dr[:, None, :]).sum(-1).astype(np.complex64)
path3 = os.path.join(tmp, 'strip.cphd')
sim.write_cphd(path3, col, S2, lat0, lon0, h0, heading=heading, srp=srp)
st = fastsar.form_cphd(path3, backend='cpu', spacing=0.4, height=h0, extent=(80.0, 80.0))
truth2 = io.ecf_to_geodetic(sim.to_ecf(tg2, lat0, lon0, h0, heading))
pk2 = peaks(st['image'], products.locate(st, *truth2), st['spx'], st['spy'])
g2 = products.geolocate(st, pk2[:, 0], pk2[:, 1], height=truth2[2])
d2 = ground_m(g2[0], g2[1], truth2[0], truth2[1])
print(f'moving beam ({st["mode"]}, {st["image"].shape}): geolocated peaks vs truth {d2.max() * 100:.2f} cm')
assert st['mode'] == 'moving' and d2.max() < 0.03
print('ok')
