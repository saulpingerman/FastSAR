"""Products from formed images: multilooked detection, interferograms and coherence (interferometry and coherent change
detection), range-Doppler projection of 3-D points into the image (geocoding and terrain correction), GeoTIFF and
SICD output.

    amp_db = products.to_db(products.multilook(img, 2, 2))
    vals = products.geocode(img, grid, ant, points)           # image sampled at 3-D points (e.g. a map grid on a DEM)
    products.write_sicd('out.nitf', img, 'vendor_SICD.nitf')  # image formed on the vendor's grid by read_cphd

grid is the dict of nx, ny, spx, spy, e1, e2 the image was formed on (read_cphd's output carries them), and ant the
antenna positions in the same local frame. A point x off the image plane appears where the plane has the same range
and the same Doppler cone angle at the aperture center: |p - a_c| = |x - a_c| and (p - a_c) . v = (x - a_c) . v,
with a_c and v the mid-aperture antenna position and the direction of motion. project solves the two conditions in
closed form (a line in the plane intersected with a circle), so terrain correction is exact for that model; for
apertures much longer than the image is wide it holds to second order in the angle the aperture subtends.
"""
import numpy as np


def multilook(img, la=1, lr=1):
    """Power |img|^2 averaged over la x lr pixel boxes (axis 0, axis 1) and decimated by them."""
    p = np.abs(np.asarray(img)) ** 2
    nx, ny = (p.shape[0] // la) * la, (p.shape[1] // lr) * lr
    return p[:nx, :ny].reshape(nx // la, la, ny // lr, lr).mean((1, 3))


def _box(x, w1, w2):
    """Moving average over w1 x w2 windows, same shape (edges average what is there)."""
    from scipy.ndimage import uniform_filter
    f = lambda v: uniform_filter(v, (w1, w2), mode='nearest')
    return f(x.real) + 1j * f(x.imag) if np.iscomplexobj(x) else f(x)


def interferogram(a, b, la=1, lr=1):
    """Multilooked interferogram: a conj(b) averaged over la x lr boxes and decimated, complex (np.angle is the
    interferometric phase). a and b must be formed on the same grid (coregistered), which backprojection onto one
    set of points or one planar grid gives directly."""
    x = np.asarray(a) * np.conj(np.asarray(b))
    nx, ny = (x.shape[0] // la) * la, (x.shape[1] // lr) * lr
    return x[:nx, :ny].reshape(nx // la, la, ny // lr, lr).mean((1, 3))


def coherence(a, b, w1=5, w2=5):
    """Sample coherence |<a b*>| / sqrt(<|a|^2> <|b|^2>) over moving w1 x w2 windows, same shape as the images:
    the coherent change detection statistic (low where the scene changed between passes). It is biased upward for
    small windows, to about 1/sqrt(w1 w2) for incoherent pairs."""
    a, b = np.asarray(a).astype(np.complex128), np.asarray(b).astype(np.complex128)
    num = np.abs(_box(a * np.conj(b), w1, w2))
    den = np.sqrt(_box(np.abs(a) ** 2, w1, w2) * _box(np.abs(b) ** 2, w1, w2))
    return num / np.maximum(den, 1e-300)


def pauli(hh, hv, vv, vh=None, la=1, lr=1):
    """Pauli decomposition of a quad-polarization image set (formed on one grid): the powers of
    (HH - VV)/sqrt(2) (double bounce), 2 HV/sqrt(2) with HV the mean of HV and VH when both are given (volume), and
    (HH + VV)/sqrt(2) (surface), multilooked over la x lr boxes; stacked [..., 3] in the usual red, green, blue order."""
    x = np.asarray(hv) if vh is None else 0.5 * (np.asarray(hv) + np.asarray(vh))
    k = [(np.asarray(hh) - np.asarray(vv)) / np.sqrt(2), np.sqrt(2) * x, (np.asarray(hh) + np.asarray(vv)) / np.sqrt(2)]
    return np.stack([multilook(c, la, lr) for c in k], -1)


def to_db(power, floor=1e-30):
    return 10 * np.log10(np.maximum(power, floor))


def _aperture(ant):
    ant = np.asarray(ant, np.float64)
    m = len(ant) // 2
    v = ant[min(m + 1, len(ant) - 1)] - ant[max(m - 1, 0)]
    return ant[m], v / np.linalg.norm(v)


def project(points, ant, nx, ny, spx, spy, e1, e2, **_):
    """Fractional pixel coordinates (i, j) [..., 2] at which points [..., 3] appear in an image formed on the grid
    (nx, ny, spx, spy, e1, e2); NaN where no plane point has the point's range and Doppler."""
    a, v = _aperture(ant)
    e1, e2 = np.asarray(e1, np.float64), np.asarray(e2, np.float64)
    x = np.asarray(points, np.float64)
    R2 = ((x - a) ** 2).sum(-1)
    D = (x - a) @ v
    # p = s e1 + t e2: s (e1.v) + t (e2.v) = D + a.v defines a line p0 + l d in the plane
    g1, g2 = e1 @ v, e2 @ v
    nrm = np.hypot(g1, g2)
    d = (-g2 * e1 + g1 * e2) / nrm
    p0 = ((D + a @ v) / nrm ** 2)[..., None] * (g1 * e1 + g2 * e2)
    # |p0 + l d - a|^2 = R2, with d a unit vector in the plane: l^2 + 2 l (p0 - a).d + |p0 - a|^2 - R2 = 0
    w = p0 - a
    b = w @ d
    c = (w * w).sum(-1) - R2
    disc = b * b - c
    sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
    # the root on the far side of the antenna's footprint on the plane, nearest to the point's own projection
    l1, l2 = -b - sq, -b + sq
    lx = (x - p0) @ d
    l = np.where(np.abs(l1 - lx) < np.abs(l2 - lx), l1, l2)
    p = p0 + l[..., None] * d
    return np.stack([p @ e1 / spx + nx / 2.0, p @ e2 / spy + ny / 2.0], -1)


def sample(img, ij, order=1):
    """img at fractional pixel coordinates ij [..., 2]: bilinear (order=1) or nearest (0); NaN outside."""
    img = np.asarray(img)
    i, j = ij[..., 0], ij[..., 1]
    ok = (i >= 0) & (j >= 0) & (i <= img.shape[0] - 1) & (j <= img.shape[1] - 1)
    i, j = np.where(ok, i, 0), np.where(ok, j, 0)
    if order == 0:
        out = img[np.rint(i).astype(int), np.rint(j).astype(int)]
    else:
        i0 = np.minimum(np.floor(i).astype(int), img.shape[0] - 2)
        j0 = np.minimum(np.floor(j).astype(int), img.shape[1] - 2)
        fi, fj = i - i0, j - j0
        out = ((1 - fi) * (1 - fj) * img[i0, j0] + fi * (1 - fj) * img[i0 + 1, j0]
               + (1 - fi) * fj * img[i0, j0 + 1] + fi * fj * img[i0 + 1, j0 + 1])
    return np.where(ok, out, np.nan)


def geocode(img, grid, ant, points, order=1):
    """img (any real or complex array on the grid, e.g. amplitude or multilooked power at full resolution) at the
    3-D points [..., 3] of a map grid (local frame of the image; see io.geodetic_to_ecf and io.ecf_to_local), with
    terrain correction through the points' heights."""
    return sample(img, project(points, ant, **grid), order)


def write_geotiff(path, data, transform, crs='EPSG:4326', nodata=np.nan):
    """Single-band GeoTIFF (float32) with an affine transform (rasterio.Affine or its 6 coefficients a, b, c, d, e,
    f: x = a col + b row + c, y = d col + e row + f) in crs. Needs rasterio."""
    import rasterio
    from rasterio.transform import Affine
    tr = transform if isinstance(transform, Affine) else Affine(*transform)
    data = np.asarray(data, np.float32)
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, dtype='float32',
                       crs=crs, transform=tr, nodata=nodata, compress='deflate') as dst:
        dst.write(data, 1)


def write_sicd(path, img, template, transpose=None):
    """Write img as a SICD with the metadata of the template SICD (path or sarpy SICDType), for an image formed on
    the template's grid (read_cphd(..., sicd=template)); transpose as meta['sicd_transpose'] (default: when the
    template's rows run along range). ImageFormAlgo becomes OTHER and Grid.Type PLANE: the pixels are backprojection's
    on the template's image plane, which keep each scatterer's phase, not polar format's. Needs sarpy."""
    from sarpy.io.complex.converter import open_complex
    from sarpy.io.complex.sicd import SICDWriter
    sm = template if not isinstance(template, str) else open_complex(template).sicd_meta
    sm = sm.copy()
    rows, cols = int(sm.ImageData.NumRows), int(sm.ImageData.NumCols)
    a = np.asarray(img)
    if transpose is None:
        transpose = a.shape == (cols, rows) and rows != cols
    a = a.T if transpose else a
    if a.shape != (rows, cols):
        raise ValueError(f'image {a.shape} does not match the template grid {(rows, cols)}')
    sm.ImageFormation.ImageFormAlgo = 'OTHER'
    sm.PFA = None
    sm.Grid.Type = 'PLANE'          # pixels on a plane, linear in its coordinates: the grid backprojection used
    sm.ImageData.PixelType = 'RE32F_IM32F'
    with SICDWriter(path, sm, check_existence=False) as w:
        w.write_chip(a.astype(np.complex64), start_indices=(0, 0))
