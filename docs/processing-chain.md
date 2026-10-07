# Processing chain

This page lists the steps from a phase history to a delivered product, with the FastSAR call for each. The
[API reference](api.md) gives the signatures; the other pages explain the methods.

| Step | FastSAR call | Not covered yet |
|---|---|---|
| 1. Read | `io.read_cphd`, `patches.echoes_to_fx` | TOA-domain CPHD, raw Level-0 formats |
| 2. Form | `form_cphd`, `form_image`, `ImageFormer`, `backproject`, `patches.form_mosaic` | Capella's 2022 dynamic stripmap delay convention |
| 3. Autofocus | `autofocus.autofocus`, `autofocus.pga` | autofocus of moving-beam mosaics |
| 4. Multilook and detect | `products.multilook`, `products.to_db` | speckle filters |
| 5. Interferometry, change detection, polarimetry | `products.interferogram`, `products.coherence`, `products.pauli` | coregistration of images on different grids, phase unwrapping |
| 6. Geocode | `io.local_to_ecf`, `io.ecf_to_geodetic`, `products.geocode` | DEM readers, terrain correction of moving-beam mosaics, radiometric calibration |
| 7. Export | `products.write_sicd`, `products.write_geotiff` | SICD output for `form_cphd`'s ground-plane grid |

## 1. Read

`io.read_cphd` reads one channel of a frequency-domain CPHD (needs sarpy) and returns the arguments of
`form_image`. With `meta=True` it also returns the transmit and receive positions, the per-pulse reference ranges,
the local frame and a list of notes on what it did to the data.

```python
col, meta = fastsar.io.read_cphd('scene_CPHD.cphd', sicd='scene_SICD.nitf', channel='VV', meta=True)
print(col['S'].shape, meta['notes'])         # [pulses, samples] complex64
```

A time-of-arrival (TOA) CPHD raises a `ValueError`. Raw Level-0 products (Sentinel-1, NISAR, ALOS) have no
decoder. Time-domain echoes that follow the conventions of `fastsar.stripmap` (simulated, or decoded elsewhere)
become a frequency-domain phase history with per-pulse reference ranges by `patches.echoes_to_fx(raw, p)`, where
`p` is a `stripmap.StripParams` with the chirp and the fast-time grid.

## 2. Form

`form_cphd` goes from the file to an image in one call. It treats a collection as a moving beam when the scene
reference point moves by more than one range resolution, else as a spotlight.

```python
out = fastsar.form_cphd('scene_CPHD.cphd', sicd='scene_SICD.nitf', backend='cuda')
img, origin, e1, e2 = out['image'], out['origin'], out['e1'], out['e2']
print(out['mode'], out['spx'], out['spy'], out['notes'])
```

The grid is a ground plane along and across track at the height of the scene reference point (`height=` sets
another), over the SICD footprint, the CPHD image area, or `extent=` meters around the scene center. The SICD is
optional; it sets the footprint, the spacing and the processed azimuth band. A spotlight is one factorized
backprojection. A moving beam (stripmap, sliding spotlight) is a mosaic of patches whose Hann azimuth window is
applied inside the final stage of each patch's factorized backprojection.

For control over each step, form a spotlight on the vendor's grid, or any points by exact backprojection:

```python
img = fastsar.form_image(**col, precision='float16', backend='cuda')     # col from read_cphd(..., sicd=...)
pts = fastsar.io.sicd_points('scene_SICD.nitf', rows, cols, meta)        # any SICD grid, or a DEM surface
img = fastsar.backproject(col['S'], col['ant'], col['fmin'], col['df'], pts, ref=meta['ref'])
```

For a moving beam on a grid of one's own, `patches.form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2,
beam=..., awin='hann')` is the call `form_cphd` makes ([stripmap-burst.md](stripmap-burst.md#long-apertures-and-arbitrary-tracks)).
Simulated straight-track stripmap and burst echoes also focus by omega-k or range-Doppler
(`stripmap.focus_stripmap`, `burst.focus_burst`).

## 3. Autofocus

Phase gradient autofocus estimates a phase error per pulse from a factorized backprojection image and returns
the corrected image and the estimate:

```python
img, phi = fastsar.autofocus.autofocus(col['S'], col['ant'], col['fmin'], col['df'], col['nx'], col['ny'],
                                       col['spx'], col['spy'], col['e1'], col['e2'], backend='cpu')
S_fixed = col['S'] * np.exp(-1j * phi)[:, None]
```

It works on one spotlight aperture and has been tested on simulated images only
([algorithms.md](algorithms.md#autofocus)).

## 4. Multilook and detect

```python
from fastsar import products
power = products.multilook(img, 2, 2)        # |img|^2 over 2 by 2 boxes, decimated
amp_db = products.to_db(power)
```

## 5. Interferometry, change detection and polarimetry

Images of two passes formed onto one grid are coregistered by construction ([products.md](products.md)).

```python
ifg = products.interferogram(img1, img2, la=2, lr=2)      # np.angle(ifg): interferometric phase
coh = products.coherence(img1, img2, 5, 5)                # coherent change detection
rgb = products.pauli(hh, hv, vv)                          # one grid, quad polarization
```

## 6. Geocode

A pixel of a `form_cphd` image lies on its ground plane at a known local position, so map coordinates follow from
the frame conversions:

```python
i, j = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
xyz = out['origin'] + (i * out['spx'])[..., None] * out['e1'] + (j * out['spy'])[..., None] * out['e2']
lat, lon, h = fastsar.io.ecf_to_geodetic(fastsar.io.local_to_ecf(xyz, out['meta']))     # degrees, meters
```

A scatterer off the plane appears displaced (layover). `products.geocode` corrects for it by sampling the image at
the range-Doppler projection of each map point. Its grid is centered, as `form_image`'s is, so a `form_cphd`
spotlight image is shifted to its center first:

```python
m, (nx, ny) = out['meta'], img.shape
c = out['origin'] + nx / 2 * out['spx'] * out['e1'] + ny / 2 * out['spy'] * out['e2']
grid = dict(nx=nx, ny=ny, spx=out['spx'], spy=out['spy'], e1=out['e1'], e2=out['e2'])
pts = fastsar.io.ecf_to_local(fastsar.io.geodetic_to_ecf(lat_map, lon_map, dem_h), m)   # map grid on a DEM
vals = products.geocode(np.abs(img), grid, 0.5 * (m['tx'] + m['rcv']) - c, pts - c)
```

An image from `read_cphd(..., sicd=...)` and `form_image` needs no shift, but the grid dict must hold only the grid
keys (passing `col` itself repeats `ant` and raises a `TypeError`):

```python
grid = {k: col[k] for k in ('nx', 'ny', 'spx', 'spy', 'e1', 'e2')}
vals = products.geocode(np.abs(img), grid, col['ant'], pts)
```

The projection uses one mid-aperture position, which models a spotlight; FastSAR has no layover model for a
moving-beam mosaic. FastSAR reads no DEM files and does no radiometric calibration.

## 7. Export

```python
products.write_sicd('out.nitf', img, 'scene_SICD.nitf')                 # image on the template's grid
products.write_geotiff('out.tif', vals, (dlon, 0, lon0, 0, -dlat, lat0)) # map grid, EPSG:4326
np.save('out.npy', img)
```

`write_sicd` (needs sarpy) takes an image formed on the template SICD's grid by `read_cphd(..., sicd=...)` and
`form_image`; it writes ImageFormAlgo OTHER and Grid.Type PLANE. A `form_cphd` image is not on the vendor's grid
and has no SICD writer yet. `write_geotiff` (needs rasterio) writes one float32 band with an affine transform.
