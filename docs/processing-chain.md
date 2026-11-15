# Processing chain

The calls from a CPHD file to products, as [`examples/chain.py`](../examples/chain.py) runs them; signatures are in
the [API reference](api.md). Not covered: TOA-domain CPHD, Capella's 2022 dynamic stripmap
convention, speckle filters, coregistration across grids, phase unwrapping,
radiometric calibration beyond the SICD `Radiometric` block of a spotlight collection whose file states `PTRef`
([product-options.md](product-options.md#radiometric-calibration)).

## 1. Read

```python
col, meta = fastsar.io.read_cphd('scene_CPHD.cphd', sicd='scene_SICD.nitf', channel='VV', meta=True)
```

`col` holds the arguments of `form_image`; `meta` adds positions, reference ranges and `notes`
([real-data.md](real-data.md#reading-cphd)). `patches.echoes_to_fx` converts time-domain echoes.

## 2. Form

```python
out = fastsar.form_cphd('scene_CPHD.cphd', sicd='scene_SICD.nitf', backend='cuda')
print(out['mode'], out['spx'], out['spy'], out['notes'])
```

A collection whose scene reference point moves by more than one range resolution is a moving beam (stripmap,
sliding spotlight, dynamic stripmap), formed as a mosaic of patches with a Hann azimuth window
([stripmap-burst.md](stripmap-burst.md#long-apertures-and-arbitrary-tracks); `azimuth_window='taylor'` or
`'uniform'` gives each pixel a Taylor window or a rectangle over its processed band); otherwise it is a spotlight. The
image lies on a ground-plane grid along and across track, over `extent=` meters if given, else the SICD footprint,
else the CPHD image area. Its height is `height=`, else that of the SICD's scene center point, else that of the
CPHD's image-area reference point. The optional SICD also sets the
spacing and the azimuth band; without it a moving beam uses 0.8 of the PRF band (`azimuth_fraction`). Pixel (i, j)
lies at `origin + i*spx*e1 + j*spy*e2` in `read_cphd`'s local frame.

Lower-level calls:

```python
img = fastsar.form_image(**col, backend='cuda')                          # spotlight on the vendor's grid
former = fastsar.ImageFormer(col['ant'], col['fmin'], col['df'], col['S'].shape[1], col['nx'], col['ny'],
                             col['spx'], col['spy'], col['e1'], col['e2'])   # plan and compile once
img = former(col['S'])                                                   # then one call per image
pts = fastsar.io.sicd_points('scene_SICD.nitf', rows, cols, meta)        # any SICD grid, or points on a DEM
img = fastsar.backproject(col['S'], col['ant'], col['fmin'], col['df'], pts, ref=meta['ref'])
```

## 3. Autofocus

`form_cphd(..., autofocus=True)` runs two rounds of phase gradient autofocus on a spotlight (three image
formations) and returns the per-pulse estimate as `out['phase_error']`. `process_cphd` instead corrects the finished
image, of a spotlight or of a moving beam ([product-options.md](product-options.md#autofocus-on-the-image)).
Directly:

```python
img, phi = fastsar.autofocus.autofocus(col['S'], col['ant'], col['fmin'], col['df'], col['nx'], col['ny'],
                                       col['spx'], col['spy'], col['e1'], col['e2'])
S_fixed = col['S'] * np.exp(-1j * phi)[:, None]
```

## 4. Detection, interferometry, polarimetry

```python
from fastsar import products
power = products.multilook(img, 2, 2)                     # mean |img|^2 over 2 by 2 pixels
amp_db = products.to_db(power)
ifg = products.interferogram(img1, img2, la=2, lr=2)      # np.angle(ifg): interferometric phase
coh = products.coherence(img1, img2, 5, 5)                # coherent change detection
rgb = products.pauli(hh, hv, vv)                          # quad polarization on one grid
```

## 5. Geolocate and geocode

```python
lat, lon, h = products.geolocate(out, i, j)                 # pixel (i, j) on the image plane
dem = products.read_dem('dem.tif', offset=geoid)            # DEM GeoTIFF -> dem(lat, lon); geoid height in m
lat, lon, h = products.geolocate(out, i, j, height=dem)     # the terrain point imaged at (i, j)
ij = products.locate(out, lat, lon, h)                      # inverse: fractional (i, j) [..., 2]
geo = products.geocode_image(out, products.multilook(out['image'], 2, 2), height=dem)   # north up, UTM
```

With a height (a number or `dem(lat, lon)`), `geolocate` returns where the scatterer imaged at the pixel lies on
that surface. `geocode_image` takes the amplitude by default, or real or complex data on the image grid or
multilooked from it. Its map can use any rasterio CRS or `'EPSG:4326'`; without a DEM it lies at the ellipsoid
height of the image center ([products.md](products.md#geolocation-and-terrain-correction)).

`products.geocode` samples a `form_image` image at local 3-D points (via `io.geodetic_to_ecf`, `io.ecf_to_local`):

```python
grid = {k: col[k] for k in ('nx', 'ny', 'spx', 'spy', 'e1', 'e2')}    # passing col raises a TypeError
vals = products.geocode(np.abs(img), grid, col['ant'], pts)
```

## 6. Export

```python
products.write_geotiff('power.tif', **geo)                  # float32 or complex64 bands
products.write_sicd('scene.nitf', out)                      # form_cphd output, metadata built from it
products.write_sicd('out.nitf', img, 'scene_SICD.nitf')    # image on the template SICD's grid
```

The SICD metadata is described in [products.md](products.md#sicd-output).

## Quick look

When a look at the data is wanted, `quicklook` forms a picture of a spotlight collection by polar format from the
part of the file its resolution needs:

```python
out = fastsar.quicklook('scene_CPHD.cphd', 'look.png')                    # 2048 pixels on the longer side
out = fastsar.quicklook('scene_CPHD.cphd', 'full.png', resolution='full') # the whole scene, full resolution
out = fastsar.quicklook('scene_CPHD.cphd', size=4096, correct=False)      # the fastest picture of that size
out = fastsar.quicklook('scene_CPHD.cphd', 'look.nitf', resolution=1.0)   # 1 m, the complex image as a SICD
out['display'], out['corners'], out['cut'], out['notes'], out['timing']
```

The header and a few hundred per-pulse parameters decide the cut: the central run of frequency samples and the
central run of pulses whose bands give the pixel asked for, never more than the collection holds. Only those are
read (`read_cphd(pulses=, samples=)`). The picture is the tasked area on the ground, on the grid of
`form_cphd(axes='los')`: the range axis along the mid-aperture line of sight projected on the ground, square pixels
of 0.96 of the reciprocal of the band on each axis. It is detected, remapped (`remap='nrl'`, as a SIDD from
`process_cphd`) and laid out shadows down. `resolution='full'` is the whole scene at the finest resolution the
collection holds on both axes.

The grid is on the ground because the slant plane of a squinted pass is twisted about the line of sight: a
rectangle of a slant plane that holds a 5 km ground square under a twist of 54 degrees spans 6.7 km of Doppler,
more than the 4.2 km the pulse rate leaves unambiguous, while at any one range the scene spans 2.8 km. Polar
format on the ground axes keeps each range line inside the unambiguous width. The samples then fill a sheared
part of the wavenumber raster, which the stages follow ([algorithms.md](algorithms.md#quick-look)).

`correct=True` (the default) removes polar format's displacement of scatterers; `correct=False` skips that step.
At a resolution finer than about a meter over several kilometers polar format defocuses the corners of the
picture; the product estimates by how much (`corner_error`), a note says so, and `form_cphd` is the tool for that
image. `looks=n` averages the power of n by n images of the same pixels, formed from sub-apertures and sub-bands
side by side, for less speckle at n by n times the bytes and the formation time. `output` is a `.png` file,
written with the standard library alone, or a `.nitf` file, the complex image as a ground-plane SICD. A
full-resolution picture of a large spotlight can pass 179 megapixels, the size above which Pillow refuses to open
an image unless `PIL.Image.MAX_IMAGE_PIXELS` is raised (or set to `None`); `fastsar.quick.read_png` reads any size.

Every backend runs polar format's stages written for it: `'cpu'` in C++ on all cores, `'cuda'` in CUDA kernels
and cuFFT, `'jax'` and `'tpu'` in float32 planes on JAX's device. The device paths work in blocks sized from the
device's free memory, and what does not fit there stays on the host and crosses in blocks, with a note.
`backend='auto'` takes the device that forms the cut soonest by its size: the CPU for a small cut, without
starting a device. `FASTSAR_QUICKLOOK_JAX=ffbp` forms the same cut on the same grid by the factorized former on
the jax and tpu backends.

A moving-beam collection (stripmap, sliding spotlight) is refused with a `ValueError` that names `process_cphd`:
polar format assumes one scene reference point, and a moving beam's azimuth resolution is set per pixel by the
processed Doppler band, not by a run of pulses. Times, bytes read and accuracy are in
[performance.md](performance.md#quick-look) and [algorithms.md](algorithms.md#quick-look).
