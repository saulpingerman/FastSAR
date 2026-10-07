# Processing chain

The calls from a CPHD file to products, as [`examples/chain.py`](../examples/chain.py) runs them; signatures are in
the [API reference](api.md). Not covered: TOA-domain CPHD, raw Level-0 data, Capella's 2022 dynamic stripmap
convention, autofocus of moving-beam mosaics, speckle filters, coregistration across grids, phase unwrapping,
radiometric calibration.

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
([stripmap-burst.md](stripmap-burst.md#long-apertures-and-arbitrary-tracks)); otherwise it is a spotlight. The
image lies on a ground-plane grid along and across track at the scene reference point's height (or `height=`),
over `extent=` meters if given, else the SICD footprint, else the CPHD image area. The optional SICD also sets the
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
formations) and returns the per-pulse estimate as `out['phase_error']`. Directly:

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
i, j = products.locate(out, lat, lon, h)                    # inverse
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
