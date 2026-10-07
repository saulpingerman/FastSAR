# Products: interferometry, geocoding and output

`fastsar.products` turns formed images into products:

```python
from fastsar import products
amp_db = products.to_db(products.multilook(img, 2, 2))
ifg = products.interferogram(img1, img2, la=2, lr=2)       # np.angle(ifg) is the interferometric phase
coh = products.coherence(img1, img2, 5, 5)                 # coherent change detection statistic
rgb = products.pauli(hh, hv, vv)                           # quad-polarization Pauli decomposition
vals = products.geocode(img, grid, ant, points)            # image sampled at 3-D points (a map grid, a DEM)
products.write_sicd('out.nitf', img, 'vendor_SICD.nitf')   # image formed on the vendor's grid by read_cphd
products.write_geotiff('out.tif', data, transform)          # needs rasterio
```

`grid` is a dict of exactly `nx, ny, spx, spy, e1, e2`, the centered grid of `form_image` that the image was formed
on, and `ant` the antenna positions in the same local frame. The output of `read_cphd` carries these keys but also
`ant`, so pass `{k: col[k] for k in ('nx', 'ny', 'spx', 'spy', 'e1', 'e2')}`, not `col` itself. For a `form_cphd`
image, whose origin is pixel (0, 0), see [processing-chain.md](processing-chain.md#6-geocode).

## Interferometry and change detection

Images of two passes formed onto the same grid or the same points are coregistered by construction, and a
scatterer that lies on the grid surface contributes zero interferometric phase, so no flat-earth or topographic
phase needs removing. `interferogram` gives the multilooked interferogram and `coherence` the sample coherence over
moving windows. `pauli` stacks the double-bounce, volume and surface powers of a quad-polarization set formed on
one grid.

Test results (`tests/test_insar.py`, simulated X-band clutter at 5 km; needs finufft):

- change detection, same geometry, coherence 0.98 outside two disturbed areas and a vehicle track: median sample
  coherence 0.981 on unchanged ground and 0.275 inside the changes (7 by 7 windows)
- height, passes 3.5 m apart vertically (height of ambiguity 22.3 m), a 6 m block on flat ground, both formed onto
  the ground plane: the phase converts to 5.98 m on the block (at its layover position) and 0.00 m on the flat
  ground, with coherence 0.999 there
- the same pair formed by exact backprojection onto the terrain itself: median phase 0.008 rad on the block top
  and 0.003 rad on the flat ground

Interferometric and polarimetric products are tested on simulated data only.

## Geocoding and terrain correction

A point off the image plane appears where the plane has its range and Doppler cone angle at the aperture center.
`project` solves that pair of conditions in closed form, so terrain correction is exact under that model, and
`geocode` samples the image at the projected positions. Test result (`tests/test_products.py`, simulated): on an
image formed on the ground plane, targets 0 to 30 m above it appear within 0.5 cm of their projection, with
layover up to 21 m. For geolocation on the real Panama collection, see [real-data.md](real-data.md#geolocation).

## SICD and GeoTIFF output

`write_sicd` writes an image formed on a vendor SICD's grid (`read_cphd(..., sicd=...)`) with that SICD's
metadata, ImageFormAlgo OTHER and Grid.Type PLANE: the pixels are backprojection's on the template's image plane.
A round trip of the Umbra Panama SICD keeps every pixel and passes sarpy's validity check. Its projection of the
center pixel agrees with the vendor's exactly and departs from it by up to about 6 m at the corners, where the
vendor's polar-format image is displaced by the distortion its own projection model accounts for.

`write_geotiff` needs rasterio, `write_sicd` needs sarpy.
