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

`grid` is the dict of `nx, ny, spx, spy, e1, e2` the image was formed on (the output of `read_cphd` carries
them), and `ant` the antenna positions in the same local frame. The output of `form_cphd` goes to products
directly; see [From form_cphd to map and SICD](#from-form_cphd-to-map-and-sicd).

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

## From form_cphd to map and SICD

The dict `form_cphd` returns carries everything the products need: the grid, the antenna track and the
collection's metadata.

```python
out = fastsar.form_cphd('scene_CPHD.cphd', autofocus=True)          # autofocus: spotlight only
lat, lon, h = products.geolocate(out, i, j)                         # pixel (i, j) on the image plane
lat, lon, h = products.geolocate(out, i, j, height=dem)             # the point on the terrain imaged there
i, j = products.locate(out, lat, lon, h)                            # where a ground point appears
dem = products.read_dem('dem.tif', offset=geoid)                    # DEM GeoTIFF -> dem(lat, lon)
geo = products.geocode_image(out, products.multilook(out['image'], 2, 2), height=dem)   # UTM, north up
products.write_geotiff('power.tif', **geo)
products.write_sicd('scene.nitf', out)                              # complex, with its geometry
```

`geolocate` with `height=None` gives the pixel's own position on the image plane. With a height (a number, or a
function of latitude and longitude such as the one `read_dem` returns) it gives the point on that surface whose
range and Doppler cone angle at the aperture center match the pixel's, which is where a scatterer imaged at that
pixel actually lies. `locate` is its inverse. `geocode_image` projects the center of every pixel of a north-up map
grid (UTM by default, or any CRS rasterio knows, or `EPSG:4326`) through its range and Doppler onto the image
plane and samples the data there, bilinear or nearest; it takes the full-resolution image, anything computed from
it on the same grid, or data multilooked from it by integer factors, real or complex. Without a DEM the map lies at
the ellipsoid height of the image center. DEM heights are often above the geoid (Copernicus DEM: EGM2008);
`read_dem`'s offset adds the geoid's height at the scene.

`write_sicd(path, out)` builds the metadata from the CPHD and the grid (`sicd_meta`): a PLANE grid on the
ground plane with rows along range away from the radar and columns along the track, the aperture position as a
polynomial in time, and one center-of-aperture time for the whole image. SICD's projection of a pixel is then the
same range-Doppler model as `geolocate` and `locate`. ImageFormAlgo is OTHER, which sarpy's `is_valid` reports as
not valid for any SICD; the files pass sarkit's SICD consistency check without errors.

Amplitude. `form_image` and `form_cphd` sum the phase history without normalization: a point scatterer whose
samples have amplitude a peaks at a times the sum of the window over pulses and samples (the Taylor windows have
mean 1, so a Np K), and noise of power s^2 per sample gives a mean pixel power s^2 times the sum of the squared
window. CPHD carries no radiometric calibration, so pixel values are in the units of the phase history; beta0 and
sigma0 need the vendor's calibration constant, and the SICD carries no Radiometric block.

`examples/chain.py` runs the chain from a CPHD file to a SICD, a dB GeoTIFF and a PNG quicklook (`--simulate`
writes a simulated CPHD with `sim.write_cphd` first).

Test results (`tests/test_chain.py`, a simulated X-band spotlight at 6 km written as CPHD, targets on sloped
terrain up to 9.6 m above or below the grid plane, 0.4 m pixels):

- a unit point target on the grid plane peaks at 0.996 Np K
- target peaks lie 0.4 cm from where `locate` puts them; geolocated at their heights or on the DEM (as a
  function or read from a DEM GeoTIFF), 0.4 cm from their true positions, against up to 6.7 m on the image plane
  (layover); pixel to ground to pixel round trips agree to 1e-8 pixels
- map GeoTIFFs in UTM and latitude/longitude: the pixel coordinates geocoded with the DEM hold each target's own
  pixel at its map position to 1e-4 pixels; amplitude peaks in the map lie within 0.16 m of the true positions
  (0.4 m map pixels, bilinear amplitude)
- SICD round trip: pixels identical; sarpy's projection of the target peaks agrees with `geolocate` to better
  than a micrometer and with the truth to 0.4 cm; no sarkit consistency errors
- `form_cphd(..., autofocus=True)` on a quadratic plus cubic phase error (8 rad at the aperture ends): peaks from
  0.63 to 0.999 of the focused ones, residual 0.034 rad rms
- a stripmap collection (moving scene reference point, formed as a mosaic): geolocated peaks 0.4 cm from the
  truth

## SICD and GeoTIFF output

`write_sicd` writes an image formed on a vendor SICD's grid (`read_cphd(..., sicd=...)`) with that SICD's
metadata, ImageFormAlgo OTHER and Grid.Type PLANE: the pixels are backprojection's on the template's image plane.
A round trip of the Umbra Panama SICD keeps every pixel and passes sarpy's validity check. Its projection of the
center pixel agrees with the vendor's exactly and departs from it by up to about 6 m at the corners, where the
vendor's polar-format image is displaced by the distortion its own projection model accounts for.

`write_geotiff` writes float32 or, for complex data, complex64 (GDAL CFloat32) bands; it needs rasterio, and
`write_sicd` needs sarpy.
