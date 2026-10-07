# Products: interferometry, geolocation and output

Methods and test results; the calls are in [processing-chain.md](processing-chain.md).

## Interferometry and change detection

Two passes formed onto one grid are coregistered by construction, and a scatterer on the grid surface has zero
interferometric phase, so no flat-earth or topographic phase needs removing. `pauli` gives
double-bounce, volume and surface powers. Test results (`tests/test_insar.py`, X-band clutter at 5 km):

- change detection: median coherence 0.981 on unchanged ground (true value 0.98) and 0.275 inside disturbed areas
  and a vehicle track (7 by 7 windows)
- passes 3.5 m apart vertically (height of ambiguity 22.3 m), a 6 m block on flat ground, formed onto the ground
  plane: the phase converts to 5.98 m on the block (at its layover position) and 0.00 m on flat ground (coherence
  0.999); formed by exact backprojection onto the terrain, median phase 0.008 rad on the block and 0.003 rad on
  flat ground

## Geolocation and terrain correction

A point off the image plane appears where the plane has its range and Doppler cone angle at the aperture center.
`project` solves this in closed form, so terrain correction is exact under that model. It also holds for moving-beam
mosaics on a straight track, and on an orbit while the track over the image is close to a line. `geolocate` inverts
it by Newton iteration. `geocode_image` projects every map pixel center into the image and samples there, NaN
outside. DEM heights are often above the geoid (Copernicus DEM: EGM2008); `read_dem`'s `offset` converts them to
ellipsoid heights.

Test results:

- `tests/test_products.py`: targets 0 to 30 m above the image plane appear within 0.5 cm of their projection, with
  layover up to 21 m.
- `tests/test_chain.py` (X-band spotlight at 6 km written as CPHD, targets on terrain up to 9.6 m off the grid
  plane, 0.4 m pixels): peaks lie 0.4 cm from `locate`'s prediction; geolocated on the DEM they lie 0.4 cm from the
  truth, against up to 6.7 m of layover; round trips agree to 1e-8 pixels; UTM and latitude/longitude GeoTIFFs place
  each target to 1e-4 pixels, amplitude peaks within 0.16 m; a stripmap mosaic geolocates to 0.4 cm.
- `tests/test_chain.py`, `form_cphd(..., autofocus=True)` on a quadratic plus cubic phase error (8 rad at the
  aperture ends): peaks rise from 0.63 to 0.999 of the focused ones, residual 0.034 rad rms.

Real-data geolocation is in [real-data.md](real-data.md#geolocation).

## Amplitude

Images are not normalized: a scatterer of sample amplitude a peaks at a times the window sum (a Np K for the mean-1
Taylor windows; 0.996 Np K measured in `tests/test_chain.py`), and noise of power s^2 per sample gives s^2 times the
sum of the squared window. CPHD has no radiometric calibration; beta0 and sigma0 need the vendor's constant.

## SICD output

`write_sicd(path, out)` builds the metadata with `sicd_meta`: a PLANE ground grid, rows along range away from the
radar, columns along track, the aperture position as a polynomial in time and one center-of-aperture time, so
SICD's projection is the model of `geolocate`. Impulse response parameters are nominal; there is no Radiometric
block. ImageFormAlgo is OTHER, which sarpy's `is_valid` rejects; sarkit's consistency check passes. In
`tests/test_chain.py` the round trip keeps every pixel, and sarpy's projection of the peaks agrees with
`geolocate` to better than a micrometer.

`write_sicd(path, img, template)` keeps a vendor SICD's metadata, with ImageFormAlgo OTHER and Grid.Type PLANE. A
round trip of the Umbra Panama SICD keeps every pixel and passes sarpy's validity check; its projection matches the
vendor's at the center and departs by up to about 6 m at the corners, where the vendor's polar-format image carries
the distortion its own model accounts for.
