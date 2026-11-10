# Product options: area, plane, autofocus, weighting, SICD and SIDD

`process_cphd` goes from a CPHD file to a product on disk in one call and exposes the choices a production
processor offers. `form_cphd` remains the formation step underneath it and keeps its own defaults.

```python
import fastsar
fastsar.process_cphd('scene_CPHD.cphd', 'scene_SICD.nitf')                    # complex image, slant plane
fastsar.process_cphd('scene_CPHD.cphd', 'scene_SIDD.nitf', product='sidd')    # detected image, ground plane
out = fastsar.process_cphd('scene_CPHD.cphd', product='sidd', area='black', remap='density')   # nothing written
```

The options and their defaults:

- `product`: `'sicd'` (complex image, the default) or `'sidd'` (detected, remapped 8-bit image).
- `area`: `'tasked'` (default), `'tasked_quality'`, an RNIIRS value, or `'black'`.
- `plane`: `'slant'` or `'ground'`. The default is the slant plane for a SICD and the ground plane for a SIDD.
- `autofocus`: `'quadratic'` (default), `'pga'`, `'spatially_variant'`, `'spatially_variant_pga'` or `False`.
- `weighting`: `'taylor'`, `'uniform'` or `'sva'`. The default is Taylor for a SICD and spatially variant
  apodization for a SIDD.
- `remap`: for a SIDD, one of the remaps of the NGA MATLAB SAR Toolbox (`'nrl'` by default).
- `spacing`: the product's pixel spacing. The default follows the bandwidth of the data.

A moving-beam collection (stripmap, sliding spotlight) is formed on the ground plane on axes along the track.
The slant plane, autofocus, apodization and the quality areas apply to spotlight collections.

## Planes

FastSAR forms every spotlight image on a ground plane, where backprojection is exact for any track. A planar
slant grid is exact only for a straight track: on an orbit, the range history of a slant-plane pixel departs from
that of the ground point it stands for, and a long aperture defocuses. The slant-plane product is therefore the
ground image resampled.

The slant plane is the plane through the scene center point that holds the antenna's position and velocity at the
center of aperture. A pixel at azimuth a and range r shows the ground point that has, at the center of aperture,
the range and range rate of the slant-plane point at (a, r). This is the projection of a SICD with `Grid.Type`
PLANE and `ImagePlane` SLANT, so a SICD reader's own projection of the product agrees with the pixels. The maps
both ways are closed forms (`planes.ground_to_slant`, `planes.slant_to_ground`).

The resampling works on ground axes along the line of sight (`form_cphd(axes='los')`). After the carrier is
removed (`planes.deramp`, exact at every pixel), the image's spectrum is a parallelogram whose extent across the
line of sight is the azimuth band alone. Each line across the line of sight is then band limited however the
spectrum wraps along it. `planes.to_slant` interpolates along that axis first, which leaves a spectrum that is a
rectangle in azimuth and ground range, and then along the line of sight. Each pass oversamples by two with an FFT
and applies an 8-tap Kaiser sinc from a table. `planes.sampling` gives the pixel spacings the ground image needs
from the collection's own spectrum at points of the scene; `process_cphd` uses 0.96 of their reciprocal.

A ground-plane SICD is the image as formed, at baseband, on axes along the line of sight. Its rows run along
ground range away from the radar whatever the squint.

In `tests/test_planes.py` and `tests/test_process.py`, on a simulated collection with a 20 to 25 degree squint
and a curved track, flown to either side, each target falls within 0.05 pixel of the position sarpy's projection
of the written SICD gives, on both planes, and its phase is within 0.05 rad of SICD's convention.

## Weighting and spatially variant apodization

`'taylor'` applies a Taylor window (nbar 4, 35 dB) to the phase history along frequency and along pulses.
`'uniform'` applies none. `'sva'` forms the image unweighted and applies spatially variant apodization
(`weighting.sva`) on the slant plane: at every pixel it picks, between the unweighted aperture and a Hann window,
the cosine-on-pedestal weighting that gives the least output. Main lobes keep the unweighted width and sidelobes
are removed where the spectrum is a rectangle sampled at an integer multiple of Nyquist. Two steps make it so.
`weighting.keystone` drops, at each frequency, the pulses outside the azimuth band the lowest frequency reaches
(the azimuth band of a pulse fan grows with frequency; about 3% of the samples go on an X-band collection with
600 MHz of bandwidth). The slant image is then resampled at exactly twice Nyquist of the rectangle that is left.
The result is not band limited and not linear in the scene, so it suits detection and display. A SICD with
`weighting='sva'` is allowed on the slant plane and records the weighting as SVA.

## Display orientation and SIDD output

A SIDD from `process_cphd` is detected on the slant plane (after apodization, when that is the weighting) and
projected with a cubic kernel to square ground pixels on axes along the line of sight, so that range runs straight
down the rows. The default pixel is 0.7 of the finer ground resolution.

A detected image is laid out with the radar at the top. Ground range increases down the rows, so shadows, which
fall away from the radar, point down, and the view is the one from above rather than its mirror image (row x col
points up). `fastsar.sidd.shadows_down(out)` applies this rule to a `form_cphd` product without resampling. Of the
eight layouts a transpose and two flips can give, four are views from above, and the function takes the one whose
rows lie nearest the range direction at the image center, judged from the mid-aperture antenna position. It
returns the reoriented array (a view), the local position of its first pixel, the row and column unit vectors and
spacings, the transform applied, and `angle`, the lean that remains between the rows and the range direction.
The lean is zero on a grid that follows the line of sight and can reach 45 degrees on a squinted one, where only
a resampling would remove it. `orient(array, transform)` lays another array, such as a mask, out the same way.
Slant-plane products (`out['plane'] == 'slant'`) follow the same rule about the plane's upward normal.

```python
disp = fastsar.sidd.to_display(out)                     # uint8, NRL remap, shadows down
fastsar.sidd.write_sidd('image.nitf', out)              # the same pixels as a SIDD 3.0 NITF
```

`to_display(out, remap='nrl', valid=None, shadows_down=True, **remap_args)` detects a complex image, applies one
of the remaps of `fastsar.remap` (`nrl`, `density`, `brighter`, `darker`, `highcontrast`, `pedf`, `log`, or a
function) and lays the result out. Pixels that are exactly zero are taken as unprocessed. They stay zero and are
left out of the statistics the remap draws from the image; `valid`, or a mask in `out['valid']`, replaces that
default. A uint8 image is taken as display values already.

`write_sidd(path, out, *, remap='nrl', remap_args=None, valid=None, shadows_down=True, version=3,
product_name=None, classification='UNCLASSIFIED', sicd=None)` writes those pixels as a MONO8I SIDD through sarpy
and returns the SIDD structure. It accepts ground-plane products only. The metadata are:

- Measurement: a plane projection (reference point at the center pixel, sample spacings, row and column unit
  vectors in ECF), the aperture position polynomial and one center-of-aperture time, so the SIDD's own projection
  is the range-Doppler model of `geolocate`; the pixel footprint; and ValidData, the convex hull of the valid
  pixels.
- GeoData: the image corners and the valid-data polygon in latitude and longitude.
- Display: the remap's name and the shadow direction (DOWN when shadows follow the rows to within one degree,
  otherwise UP, LEFT, RIGHT or ARBITRARY).
- ExploitationFeatures: azimuth, graze, slope, tilt, squint and Doppler cone angles, the shadow, layover,
  multipath, ground-track and north directions in the pixel frame, and the resolution, all computed by sarpy
  from the SICD metadata of the same product (`sicd_meta(out)`, or the `sicd` argument, which is then stored in
  the file as well).
- ProductCreation: FastSAR and its version as the processor.

SIDD 3.0 is written by default and 2.0 on request. Two NITF header fields that sarpy 2.1.1 fills differently from
the standard are corrected: the specification version and date in the XML segment's subheader, and the image
segment's corner coordinates, which sarpy truncates to the arcsecond where the standard rounds.

In `tests/test_sidd.py`, on a simulated collection flown both ways, the round trip keeps every pixel, the SIDD's
projection of the measured peaks falls within 5 mm (0.01 pixel) of the true target positions, an asymmetric
target pattern appears with range down the rows and unmirrored for either look direction, and sarkit's
consistency check passes without errors or warnings for SIDD 3.0, SIDD 2.0, a masked product and an embedded SICD.
