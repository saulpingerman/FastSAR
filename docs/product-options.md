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

## Processed area by image quality

`fastsar.rgiqe` predicts the signal-to-noise ratio and the RNIIRS rating of the image a CPHD collection forms, at
any ground point, and turns the prediction into the region worth processing. The radar generalized image quality
equation rates an image by its information density:

```
RNIIRS = 3.4761 + 0.4357 log2(bandwidth_area * log2(1 + SNR))
```

`bandwidth_area` is the ground-plane area of the spatial-frequency support in cycles^2 / m^2 (range bandwidth times
azimuth bandwidth times the cosine of the slope angle). Below a transition density the rating follows the tangent
line through the origin. `rniirs`, `information_density`, `snr_to_rniirs` and `rniirs_to_snr` reproduce sarpy 2.1.1.

`Radiometry` supplies the SNR. For a uniformly weighted image formed from N phase-history samples,

```
NESZ(x) = PN * bandwidth_area / (N * PTRef * G(x))
SNR(x)  = sigma0 * G(x) / (NESZ_ref + sigma0 * sum_k G(x_k))
```

where PN is the noise power per sample, PTRef the per-sample power of a 1 m^2 target at the scene reference point,
G(x) the two-way antenna gain relative to that point (averaged over the aperture, with the R^-4 spreading loss), and
x_k the first two azimuth ambiguities of x on either side, whose echoes count as noise. Points whose echo falls
outside the receive window TOA1..TOA2 get an SNR of zero. The reference sigma0 defaults to 0 dB for a co-polarized
channel and -6.02 dB for a cross-polarized one, as in sarpy.

### Noise level

The level at the scene reference point comes from the first source available (`level_source`; `notes` lists what
was assumed). A measurement reads 512 pulses and fits the noise floor in the extended delay swath beside the scene
when the file saves one; otherwise it takes the lowest cell of a range-Doppler power map.

1. `nesz_db` given by the caller (`'argument'`).
2. The Radiometric block of a SICD of the same collection (`'sicd'`).
3. PTRef and PNRef of the CPHD file (`'cphd'`).
4. PTRef with the noise power measured from the phase history (`'cphd+measured'`).
5. The measured clutter-to-noise ratio with an assumed mean scene sigma0, `scene_sigma0_db` (`'uncalibrated'`).

### Areas

`area_mask` returns a mask over a ground grid and `area_bounds` its bounding rectangle along two ground axes. Each
area is the connected region around the scene reference point, joined with the tasked polygon:

- `'tasked'`: the tasked polygon.
- `'tasked_quality'`: where the RNIIRS is at least the tasked one (given, from the SICD, or the model's minimum).
- a number: where the predicted RNIIRS is at least that number.
- `'black'`: where the SNR is at least 1, with the range swath opened to the extended window TOAE1..TOAE2.

### Example

```python
from fastsar import io, rgiqe
data, meta = io.read_collection('collect.cphd', meta=True)
rad = rgiqe.Radiometry('collect.cphd', meta)                  # see rad.level_source and rad.notes
c = rad.ref_point[None]                                       # points are [n, 3] in the local frame of meta
print(rad.nesz_db(c, bw_area, samples), rad.rniirs(c, bw_area, samples))
lo1, hi1, lo2, hi2 = rgiqe.area_bounds(rad, 'black', c[0], e1, e2, bw_area, samples, tasked=corners)
```

### Limitations

- The image is assumed uniformly weighted and not corrected for the antenna pattern.
- Gain polynomials are trusted to 15 dB one-way below their peak; beyond that no area criterion is met.
- An `'uncalibrated'` level rests on the assumed scene sigma0; a range-Doppler noise level is an upper bound.
- Ambiguities assume a uniform scene and are placed with the mid-aperture geometry; range ambiguities are ignored.
- A low RNIIRS threshold at a large bandwidth area needs an SNR below 1, and its area can exceed `'black'`.

`process_cphd(..., area=)` takes the same four kinds. The image is formed over the bounding rectangle of the area
on the ground axes, with one impulse response everywhere and a noise-equivalent sigma zero that rises toward the
edges, and the pixels outside the area are set to 0. A file that describes no antenna pattern gives the signal no
edge to follow, and only the tasked area can then be processed.

## Autofocus on the image

`fastsar.autofocus.image_autofocus(img, mode)` removes an azimuth phase error from a finished image, without
the phase history, and returns `(img2, info)`.

**Input convention.** `img` is a complex baseband spotlight image whose axis 0 is azimuth: a slant-plane image
from `planes.to_slant`, or a ground-plane image formed with `axes='los'` after `planes.deramp`. The error model
is one phase function of azimuth spatial frequency shared by all columns: the FFT along axis 0 of every column
is multiplied by `exp(1j * phi(k))`. Only transforms along axis 0 are taken, so the spectrum along axis 1 may
be wrapped or sheared. The occupied azimuth band is detected from the data, which covers oversampled images.
Pixels that are exactly zero stay exactly zero, and columns that are mostly zero are left out of the estimate.

**Modes.** `None` or `False` returns the image as it came.

- `'quadratic'`: one quadratic error for the whole image, estimated by map drift (the shift between the
  detected images of the two halves of the band). It is the fastest mode. `info['edge_rad']` is the
  coefficient in radians at the band edge and `info['phi']` the phase removed per FFT bin. A phase error on
  pulse `p` appears with the same sign and scale at that pulse's azimuth frequency.
- `'pga'`: one error of any shape for the whole image, estimated by phase gradient autofocus on the brightest
  columns. Use it when the error has cubic or higher terms; a quadratic in pulse index is such a case when the
  collection is squinted or the track is curved. The constant and linear parts of the error are not estimated.
- `'spatially_variant'`: the quadratic coefficient is estimated on overlapping blocks, fitted over the scene by
  a low-order 2-D polynomial, and removed with a correction that follows range column by column and is
  cross-faded along azimuth. Use it for wide scenes in which the defocus changes with position.
- `'spatially_variant_pga'`: the same with a per-block PGA estimate reduced to the coefficients of the second
  to fourth powers of azimuth frequency.

With `guard=True` a result is kept only if the contrast did not drop; otherwise `info['applied']` is `False`.

**Device arrays.** `img` may be a NumPy, CuPy or JAX array; the result is a complex64 array of the same kind
on the same device. The correction of the full image and the map-drift estimate run on the device; PGA
estimates run on the host from a subset of columns.

```python
import fastsar
from fastsar import autofocus, planes

out = planes.deramp(fastsar.form_cphd('scene_CPHD.cphd', axes='los', window=False))
img, info = autofocus.image_autofocus(out['image'], 'quadratic')
print(info['applied'], info['edge_rad'], info['contrast_before'], info['contrast_after'])
img, info = autofocus.image_autofocus(out['image'], 'spatially_variant', block=512, order=2)
```

**Limitations.** Map drift needs point-like scatterers or texture; uniform clutter gives no estimate. The
global modes apply a circular correction, so a strip as long as the blur at each azimuth end of the image is
contaminated. The blur must be much shorter than the block of the spatially variant modes, and a response cut
by a zero region cannot be restored. Along azimuth the spatially variant correction blends neighbouring
strips: a coefficient that changes by more than about one radian per strip step leaves a residual, which a
smaller block or a larger `overlap` reduces. The block PGA estimate is single-threaded and slow.

`process_cphd(..., autofocus=)` runs these on the baseband ground image, before any resampling, and records the
mode in the SICD (`AzAutofocus` GLOBAL or SV). In `tests/test_autofocus_image.py` a quadratic error of 2 to 40
rad at the band edge is recovered within 0.5% on synthetic images, and a simulated collection with a quadratic
error injected into its phase history returns to the error-free azimuth resolution within 0.7% and peak sidelobe
within 0.3 dB.

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
