# Product options: area, plane, autofocus, weighting, SICD and SIDD

`process_cphd` goes from a CPHD file to a product on disk in one call. `form_cphd` remains the formation step
underneath it and keeps its own defaults.

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
  apodization for a SIDD (Taylor for both on a moving beam).
- `remap`: for a SIDD, one of the remaps of the NGA MATLAB SAR Toolbox (`'nrl'` by default).
- `spacing`: the product's pixel spacing. The default follows the bandwidth of the data: 0.96 of the reciprocal
  of the band for a complex product, half of it with `'sva'`, and 0.7 of the finer ground resolution for a SIDD
  (one number: its pixels are square). A spacing coarser than the band allows is used as given, with a note in
  the product that the image is aliased.
- `extent`: the sides of the tasked area in meters about the scene center, for a collection that states no image
  area or in place of the one it states.

Some combinations are refused with a `ValueError` before the image is formed: a SIDD on the slant plane, a
ground-plane SICD with `'sva'` (apodization runs on the slant plane), and a `spacing` with `'sva'` for a SICD
(apodization fixes the spacing at twice Nyquist). Without an `output` nothing is written, and the image of a SIDD
product is returned as the detected amplitude; `sidd.to_display(out, out['remap'])` gives its display values.

A moving-beam collection (stripmap, sliding spotlight) is formed on the ground plane on axes along the track.
The weightings, autofocus and the quality areas apply to it in the forms that
[Moving-beam collections](#moving-beam-collections) describes; the slant plane is for spotlight collections, and
`plane='slant'` on a moving beam raises a `ValueError`.

## Speed

The stages after formation take the image as a NumPy, CuPy or JAX array and run where it lives, so a GPU or TPU
run keeps the image on the device from formation until the file is written. Seconds for the default products of
two collections, from the CPHD file on disk (in the page cache) to the product file, second run:

| Collection, product | Processor | Read | Form | After formation | Write | Total |
|---|---|---|---|---|---|---|
| Umbra spotlight, slant SICD, 51 Mpx | Nvidia RTX PRO 6000 | 0.73 | 1.75 | 0.07 | 0.26 | 2.8 |
| | 48 CPU cores | 0.75 | 17.5 | 1.1 | 0.21 | 19.6 |
| Umbra spotlight, ground SIDD, 91 Mpx | Nvidia RTX PRO 6000 | 0.75 | 1.57 | 0.21 | 0.15 | 2.7 |
| | 48 CPU cores | 0.74 | 17.3 | 3.6 | 0.69 | 22.2 |
| Capella spotlight, slant SICD, 220 Mpx | Nvidia RTX PRO 6000 | 1.14 | 3.75 | 0.33 | 1.09 | 6.3 |
| | 48 CPU cores | 1.14 | 37.6 | 6.3 | 0.94 | 46.0 |
| Capella spotlight, ground SIDD, 429 Mpx | Nvidia RTX PRO 6000 | 1.14 | 3.48 | 0.91 | 0.56 | 6.1 |
| | 48 CPU cores | 1.16 | 35.4 | 17.8 | 3.6 | 57.9 |

"After formation" is the baseband step, quadratic autofocus, the resampling to the slant plane and, for the SIDD,
apodization, detection and the projection to the ground; the SIDD's write includes the remap. The CPHD files are
8.7 GB (Umbra) and 9.2 GB (Capella).

The same two products on a cloud instance, for another Capella spotlight collection (2025-01-15), second run, in
seconds:

| Product | Processor | Read | Form | After formation | Write | Total |
|---|---|---|---|---|---|---|
| Slant SICD, 223 Mpx | Nvidia L4 (g2-standard-32) | 1.76 | 13.0 | 1.14 | 1.20 | 17.1 |
| | the 32 vCPUs of that instance | 1.76 | 77.4 | 11.5 | 0.56 | 91.2 |
| Ground SIDD, 428 Mpx | Nvidia L4 (g2-standard-32) | 1.77 | 13.1 | 3.57 | 1.23 | 19.7 |
| | the 32 vCPUs of that instance | 1.76 | 78.3 | 39.0 | 7.68 | 126.8 |

At the instance's on-demand price ($1.734 per hour) a thousand slant products cost $8.2 on the L4. Over six
spotlight collections (slant products of 9 to 341 Mpx) the median totals on the L4 were 13.5 s for the slant SICD
and 15.2 s for the SIDD. The SIDD of a collection whose slant image is about 1,000 Mpx took 50 s on the same
L4, and a product whose images do not fit the GPU's memory is streamed through it from the host
([performance.md](performance.md)).
On a TPU v6e the resampling to the slant plane takes about 0.01 s per megapixel, from 35 to 1,155 Mpx.

On a TPU the image likewise stays on the device. Seconds per stage on a Cloud TPU v5e (one chip, 16 GB), second
call, on synthetic speckle images with the maps of a spaceborne geometry: a 23 Mpx ground image to a slant grid of
the same sampling (the SICD path), and a 13 Mpx and a 53 Mpx ground image to slant grids of 49 and 196 Mpx at twice
Nyquist, apodized, detected and projected to 49 Mpx of ground pixels with the cubic kernel (the SIDD path). The
resampling to the slant plane includes a mask of valid ground pixels.

| Stage | 23 Mpx slant | 49 Mpx slant, 49 Mpx ground | 196 Mpx slant, 49 Mpx ground |
|---|---|---|---|
| Baseband | 0.01 | 0.004 | 0.02 |
| Quadratic autofocus: estimate, correction | 0.04, 0.10 | 0.03, 0.03 | 0.08, 0.28 |
| To the slant plane | 0.57 | 0.89 | 4.4 |
| Apodization | | 0.03 | 0.21 |
| To the ground plane (cubic) | | 0.24 | 0.63 |
| Remap (NRL) and layout | | 0.08 | 0.07 |
| Copy to the host | 0.04 (complex) | 0.01 (8-bit) | 0.01 (8-bit) |
| Device memory in use at the peak | 1.1 GB | 1.8 GB | 6.4 GB |

Each stage compiles once per image shape: 45, 28 and 110 s for the resampling to the slant plane of the three
sizes, 8 to 14 s for the projection to the ground, and 30 to 50 s for the autofocus estimate.

The default products of a small Umbra spotlight collection (a 0.64 GB CPHD file) on the same TPU v5e, second run,
in seconds:

| Product | Read | Form | After formation | Write | Total |
|---|---|---|---|---|---|
| Slant SICD, 23 Mpx | 13.0 | 1.49 | 0.46 | 0.22 | 15.2 |
| Ground SIDD, 50 Mpx | 13.0 | 1.42 | 1.32 | 0.22 | 16.0 |

The first run of each product, which compiles, takes 103 and 84 s.

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
line of sight is the azimuth band alone. Each line across the line of sight is then band limited, and a
one-dimensional interpolation along it is exact even though the spectrum wraps along the other axis. `planes.to_slant` interpolates along that axis first, which leaves a spectrum that is a
rectangle in azimuth and ground range, and then along the line of sight. Each pass oversamples by two with an FFT
and applies an 8-tap Kaiser sinc from a table. `planes.sampling` gives the pixel spacings the ground image needs
from the collection's own spectrum at points of the scene; `process_cphd` uses 0.96 of their reciprocal.

### Forming on the ground, or on the slant plane and projecting

A ground-plane product can be made two ways: formed on the ground directly, or formed on a planar slant grid and
projected to the ground. The two were compared on an Nvidia RTX PRO 6000 with the unweighted image of two
collections over their tasked areas, the slant grid being the rectangle that covers the same ground.

| Collection | Aperture | Ground, direct | Slant, then projection | Agreement of the two (coherence in 32 x 32 pixel blocks, median) |
|---|---|---|---|---|
| Umbra spotlight, 22 Mpx on the ground | 1.0 degree, 5,202 pulses | 2.7 s | 2.9 s + 1.0 s | 0.9995 |
| Capella spotlight, 152 Mpx on the ground | 2.3 degrees, 33,073 pulses | 5.8 s | 12.9 s + 4.4 s | 0.45 |

Direct formation is the cheaper one, for two reasons. The projection is a second pass over the image, and the
slant rectangle that covers a ground rectangle is larger than it wherever the collection is squinted (2.3 times the pixels
for the Umbra collection and 1.4 times for the Capella one). It is also the exact one. Backprojection onto a plane is exact for the points of that
plane, and a ground scatterer is not on the slant plane: its range history equals that of its slant-plane image
point only on a straight track. Over the Umbra aperture the difference is small. Over the longer Capella aperture
the slant-formed image no longer matches the ground-formed one (the correlation of the two amplitude images, which
ignores phase altogether, is 0.56). FastSAR therefore
forms on the ground plane always, and derives the slant-plane product from it by the range and range-rate map,
which is exact at the center of aperture for any track.

A ground-plane SICD is the image as formed, at baseband, on axes along the line of sight. Its rows run along
ground range away from the radar whatever the squint. It is sampled for the whole extent of its spectrum along
the line of sight, which under squint is wider than the range band alone, so that the product's spectrum does not
wrap.

In `tests/test_planes.py` and `tests/test_process.py`, on a simulated collection with a 20 to 25 degree squint
and a curved track, flown to either side, each target falls within 0.05 pixel of the position sarpy's projection
of the written SICD gives, on both planes, and on the slant plane its phase is within 0.01 rad of SICD's
convention.

### Against a vendor's SICD of the same collection

A FastSAR product and the vendor's SICD of the same collection can place a scatterer at slightly different
positions, for reasons that are not errors of either.

- The troposphere delay. `read_cphd` removes the delay the CPHD file states ([real-data.md](real-data.md)). Umbra's
  SICD images keep it, so the FastSAR image sits nearer by c times that delay over two: 1.2 m on a collection
  that states 8.1 ns, measured as a constant 4.6 rows against the vendor's image, and 0.07 pixel with
  `troposphere=False`.
- The image formation geometry. Against one Capella spotlight SICD whose scene center point lies 590 m from the
  CPHD's scene reference point, the FastSAR image differs by an azimuth scale of 4e-4 to 6e-4 (3.6 m across a 7 km
  scene), which matches the ratio of the polar angle rates about the two points (5.4e-4). Synthetic point targets
  on that collection's real track, spread over 7 km and processed by `process_cphd`, land where FastSAR's own
  SICD metadata puts them to an azimuth scale of 1e-8.
- Height. Scatterers above or below the projection surface map to slightly different ground points in two
  products whose centers of aperture differ, by about the height times the difference in look direction.

`process_cphd(..., sicd=)` takes the vendor's footprint; the product is the rectangle that bounds it on FastSAR's
own axes.

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
import numpy as np
from fastsar import io, rgiqe
data, meta = io.read_collection('collect.cphd', meta=True)
rad = rgiqe.Radiometry('collect.cphd', meta)                  # see rad.level_source and rad.notes
c = rad.ref_point                                             # the scene reference point, local frame of meta
e2 = rad.range_dir                                            # ground axes: along the line of sight, and across
e1 = np.cross(e2, [0.0, 0.0, 1.0])
bw_area = 4.0 * 3.5 * np.cos(rad.graze)                       # range band x azimuth band (cycles/m) x cos(slope)
samples = data['S'].size                                      # pulses x frequency samples
print(rad.nesz_db(c[None], bw_area, samples), rad.rniirs(c[None], bw_area, samples))
lo1, hi1, lo2, hi2 = rgiqe.area_bounds(rad, 'black', c, e1, e2, bw_area, samples)
```

### Limitations

- The image is assumed uniformly weighted and not corrected for the antenna pattern.
- Gain polynomials are trusted to 15 dB one-way below their peak; beyond that no area criterion is met.
- An `'uncalibrated'` level rests on the assumed scene sigma0; a range-Doppler noise level is an upper bound.
- Ambiguities assume a uniform scene and are placed with the mid-aperture geometry; range ambiguities are ignored.
- A low RNIIRS threshold at a large bandwidth area needs an SNR below 1, and its area can exceed `'black'`.

`process_cphd(..., area=)` takes the same four kinds. The image is formed over the bounding rectangle of the area
on the ground axes, with one impulse response everywhere and a noise-equivalent sigma zero that rises toward the
edges, and the pixels outside the area are set to 0. For a file that describes no antenna pattern the model cannot
predict where the signal fades, and an `area` other than `'tasked'` raises a `ValueError`. The model's beam is
the one the file's antenna block states; where a producer's stated boresight differs from the true beam, the
area is displaced with it. On a moving beam the areas extend in ground range only
([below](#areas-in-ground-range)).

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

**The guard.** With `guard=True` (the default) a correction is applied only if it passes two tests; otherwise the
image comes back as it was given, with `info['applied'] = False` and the cause in `info['reason']`.

- *The parts of the image agree.* For `'quadratic'` the columns of the estimate are cut into eight contiguous
  sets, each set gives its own coefficient, and the sets whose correlation peak is significant are compared. With
  `sigma` the uncertainty of the estimate from their scatter (`info['edge_rad_std']`), the estimate must lie
  within `3 sigma` of their median, be at least `3 sigma` itself, and at most a quarter of the sets may be
  outliers. For `'pga'` the two halves of the columns are estimated separately and the rms of the estimate must be
  at least three times `info['rms_rad_std']`, half the rms of their difference. A scene whose parts ask for
  different corrections gets none from a global mode: relief seen from a curved track (a scatterer above or below
  the focus plane is defocused in proportion to its height), moving water, or one reflector that dominates the
  estimate.
- *The image gains.* The contrast (std / mean of the intensity over the nonzero pixels) is measured on every
  column of the image before and after the correction, on the device that holds it, for the whole image and for
  16 contiguous sets of columns. The contrast of the whole image must rise. The median over the sets of the log
  of their contrast ratio (`info['part_gain']`) must not be below minus twice its standard error
  (`info['part_gain_std']`, from the scatter of the sets), so that a gain of one bright part of the scene at the
  cost of most of it does not count. And one of the two gains, that median or the log ratio of the whole image
  (`info['gain']`, with `info['gain_std']` from a jackknife over the sets), must be at least twice its standard
  error. All four modes take this test.

`info` reports the estimate whether it was applied or not (`edge_rad` and `edge_rad_std`, the sets' own estimates
in `subsets`; `rms_rad` and `rms_rad_std` for `'pga'`), `contrast_before` and `contrast_after` of the whole image
(equal when no correction was evaluated), the gains and their standard errors, `applied` and `reason`. With
`guard=False` the estimate is applied as it is and the same measurements are reported. `process_cphd` records in
the product the mode that was applied, or none.

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

**Limitations.** Map drift needs point-like scatterers or texture; uniform clutter gives no estimate. A global
mode corrects one error for the whole image: where the error changes over the scene by more than its mean, the
guard declines and a spatially variant mode is the one to ask for. The
global modes apply a circular correction, so a strip as long as the blur at each azimuth end of the image is
contaminated. The blur must be much shorter than the block of the spatially variant modes, and a response cut
by a zero region cannot be restored. Along azimuth the spatially variant correction blends neighbouring
strips: a coefficient that changes by more than about one radian per strip step leaves a residual, which a
smaller block or a larger `overlap` reduces. The block PGA estimate is single-threaded and slow.

`process_cphd(..., autofocus=)` runs these on the baseband ground image of a spotlight, before any resampling,
and records the mode in the SICD (`AzAutofocus` GLOBAL or SV). A moving beam has its own estimator
([below](#autofocus-along-the-track)). In `tests/test_autofocus_image.py` a quadratic error of 2 to 40
rad at the band edge is recovered within 3% on synthetic images (0.5% measured), and a simulated collection
with a quadratic error injected into its phase history returns to the error-free azimuth resolution within 2%
and peak sidelobe within 0.5 dB (0.7% and 0.3 dB measured).

## Weighting and spatially variant apodization

This section is the spotlight's; [Moving-beam collections](#moving-beam-collections) has the same three for a
stripmap or a sliding spotlight.

`'taylor'` applies a Taylor window (nbar 4, 35 dB) to the phase history along frequency and along pulses.
`'uniform'` applies none. `'sva'` forms the image unweighted and applies spatially variant apodization
(`weighting.sva`) on the slant plane: at every pixel it picks, between the unweighted aperture and a Hann window,
the cosine-on-pedestal weighting that gives the least output. Main lobes keep the unweighted width and sidelobes
are removed where the spectrum is a rectangle sampled at an integer multiple of Nyquist. Two steps make it so.
`weighting.keystone` drops, at each frequency, the pulses outside the azimuth band the lowest frequency reaches
(the azimuth band of a pulse fan grows with frequency; 2.8% and 2.6% of the samples on the Umbra and Capella
collections of the table above). The slant image is then resampled at exactly twice Nyquist of the rectangle that is left.
The result is not band limited and not linear in the scene, so it suits detection and display. A SICD with
`weighting='sva'` is allowed on the slant plane and records the weighting as SVA.

## Moving-beam collections

A stripmap or sliding-spotlight collection gives every pulse its own scene reference point (SRP), and a ground
point is seen only while the beam passes over it. `form_cphd` forms such a collection as a mosaic on the ground
plane, on axes along and across the track ([stripmap-burst.md](stripmap-burst.md)), and `process_cphd` applies
its options to that image with `fastsar.moving`. What applies to which mode:

| Option | Spotlight | Stripmap, sliding spotlight |
|---|---|---|
| `plane` | slant or ground | ground (`'slant'` is refused) |
| `weighting='taylor'`, `'uniform'` | on the phase history | frequency window on the phase history, azimuth window per pixel over its processed band |
| `weighting='sva'` | on the slant plane | on the ground grid, where the spectrum is a rectangle on its axes (see the limits below) |
| `autofocus='quadratic'` | one error for the image | blocks along the track, tied together by a smooth fit |
| `autofocus='spatially_variant'` | blocks and a polynomial over the scene | the same blocks along the track, with a parabola across the swath |
| `autofocus='pga'`, `'spatially_variant_pga'` | yes | refused: one phase function per aperture, and every block has its own |
| `area` other than `'tasked'` | in both directions | in ground range; the tasked extent along the track |

### The spectrum of the image and baseband

Pulse p at frequency f adds to the image around a ground point x the spatial frequency (2 f / c) u_p(x), u being
the horizontal part of the unit vector from the antenna to x. A pixel takes the pulses whose look angle lies
within the processed spread of the beam center's: |sin_p(x) - s_p| <= dsin / 2, where sin_p(x) is the cosine of the
angle between the direction of flight and the line to x, s_p the same for the pulse's SRP, and dsin the spread
that `azimuth_fraction` of the PRF (or the vendor SICD's processed bandwidth) corresponds to.

- Along the track the band at a pixel is (2 f / c) [s - dsin / 2, s + dsin / 2] around the beam center's s at the
  time the beam crosses the pixel. Its center is the Doppler centroid: zero for a broadside stripmap, constant
  for a squinted one, and changing along the track for a sliding spotlight or a beam whose squint drifts. Its
  width at the center frequency is 2 dsin / lambda times the ratio of the platform's speed to the footprint's: 1
  for a stripmap flown along a straight line, about 1.1 from orbit, several for a sliding spotlight. The default
  pixel spacing along the track follows it.
- Across the track it is the range band times the horizontal part of the line of sight, (2 B / c) cos(graze), on
  the carrier (2 fc / c) cos(graze); both change across the swath.

`moving.baseband` removes both carriers with one phase, exact at every pixel as `planes.deramp` is for a
spotlight. With F(x, t) = |x - a(t)| + G(t), G the integral of s |da/dt| over time, F is stationary in t at the
time t_c(x) at which the beam center crosses x, and psi(x) = (4 pi fc / c) F(x, t_c(x)) has the gradient
(4 pi fc / c) u(x) at that time: the center of the pixel's spectrum on both axes. For a broadside stripmap psi is
4 pi fc / c times the range of closest approach. The track and s |da/dt| are polynomials fitted to the pulses,
t_c is solved at knots every 32 pixels, and F is evaluated per pixel in double: on the host's cores, in a CUDA
kernel, or for a JAX array on the host in blocks of rows whose phasor the device multiplies. Every
`process_cphd` product of a moving beam is at baseband, and its SICD states a band centered at zero.

The support is a rectangle on the grid's axes only approximately. The band along the track grows in proportion to
the frequency (a trapezoid, 3% wider at the top of a band of 3% of the carrier), a squint moves its center with
frequency and shears the range band by tan(squint) of the azimuth band, and cos(graze) changes the ground range
band across the swath.

### Weighting

`'taylor'` is the Taylor window (nbar 4, 35 dB) along frequency and, at each pixel, over its processed azimuth
band, zero outside it: what Taylor weighting means for a spotlight, applied to the pulses a pixel takes.
`form_cphd` itself keeps the Hann window it has always applied there (`azimuth_window='hann'`), whose main lobe
is 22% wider and whose first sidelobe is 3.5 dB higher; `azimuth_window='taylor'` or `'uniform'` selects the
others. `'uniform'` is a rectangle over the processed azimuth band and no frequency window. The weights enter the
final stage of the factorized backprojection of each patch, per subaperture and tile with their slope across the
tile. Measured on a simulated broadside stripmap, a 3 degree squinted one and a sliding spotlight
(`tests/test_moving.py`; widths in units of the reciprocal band, azimuth then range):

| Azimuth window | 3 dB width | Peak sidelobe | Theory |
|---|---|---|---|
| Hann (`form_cphd`'s default) | 1.440 to 1.441 | -31.4 to -31.5 dB | 1.441, -31.5 dB |
| Taylor (`weighting='taylor'`) | 1.181 to 1.188 | -34.5 to -35.7 dB | 1.184, -35 dB |
| Rectangle (`weighting='uniform'`) | 0.882 to 0.891 | -13.2 to -13.5 dB | 0.886, -13.26 dB |

The antenna's own pattern across the processed band multiplies these windows in real data: the image is not
corrected for it, so an unweighted real image has lower sidelobes and a wider main lobe than the table.

`'sva'` needs a rectangle sampled at exactly twice Nyquist. The image is formed with the rectangular azimuth
weighting made wider than the processed band (by the trapezoid, the squint's shift over the band, and 6% for the
soft edge of the per-pixel weights; 7.7% in all on the simulated stripmap), on a grid whose spacings are half the
reciprocal of the processed azimuth band and of the ground range band at the swath center and whose rows number a
multiple of four. At baseband, `moving.band_limit` then keeps the half of the azimuth spectrum about zero: every
pixel is left with the same band, the processed band at the center frequency, with sharp edges, whatever its
Doppler centroid. `weighting.sva` runs on that image, and 24 rows formed beyond each end are dropped. The product
is on the ground grid at those spacings.

What does not fit the rectangle is refused with a `ValueError` that names the reason:

- a squint that shears the range band by more than 10% of its width across the azimuth band (`moving.SHEAR`);
- a ground range band that changes by more than 5% across the swath (`moving.RANGE`);
- a processed azimuth band that differs by more than 10% over the scene (`moving.VARY`), as under a beam whose
  speed over the ground changes;
- a beam center that does not move smoothly (an SRP that steps).

Inside these limits an isolated point keeps its apodized sidelobes: on an ideal rectangle, a shear of 10% and a
range sampling 5% off twice Nyquist both leave the highest pixel outside the main lobe at -48 dB, where the exact
rectangle has it at -48 dB and the unweighted image at -17 dB. On the simulated
collections of `tests/test_moving.py` the highest pixel within 24 pixels of a target, outside the 7 by 7 pixels of
its main lobe, is at -42 to -53 dB of the peak on the broadside stripmap, -44 to -46 dB on the sliding spotlight
and -42 to -51 dB under a squint of 0.5 degrees, against -17 to -18 dB unweighted; the 3 dB widths are 0.83 to
0.90 of the reciprocal bands (0.886 unweighted) and the peaks keep their unweighted amplitude within 5%. A squint
of 3 degrees shears the range band of that collection by 11% and is refused.

### Autofocus along the track

Each along-track position is formed from its own stretch of pulses, so a phase error over slow time defocuses
each part of the strip by its own amount, and a scatterer above or below the focus plane adds a quadratic term in
proportion to its height wherever the track curves. `moving.autofocus` estimates the error on blocks and removes
it with a correction that follows the strip.

- **Blocks.** The pulses of a pixel change completely over the footprint of the synthetic aperture (its length
  on the ground). A block is half that footprint long, held between 256 and 4096 rows and no longer than the
  image, and blocks overlap by half: the pixels of a block share at least half of their pulses.
- **Estimate.** `autofocus.quadratic`'s: map drift on the brightest 30% of up to 1024 columns, for all of them
  together and for eight contiguous sets of them across the swath. The band is the processed band, so an
  estimate is in radians at the edge of the synthetic aperture.
- **Guard, first test.** `quadratic`'s, on every block: the correlation peaks of all columns and of at least half
  of the sets must be significant, the estimate of all columns must lie within three standard errors of the sets'
  median, and no more than a quarter of the sets may lie far from it. A block that passes counts, also when its
  estimate is small: it says that its part of the strip is focused.
- **Fit.** Along the track the estimates of the blocks that count are tied by a robust local line (tricube
  weights over four block steps, Tukey's biweight); a block that does not count takes the fit of its neighbors.
  Across the swath the departures of a block's agreeing sets from their median give a linear trend where it is
  three times its standard error (`'spatially_variant'`: a parabola, always), held at its end values beyond the
  columns the sets cover; where there is a trend, the block's estimate of all columns is made again with the
  trend removed column by column, because the columns in best focus would otherwise carry it.
- **Guard, second test.** After the correction the contrast of the corrected rows must rise by more than its
  scatter over 16 sets of columns, or the image is returned as it came; a single block whose own contrast does
  not rise is given back as it came. A fit below 0.02 rad everywhere changes nothing.
- **Correction.** `autofocus.spatially_variant`'s: strips of one block, extended by the length of the blur,
  corrected column by column and cross-faded along the track with weights that sum to one. Exact zeros stay
  exact zeros.

`info` holds the blocks' estimates and their standard errors, which blocks counted, the fit, the slope across the
swath, the blocks corrected and their contrast gains. The default for a moving beam is `'quadratic'`, as for a
spotlight.

In `tests/test_moving.py`, on a simulated strip of three blocks with 36 point targets, a quadratic error of 2, 5
and 15 rad at the edge of the synthetic aperture, injected into the phase history, is estimated as 2.00, 5.03 and
15.35 rad. The azimuth width goes from 1.358, 2.902 and 7.764 reciprocal bands back to 1.190, 1.189 and 1.192
(1.187 focused) and the peak sidelobe from -32.8, -25.2 and -0.6 dB to -34.1, -34.1 and -33.8 dB (-34.4 focused).
An error that changes from 3.1 to 4.9 rad along the strip is followed by the fit (3.67, 4.05 and 4.38 rad at the
three blocks), and the widest response comes out at 1.199. Targets 40 m above the focus plane, seen from a track
that curves with a radius of 2.9 km, are 2.4 times as wide as in focus (2.838) and return to 1.188; a height that
grows from -6 to 46 m across the swath is fitted with 6.0 rad per half swath, and the widest response goes from
3.12 to 1.22. A focused collection comes back bit for bit. A slow-time error that grows along the strip also
moves the scatterers along the track (its linear part over each aperture); autofocus does not undo that.

### Areas in ground range

`area='tasked_quality'`, an RNIIRS value and `'black'` keep the tasked extent along the track and extend the
product across it, in ground range, as far as the criterion holds; the criteria and the order of the noise
sources are those of [Processed area by image quality](#processed-area-by-image-quality). The model is
evaluated for the moving beam: at seven positions along the track, on the line across the track through each,
with the pulses that the pixels there are formed from. Each pulse's reference is its own SRP, so the gain of a
ground point is the antenna's two-way pattern in elevation and in azimuth, averaged in amplitude over those
pulses, with the spreading loss; the delay windows are those pulses' TOA1..TOA2 (TOAE1..TOAE2 for `'black'`).
The noise level at the beam center comes once, from the pulses of the middle position. The mask is read between
the seven lines, and pixels outside the area are 0.

An area that reaches less than one step of the model's lattice beyond the tasked swath is the tasked area
itself, on the tasked grid, with a note that says whether the receive window or the criterion ends it there. A
file without an antenna pattern is refused as for a spotlight. In
`tests/test_moving.py` a simulated stripmap carries an antenna block whose one-way gain falls by 10^5 dB per squared
direction cosine across the track, and the areas end where the two-way gain computed from that formula meets the
criterion, to one step of the model's lattice across the track (3 m there, 50 m by default): `'black'` at +-56 m
of ground range (gain -9.6 and -9.7 dB for a noise-equivalent sigma zero of -10 dB), an RNIIRS of 4.8 at -44 and
+43 m (-7.2 dB for the -7.5 dB it needs), `'tasked_quality'` at the tasked +-25 m. With an extended window of
+-45 m `'black'` ends there, and with a window that ends at the tasked swath all three give the tasked area.

### Real collections

Two public Capella collections were processed on an Nvidia RTX PRO 6000 and 48 CPU cores that other jobs were
using at the time, so the times below are upper bounds.

**Stripmap** (2024-11-27, VV, 200 MHz, 135,477 pulses in 13.4 s, grazing angle 53 degrees, tasked area 82.5 by
5.0 km; a 10 by 5.1 km cut at the center of the strip, formed from the 36,000 pulses that illuminate it). A pixel
takes 15,944 pulses. The footprint moves at 6.6 km/s under a platform at 7.3 km/s, so the azimuth band at a pixel
is 1.228 cycles/m, 1.10 times 2 dsin / lambda, and the default product is sampled at 0.771 m along the track
(0.2.0 sampled it at 0.859 m, coarser than that band). The beam center's squint drifts from -0.18 to +0.02
degrees over the collection (a Doppler centroid of -0.11 cycles/m at the cut), which the baseband phase follows;
the shear of the range band is 0.4% and the cosine of the grazing angle changes the ground range band by about
0.5% across the swath, so apodization is accepted.

The three weightings on one grid (the apodized product's, 0.407 by 0.621 m, 61 Mpx for a 3 km cut), measured at
the 40 brightest compact points of the scene (medians; widths times the product's bandwidths):

| Weighting | Image contrast | Azimuth width | Range width | Peak sidelobe, azimuth and range |
|---|---|---|---|---|
| `'taylor'` | 34.3 | 1.41 | 1.17 | -28.9, -23.8 dB |
| `'uniform'` | 34.7 | 1.08 | 0.88 | -21.5, -12.2 dB |
| `'sva'` | 44.7 | 1.10 | 0.87 | -26.4, -23.0 dB |

In range the widths are the theoretical ones (1.184 and 0.886). In azimuth the antenna's two-way pattern, 12 dB
down at the edge of the processed band by the file's gain polynomial, weights the aperture before any window
does: the unweighted response is 1.08 wide with a -21.5 dB sidelobe, and the Taylor window on top of it gives
1.41. Apodization keeps the unweighted widths, removes the range sidelobes and raises the contrast (the standard
deviation of the intensity over its mean) from 34.7 to 44.7. The sidelobes that remain around these points are
neighboring scatterers: the highest pixel within 32 pixels outside the main lobe is at -15 to -16 dB for all
three.

Autofocus on the collection as it came: six blocks of 4,096 rows (3.2 km; the footprint is 10.4 km) give -0.27,
-0.25, -0.25, -0.14, -0.67 and -0.72 rad (standard errors 0.04 to 0.37), the fit runs from -0.24 to -0.59 rad,
three of the six blocks are corrected, the contrast goes from 21.203 to 21.212 and the median azimuth width of 30
points from 1.146 to 1.142 m: the collection is in focus to half a radian, and the default does no harm. With a
quadratic error injected into the phase history (radians at the edge of a pixel's aperture):

| Injected | Contrast, blurred | Azimuth width, blurred | Fit over the six blocks | Contrast after | Azimuth width after |
|---|---|---|---|---|---|
| none | 21.20 | 1.146 m | -0.59 to -0.24 rad | 21.21 | 1.142 m |
| 2 rad | 20.48 | 1.238 m | 1.31 to 1.75 rad | 21.39 | 1.144 m |
| 5 rad | 17.40 | 1.945 m | 4.31 to 4.74 rad | 21.36 | 1.145 m |
| 15 rad | 11.88 | 5.067 m | 14.39 to 14.73 rad | 21.27 | 1.146 m |

The fits are the injected error less what the collection has of its own, block by block, and every block is
corrected.

Areas. The file has an antenna block (axis and gain polynomials) and no PTRef, so the noise level is the fifth
source of the list above: the clutter-to-noise ratio measured from 512 pulses (the least cell of a range-Doppler
map) with the assumed mean scene sigma zero of -10 dB. The full-bandwidth window TOA1..TOA2 of the pulses that
form a pixel is narrower than the tasked swath, so `'tasked_quality'` (a model minimum of RNIIRS 4.20 in the
tasked swath) and an RNIIRS of 4.5 give the tasked area, with the note that the receive window holds nothing
beyond it. `'black'` opens the swath to the extended window TOAE1..TOAE2 and finds a signal-to-noise ratio of at
least 1 from -3,550 to +3,300 m about the swath center, 6.9 km of ground range for the 5.1 km tasked (75.6 Mpx
for 55.5).

**Sliding spotlight** (2024-08-29, HH, 300 MHz, 111,132 pulses in 10.7 s, grazing angle 61 degrees, tasked area 5
by 5 km; a 1 by 3 km cut at the center, formed from the central 150 MHz of the band, which is what the shared
host's memory allowed). The footprint moves at 1.3 km/s under a platform at 7.2 km/s: a pixel takes 86,878 pulses
(8.4 s), its azimuth band is 6.50 cycles/m, 5.8 times 2 dsin / lambda, and the default product is sampled at
0.146 m along the track (0.2.0 sampled it at 0.84 m). The beam center's squint runs from -2.7 to +2.8 degrees
over the collection, and the baseband phase follows it.

- Weighting. `'taylor'`: 6,873 by 1,507 pixels, contrast 10.12, median azimuth width of 30 points 0.255 m (1.66
  reciprocal bands) and range width 2.50 m (1.20). `'uniform'`: contrast 8.85, azimuth width 0.201 m (1.31).
  `'sva'` is refused on this cut: the squint at its ends, 0.6 degrees on the ground, shears the range band by 13%
  of its width (half of that with the whole 300 MHz). On a 0.7 by 3 km cut the shear is within the limit: the
  product is 9,052 by 2,893 pixels of 0.077 by 1.04 m, and over 40 points the contrast is 7.3, 7.1 and 8.6 for
  Taylor, uniform and apodized, the range widths 1.21, 0.90 and 0.87 reciprocal bands with peak sidelobes of
  -17.1, -12.5 and -16.7 dB, and the azimuth widths 1.62, 1.32 and 1.27.
- Autofocus on the collection as it came: three blocks give -1.31, -1.78 and -1.64 rad (standard errors 1.0, 0.5
  and 0.35), the sets of one block agree, and the fit is -1.64 rad; the contrast gain of the correction, 0.5%, is
  within its scatter over the parts of the image (0.8%), and the guard declines. The error is real: with 2 rad
  injected the image is sharper than as it came (contrast 10.15, azimuth width 0.243 m), the fit is 0.36 rad and
  is applied (contrast 10.17). With 15 rad injected (contrast 9.21, azimuth width 0.591 m) the fit is 13.32 rad,
  every block is corrected, and the image returns to contrast 10.17 and 0.249 m. With 5 rad (contrast 9.83,
  0.306 m) the fit is 3.20 to 3.35 rad, the right value, but the contrast gain it brings, 3.3%, is within twice
  its scatter over the 16 sets of columns of this 1,507 column image (2.5%), and the guard returns the image as it
  came; with `guard=False` the same fit takes the contrast to 10.17. The guard's second test is the limit here,
  not the estimate.
- Areas, with the same noise source as the stripmap. `'tasked_quality'` (a model minimum of RNIIRS 5.15): -1,650
  to +1,500 m of ground range about the swath center for the tasked +-1,500 m. An RNIIRS of 4.5: -1,650 to
  +2,150 m. `'black'`: -3,150 to +4,100 m, 7.3 km of ground range, in the extended window.

**Speed.** Seconds per stage for the stripmap cut, second call, with megapixels per second in parentheses. The
GPU and the host were shared with other jobs throughout (the GPU at full utilization, a load average of 250 to
400 on the host's 48 cores), so these are lower bounds on the rates.

| Product | Arrays | Form | Baseband | Autofocus | Apodization | Mask |
|---|---|---|---|---|---|---|
| Taylor SICD, 55.5 Mpx | CuPy, Nvidia RTX PRO 6000 | 21 to 26 | 0.065 (855) | 1.62 (34) | | |
| | JAX on the same GPU | | 4.8 (12) | 1.40 (40) | | |
| | NumPy, host | | 3.9 (14) | 6.1 (9) | | |
| Apodized SICD, 201.8 Mpx | CuPy | 116 | 1.47 (137) | 4.8 (42) | 0.044 (4,500) | |
| | NumPy, host | | 21.2 (10) | 35.6 (6) | 10.0 (20) | |
| Taylor SICD of the `'black'` area, 75.6 Mpx | CuPy | 56 | 0.14 (550) | 2.5 (30) | | 0.38 (197) |
| | NumPy, host | | 6.5 (12) | 18.2 (4) | | 1.5 (50) |

Formation is the mosaic's on the GPU (the 36,000 pulses of the cut, 8,194 samples each); on the CPU backend,
under that load, the mosaic of a 2 km cut (11.1 Mpx from 24,000 pulses) took 619 s. The stages after
formation are the new ones. The baseband step on the GPU includes the upload of the mosaic, which the former
assembles on the host; for a JAX array its phase is computed on the host's cores, so it runs at the host's rate.
Autofocus is the estimate on the blocks, the cross-faded correction and the guard's contrast sums before and
after; it was applied in every row of the table (5 rad injected). The three kinds of array give the same product
to 3e-7 of its peak, zeros included. The image goes to the device when its memory plan (`memory.cuda_plan`) holds
it with a second image of its size; otherwise the stages run on the host and a note says so.

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
product_name=None, classification='UNCLASSIFIED', sicd=None)` writes those pixels as a MONO8I SIDD
and returns the SIDD metadata (a light object over the XML, `fastsar._siddxml.SIDD`). It accepts ground-plane products only. The metadata are:

- Measurement: a plane projection (reference point at the center pixel, sample spacings, row and column unit
  vectors in ECF), the aperture position polynomial and one center-of-aperture time, so the SIDD's own projection
  is the range-Doppler model of `geolocate`; the pixel footprint; and ValidData, the convex hull of the valid
  pixels.
- GeoData: the image corners and the valid-data polygon in latitude and longitude.
- Display: the remap's name and the shadow direction (DOWN when shadows follow the rows to within one degree,
  otherwise UP, LEFT, RIGHT or ARBITRARY).
- ExploitationFeatures: azimuth, graze, slope, tilt, squint and Doppler cone angles, the shadow, layover,
  multipath, ground-track and north directions in the pixel frame, and the resolution, all computed
  by the standard's definitions from the SICD metadata of the same product (`sicd_meta(out)`, or the `sicd` argument, which is then stored in
  the file as well).
- ProductCreation: FastSAR and its version as the processor.

SIDD 3.0 is written by default and 2.0 on request, by FastSAR's own NITF writer. Two NITF header fields differ
from what sarpy 2.1.1 writes for the same product: the specification version and date in the XML segment's
subheader, and the image segment's corner coordinates, which sarpy truncates to the arcsecond where the standard
rounds. `classification` is `'UNCLASSIFIED'` or a dict of the ISM markings (`classification`, `ownerProducer`, ...).

In `tests/test_sidd.py`, on a simulated collection flown both ways, the round trip keeps every pixel, the SIDD's
projection of the measured peaks falls within 0.1 pixel of the true target positions (0.01 pixel measured), an asymmetric
target pattern appears with range down the rows and unmirrored for either look direction, and sarkit's
consistency check (Python 3.11 or later) reports no errors for SIDD 3.0, SIDD 2.0, a masked product and an
embedded SICD.
