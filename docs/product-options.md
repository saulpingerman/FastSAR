# Product options: area, plane, autofocus, weighting, SICD and SIDD

`process_cphd` goes from a CPHD file to a product on disk in one call. `form_cphd` remains the formation step
underneath it and keeps its own defaults.

```python
import fastsar
fastsar.process_cphd('scene_CPHD.cphd', 'scene_SICD.nitf')                    # complex image, ground plane, unweighted
fastsar.process_cphd('scene_CPHD.cphd', 'scene_SICD.nitf', plane='slant')     # complex image, slant plane
fastsar.process_cphd('scene_CPHD.cphd', 'scene_SIDD.nitf', product='sidd')    # detected image, ground plane
out = fastsar.process_cphd('scene_CPHD.cphd', product='sidd', area='black', remap='density')   # nothing written
```

> **The default SICD is a ground-plane SICD.**
>
> FastSAR forms every spotlight image on the ground plane, and unless asked otherwise `process_cphd` writes that
> image. The file says so: `Grid.Type` is `PLANE` and `Grid.ImagePlane` is `GROUND`. Rows run along ground range
> away from the radar, and columns run so that the array is the view from above.
>
> This is less common than the alternative. Most producers deliver slant-plane SICDs (`ImagePlane` `SLANT`), and
> `plane='slant'` gives one.
>
> A reader that projects pixels through the metadata (the SICD projection model, as sarpy and sarkit implement it)
> handles both planes. Check any tool that assumes a slant plane without reading `Grid.ImagePlane`
> ([the list is below](#the-ground-plane-sicd-the-default)).
>
> Ground is the default because it is the plane of formation. Backprojection onto the ground plane is exact for any
> track, and the ground-plane product is the image as formed, where the slant-plane product is that image
> resampled once more ([the measured comparison](#forming-on-the-ground-or-on-the-slant-plane-and-projecting)).

The options and their defaults:

- `product`: `'sicd'` (complex image, the default) or `'sidd'` (detected, remapped 8-bit image).
- `area`: `'tasked'` (default), `'tasked_quality'`, an RNIIRS value, or `'black'`.
- `plane`: `'ground'` (the default for every product) or `'slant'` (a spotlight SICD only).
- `autofocus`: `'quadratic'` (default), `'pga'`, `'spatially_variant'`, `'spatially_variant_pga'` or `False`.
- `weighting`: `'uniform'`, `'taylor'` or `'sva'`. The default is no weighting (`'uniform'`) for a SICD, so that
  whoever uses the complex image can still choose an apodization, and spatially variant apodization (`'sva'`) for
  a SIDD.
- `remap`: for a SIDD, one of the remaps of the NGA MATLAB SAR Toolbox (`'nrl'` by default).
- `spacing`: the product's pixel spacing. The default follows the bandwidth of the data: 0.96 of the reciprocal
  of the band for a complex product, half of it with `'sva'`, and 0.7 of the finer ground resolution for a SIDD
  (one number: its pixels are square). A spacing coarser than the band allows is used as given, with a note in
  the product that the image is aliased.
- `extent`: the sides of the tasked area in meters about the scene center, for a collection that states no image
  area or in place of the one it states.

The defaults by product:

| `product` | `plane` | `weighting` | `autofocus` | `area` | `remap` |
|---|---|---|---|---|---|
| `'sicd'` | `'ground'` | `'uniform'` | `'quadratic'` | `'tasked'` | |
| `'sidd'` | `'ground'` | `'sva'` | `'quadratic'` | `'tasked'` | `'nrl'` |

FastSAR 0.2.0 wrote a slant-plane SICD with Taylor weighting by default;
`process_cphd(cphd, output, plane='slant', weighting='taylor')` is that product.

Some combinations are refused with a `ValueError` before the image is formed: a SIDD on the slant plane, a
ground-plane SICD with `'sva'` (apodization runs on the slant plane, so an apodized SICD needs `plane='slant'`),
and a `spacing` with `'sva'` for a SICD (apodization fixes the spacing at twice Nyquist). Without an `output`
nothing is written, and the image of a SIDD product is returned as the detected amplitude;
`sidd.to_display(out, out['remap'])` gives its display values.

A moving-beam collection (stripmap, sliding spotlight) is formed on the ground plane on axes along the track,
with Taylor weighting in range unless another is asked for, for a SICD as for a SIDD. The slant plane, autofocus,
apodization and the quality areas apply to spotlight collections: asking for one of them on a moving-beam
collection raises a `ValueError`.

## Speed

The stages after formation take the image as a NumPy, CuPy or JAX array and run where it lives, so a GPU or TPU
run keeps the image on the device from formation until the file is written. Seconds for two products of two
collections, the slant-plane SICD with Taylor weighting (`plane='slant', weighting='taylor'`) and the default SIDD,
from the CPHD file on disk (in the page cache) to the product file, second run:

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
apodization, detection and the projection to the ground; the SIDD's write includes the remap. The default SICD, on
the ground plane, has no resampling stage. The CPHD files are 8.7 GB (Umbra) and 9.2 GB (Capella).

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

The same two products of a small Umbra spotlight collection (a 0.64 GB CPHD file) on the same TPU v5e, second
run, in seconds:

| Product | Read | Form | After formation | Write | Total |
|---|---|---|---|---|---|
| Slant SICD, 23 Mpx | 13.0 | 1.49 | 0.46 | 0.22 | 15.2 |
| Ground SIDD, 50 Mpx | 13.0 | 1.42 | 1.32 | 0.22 | 16.0 |

The first run of each product, which compiles, takes 103 and 84 s.

## Planes

FastSAR forms every spotlight image on a ground plane, where backprojection is exact for any track. A planar
slant grid is exact only for a straight track: on an orbit, the range history of a slant-plane pixel departs from
that of the ground point it stands for, and a long aperture defocuses. The ground-plane product, the default, is
the image as formed. The slant-plane product is the ground image resampled.

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

### The ground-plane SICD (the default)

A ground-plane SICD is the image as formed, at baseband, on axes along the line of sight. Its metadata state
`Grid.Type` PLANE and `Grid.ImagePlane` GROUND, and the row and column unit vectors lie in the ground plane at the
scene center point. Its rows run along ground range away from the radar whatever the squint, and its columns run
so that row x column points up (the view from above, for either look side). It is sampled for the whole extent of
its spectrum along the line of sight, which under squint is wider than the range band alone, so that the product's
spectrum does not wrap.

Most producers deliver slant-plane SICDs, so software written against their products may assume one. A consumer
of the ground-plane product should check the following.

- Projection. A reader that uses the SICD projection model with the grid's own unit vectors (sarpy's
  `point_projection`, sarkit) needs nothing: in the tests below sarpy places each target within 0.05 pixel on
  both planes.
- Spacings and bandwidths. `Grid.Row.SS` is a ground range spacing and `Grid.Row.ImpRespBW` a ground-plane
  bandwidth. A tool that takes them for slant range (c / 2B for the range resolution, or a division by the cosine
  of the grazing angle to reach the ground) applies the slant-to-ground factor twice.
- Pixel area. A tool that derives one radiometric quantity from another through the slant-plane area of a pixel,
  1 / (`Row.ImpRespBW` x `Col.ImpRespBW`), has the ground-plane area here.
- Spectrum. Under squint the spectrum of the ground-plane image is a parallelogram: its extent across the line of
  sight is the azimuth band, and its position along the line of sight shifts with the azimuth frequency. A taper
  applied along rows and along columns separately assumes a rectangle. To apodize a squinted collection after
  the fact, take `plane='slant'`, whose spectrum is the usual one.
- Pairs. An interferogram or a change detection against a vendor's slant-plane image needs both on one grid:
  `plane='slant'`, or `sicd=` with the vendor's file.

`plane='slant'` gives the slant-plane SICD for any of these.

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
area is displaced with it.

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

`process_cphd(..., autofocus=)` runs these on the baseband ground image, before any resampling, and records the
mode in the SICD (`AzAutofocus` GLOBAL or SV). In `tests/test_autofocus_image.py` a quadratic error of 2 to 40
rad at the band edge is recovered within 3% on synthetic images (0.5% measured), and a simulated collection
with a quadratic error injected into its phase history returns to the error-free azimuth resolution within 2%
and peak sidelobe within 0.5 dB (0.7% and 0.3 dB measured).

## Weighting and spatially variant apodization

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
