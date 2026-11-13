# Product options: area, plane, autofocus, weighting, SICD and SIDD

`process_cphd` goes from a CPHD file to a product on disk in one call, for a spotlight and for a moving-beam
collection (stripmap, sliding spotlight). `form_cphd` remains the formation step underneath it and keeps its own
defaults. A spotlight SICD is radiometrically calibrated where the file supports it
([Radiometric calibration](#radiometric-calibration)).

```python
import fastsar
fastsar.process_cphd('scene_CPHD.cphd', 'scene_SICD.nitf')                    # complex image, ground plane, unweighted
fastsar.process_cphd('scene_CPHD.cphd', 'scene_SICD.nitf', plane='slant')     # complex image, slant plane
fastsar.process_cphd('scene_CPHD.cphd', 'scene_SIDD.nitf', product='sidd')    # detected image, ground plane
out = fastsar.process_cphd('scene_CPHD.cphd', product='sidd', area='black', remap='density')   # nothing written
```

> **The default SICD is a ground-plane SICD.**
>
> FastSAR forms every image on the ground plane, and unless asked otherwise `process_cphd` writes that image.
> The file says so: `Grid.Type` is `PLANE` and `Grid.ImagePlane` is `GROUND`. Rows run along ground range away
> from the radar, and columns run so that the array is the view from above.
>
> This is less common than the alternative. Most producers deliver slant-plane SICDs (`ImagePlane` `SLANT`), and
> `plane='slant'` gives one for a spotlight collection.
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
- `area`: `'tasked'` (default), `'tasked_quality'`, an RNIIRS value, or `'black'` (out to where terrain of
  `black_sigma0_db`, -10 dB by default, sinks into the noise and the ambiguities).
- `plane`: `'ground'` (the default) or `'slant'` (a spotlight SICD only). One SICD is on the slant plane without
  being asked: the apodized one of a spotlight (`weighting='sva'`), because its apodization runs there.
- `autofocus`: `'quadratic'` (default), `'pga'`, `'spatially_variant'`, `'spatially_variant_pga'` or `False`. A
  moving beam takes `'quadratic'` and `'spatially_variant'`.
- `weighting`: `'uniform'`, `'taylor'` or `'sva'`. The default is no weighting (`'uniform'`) for a SICD, so that
  whoever uses the complex image can still choose an apodization, and spatially variant apodization (`'sva'`) for
  a SIDD, for a spotlight and for a moving beam alike.
- `remap`: for a SIDD, one of the remaps of the NGA MATLAB SAR Toolbox (`'nrl'` by default).
- `spacing`: the product's pixel spacing. The default follows the bandwidth of the data: 0.96 of the reciprocal
  of the band for a complex product, half of it with `'sva'`, and 0.7 of the finer ground resolution for a SIDD
  (one number: its pixels are square). A spacing coarser than the band allows is used as given, with a note in
  the product that the image is aliased.
- `extent`: the sides of the tasked area in meters about the scene center, for a collection that states no image
  area or in place of the one it states.
- `dem`, `geoid`: a DEM, with which the defocus of terrain off the image plane is removed before the autofocus
  ([terrain-informed focusing](#terrain-informed-focusing)), for a spotlight on either plane. The default is none.
- `grid`: an Earth-fixed ground grid in place of the collection's own area, axes and spacing, so that several
  collections of one scene share their pixels ([stack.md](stack.md#the-common-grid)). The product is then the
  image on that grid: the quality areas, the slant plane, apodization and a `dem` do not apply (a SIDD is Taylor
  weighted in place of its default, with a note), and the SICD carries no `Radiometric` block. `process_stack`
  takes several collections to one product ([stack.md](stack.md)).

The defaults by product:

| `product` | `plane` | `weighting` | `autofocus` | `area` | `remap` |
|---|---|---|---|---|---|
| `'sicd'` | `'ground'` | `'uniform'` | `'quadratic'` | `'tasked'` | |
| `'sidd'` | `'ground'` | `'sva'` | `'quadratic'` | `'tasked'` | `'nrl'` |

These hold for a spotlight and for a moving-beam collection. FastSAR 0.2.0 wrote a slant-plane SICD with Taylor
weighting by default; `process_cphd(cphd, output, plane='slant', weighting='taylor')` is that product.

A spotlight SICD with `weighting='sva'` and no `plane` comes out on the slant plane, with a note in the product
that says so. Some combinations are refused with a `ValueError` before the image is formed: a SIDD on the slant
plane, a spotlight SICD with `'sva'` and an explicit `plane='ground'` (apodization runs on the slant plane), and a
`spacing` with `'sva'` for a SICD (apodization fixes the spacing at twice Nyquist). Without an `output`
nothing is written, and the image of a spotlight SIDD product is returned as the detected amplitude;
`sidd.to_display(out, out['remap'])` gives its display values.

A moving-beam collection (stripmap, sliding spotlight) is formed on the ground plane on axes along the track.
The weightings, autofocus and the quality areas apply to it in the forms that
[Moving-beam collections](#moving-beam-collections) describes, with the defaults of the table. Three things
differ from a spotlight:

- The slant plane is for spotlight collections: `plane='slant'` on a moving beam raises a `ValueError`. Its
  apodized products are on the ground grid, the SICD included.
- Apodization needs a spectrum that is a rectangle on the grid's axes, which a squint or a wide swath can rule
  out. Where it is ruled out and `weighting` was not given, the SIDD is Taylor weighted and a note in the product
  says why; `weighting='sva'` given by name raises a `ValueError` with the same reason.
- Terrain-informed focusing (`dem`) and the `Radiometric` block are a spotlight's: a `dem` raises a `ValueError`,
  and the SICD carries no block, with a note.

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
  1 / (`Row.ImpRespBW` x `Col.ImpRespBW`), has the ground-plane area here. FastSAR writes all four scale factors
  of the `Radiometric` block itself, so none needs deriving ([Radiometric calibration](#radiometric-calibration)).
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
channel and -6.02 dB for a cross-polarized one, as in sarpy. Because the ambiguities are counted as noise, the
ratio is one of signal to noise and clutter: it is at least 1 exactly where the noise-equivalent sigma zero plus
the ambiguity level, NESZ(x) + sigma0 sum_k G(x_k) / G(x), is at most sigma0.

### Noise level

The level at the scene reference point comes from the first source available (`level_source`; `notes` lists what
was assumed). A measurement takes up to 512 pulses and fits the noise floor in the extended delay swath beside the
scene when the file saves one; otherwise it takes the lowest cell of a range-Doppler power map. Its cost is
bounded: about 8 million samples are transformed at most (fewer blocks of pulses when a pulse is long, each pulse
with its whole band; 64 pulses of an Umbra collection with 120,736 samples per pulse, which give the level of 512
to 0.03 dB), on all cores, and `process_cphd` hands it the phase history it has already read, so the file is not
read a second time (`Radiometry(..., history=)`). On that Umbra collection, in a fresh process held to 16 cores
on a 48-core machine that other jobs kept at a load of about 67, the measurement took 0.2 to 0.65 s of wall time
(0.3 to 0.75 s of CPU), the check of the antenna block 0.26 s, and the whole radiometric model 0.55 to 1.2 s (0.6
to 1.3 s of CPU, the first call the slowest). The model of 0.2.0, which read 512 pulses from the file and had no
such check, took 0.8 to 3.6 s (1.5 to 4.2 s of CPU) under the same conditions. A machine with no other load was
not available. When the machine was oversubscribed several times over (a load of 100 and more) the measurement of
0.2.0 took 52 to 181 s and the new one 0.4 to 0.9 s.

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
- `'black'`: where a scene of sigma zero `black_sigma0_db` (-10 dB by default) has an SNR of at least 1, that is,
  where the noise-equivalent sigma zero plus the ambiguity level is at or below -10 dB. The range swath is opened
  to the extended window TOAE1..TOAE2.

The two criteria use different reference scenes, on purpose.

| | Reference sigma zero | Why |
|---|---|---|
| RNIIRS (`'tasked_quality'`, a number) | 0 dB co-polarized, -6.02 dB cross-polarized: the RGIQE's own, sarpy's convention | An RNIIRS is a rating against a fixed scene. With the RGIQE's reference, a value computed here is the one sarpy's `PREDICTED_RNIIRS` states for the same image, and a tasked RNIIRS read from a vendor's SICD means the same thing in the model. |
| `'black'` | `black_sigma0_db`, -10 dB by default | The black edge is a statement about the ground: where terrain of an ordinary brightness is no longer above the noise and the ambiguities. A 0 dB scene is brighter than most terrain, so its SNR of 1 lies well past the last pixels that show anything. |

With the default the black area ends where the SNR of the RGIQE's reference scene is about 10 dB, not 0 dB. It
keeps most of what the wider area shows, at less area to form, less time and a smaller file
(the [measurements](#the-black-area-at--10-db) are below). `black_sigma0_db=0.0` gives the area out to an SNR of 1
of a 0 dB scene, which was the black area of 0.2.0 for a co-polarized channel.

How the areas nest: every area contains the tasked one; a lower RNIIRS threshold gives a larger area; a lower
`black_sigma0_db` gives a larger black area. `'black'` contains the area of every RNIIRS threshold that needs an
SNR of at least 10 dB of the reference scene (the 10 dB between the two references; 4 dB for a cross-polarized
channel). A threshold that needs less, such as a low RNIIRS at a large bandwidth area or a tasked quality that is
met at a low SNR, gives an area that reaches beyond `'black'` across the beam.

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
- An RNIIRS threshold that needs an SNR of the reference scene below 10 dB gives an area that exceeds `'black'`
  at its default of -10 dB.
- The pattern's pointing is checked against the data to about 0.2 half-power radii of the two-way beam at best,
  and passes within 0.4. A collection whose footprint fits neither the file's pointing nor the scene reference
  point within that tolerance has no trusted pattern: its SICD carries constant calibration factors, and no
  quality area beyond the tasked one is processed (`area` other than `'tasked'` raises a `ValueError`). Two of
  seven Umbra collections are such, halfway between the two pointings.

`process_cphd(..., area=, black_sigma0_db=-10.0)` takes the same four kinds. The image is formed over the bounding
rectangle of the area on the ground axes, with one impulse response everywhere and a noise-equivalent sigma zero
that rises toward the edges, and the pixels outside the area are set to 0. For a file that describes no antenna
pattern the model cannot predict where the signal fades, and an `area` other than `'tasked'` raises a
`ValueError`. The model's beam has the pattern the file's antenna block states and the pointing the data decide
([below](#the-antenna-patterns-pointing)); where no pointing agrees with the data, an `area` other than
`'tasked'` raises a `ValueError` that says so. On a moving beam the areas extend in ground range only, by the
same criteria and the same check of the pointing ([below](#areas-in-ground-range)).

### The antenna pattern's pointing

The areas and the calibration both rest on the two-way gain over the ground, so `Radiometry` checks the antenna
block against the data before it uses it, and the data decide how the pattern is pointed (`rad.pointing`, with
the numbers in `rad.footprint` and a sentence in `rad.notes`).

The measurement is the beam's footprint in the phase history: the range-Doppler power map of the pulses the noise
measurement uses (64 to 512 of them, from the history `process_cphd` has read). Each cell of delay and Doppler is
a patch of ground, and the model's map is the two-way gain over a ground lattice binned into the same cells, with
the azimuth ambiguities folded in and a fitted noise floor. The fit moves the modeled beam to where the two maps
(their logarithms) correlate best. A pointing passes when the fitted beam center lies within 0.4 of a half-power
radius of the two-way beam from the modeled one. That is 2 to 3 dB of gain at the half-power radius, and twice
the 0.2 radii by which scene content alone moves the fitted center
([how the tolerance was set](#the-pointing-and-the-width-against-the-images)).

1. `'file'`: the antenna frames as the file states them, when they pass.
2. `'reference point'`: when they fail, the same pattern (its shape, its orientation about the boresight and any
   electrical boresight terms) with each pulse's boresight turned onto that pulse's scene reference point, which is
   where a spotlight or a moving beam is steered. If that passes it is used, and the note gives both offsets.
3. `None`: neither passes, and the model has no trusted pattern. The areas other than `'tasked'` are refused and
   the SICD's scale factors are written as constants.

Where the check cannot be made the file's pointing is kept, and the note says that it is unverified and why: a
scene of a few point scatterers shows no footprint, and neither does one whose clutter is under the noise in the
range-Doppler map.

A moving beam (stripmap, sliding spotlight) is checked the same way, with two differences that follow from its
geometry. Each pulse has its own scene reference point, the echoes are referred to it, and the footprint moves
over the ground with the pulses, so the model takes the antenna's position and frame at the pulses of the
measured block and looks at the ground around their reference point; with the frame of another part of the
aperture the modeled footprint is displaced by the distance flown in between. And the check is made once, on the
pulses that form the middle of the strip, and serves every position along the track
([Areas in ground range](#areas-in-ground-range)). The second pointing, each boresight on its pulse's reference
point, is the natural one for a beam that is steered there.

The blocks of pulses of one measurement are added into one map in the echoes' own coordinates, delay and Doppler
about each pulse's reference point. The footprint of a stripmap stays in place there. That of a beam steered
along the track does not: where the beam center turns, as seen from the platform, by more than 0.2 half-power
radii between the first block and the last (a sliding spotlight, a squint that drifts), the blocks show no one
footprint, the check is not made, and the file's pointing is used with the note that it is unverified.

On a simulated stripmap with clutter under the beam its file states (`tests/test_moving.py`) the fitted center
lies 0.06 radii from the modeled one. With the frame of the middle pulse of the stretch, as for a beam held on
one point, it lies 1.00 radii away, the 35 m flown between that pulse and the measured block. With the file's
frames turned 1.2 radii across the track the file's pointing reads 1.19 radii and the reference points 0.06,
and the areas are those of the frames as simulated. On a Capella stripmap (2024-11-27; the 15,944 pulses of
the middle of the strip) the file's frames pass with the fitted center on the modeled one to the search's step
of a sixteenth of a radius, at a correlation of 0.85 (the frame of the stretch's middle pulse would put it 1.8
km away, 0.69 radii); the footprint is 1.40 and 0.97 times as wide as the modeled two-way beam along the
antenna's X and Y axes. On a Capella sliding spotlight (2024-08-29) the beam center turns by 17 radii over the
pulses of a pixel, and the check is not made.

Seven Umbra spotlight collections (2024-05 to 2025-01, satellites 05 to 08) and two Capella ones, by this rule:

| Collection | As the file points it | On the reference point | Pointing used |
|---|---|---|---|
| Umbra-07, 2024-09-10 | 2.00 km, 1.00 radii | 0.51 km, 0.26 | reference point |
| UMBRA-06, 2024-05-28 | 1.62 km, 0.83 | 0.37 km, 0.19 | reference point |
| UMBRA-06, 2024-07-14 | 1.24 km, 0.65 | 0.40 km, 0.21 | reference point |
| UMBRA-06, 2024-07-28 | 1.31 km, 0.70 | 0.22 km, 0.12 | reference point |
| UMBRA-06, 2024-10-26 | 1.08 km, 0.71 | 0.47 km, 0.31 | reference point |
| UMBRA-08, 2024-11-12 | 0.97 km, 0.49 | 1.09 km, 0.55 | none |
| UMBRA-05, 2025-01-19 | 0.82 km, 0.44 | 0.87 km, 0.47 | none |
| Capella, 2025-03-13 | 1.22 km, 0.46 | 0.61 km, 0.23 | reference point |
| Capella, 2025-01-19 | no fit | | file, unverified |

The offsets are those of the fitted beam center from the modeled one, in kilometers on the ground and in
half-power radii. In all seven Umbra files the per-vector frames (`TxACX`, `TxACY` and the same for receive, with
ACY along the velocity; `TxEB` and `RcvEB` zero; pattern arrays that peak within 2e-4 of their origin, 1e-3 along
X for UMBRA-05) put the boresight 0.15 to 0.18 degrees ahead of the scene reference point along the track, 1.4 to
1.6 km on the ground, and the data put the beam 0.8 to 2.0 km from there, toward the reference point in every
one. Five fail as the file points them and pass on the reference point. Two lie about halfway between the two
pointings and pass neither: their SICDs carry constant scale factors, and their quality areas are refused. The
first Capella collection fails as its file points it, by a small margin, and passes on the reference point; it
states no `PTRef`, so only its quality areas depend on the choice. The second Capella collection gives no fit: at
a range of 931 km its clutter stands 1.9 dB above the noise in the map.

One observation on the Umbra frames, which is not used as a rule: the stated offset is the rotation of the line
of sight over 0.19 to 0.21 s in each of the seven files. With the frames taken 0.20 s later than stamped the
boresight falls on the reference point in all seven (to 2e-4 along ACY), and the fitted offsets are 0.12 to 0.48
radii, against 0.44 to 1.00 as stamped and 0.12 to 0.55 on the reference point. The delay that fits each file
best scatters from 0.10 to 0.27 s (mean 0.17 s, standard deviation 0.06 s, which is the fit's noise), so the
footprints agree with one common delay near 0.2 s and cannot tell it from a boresight on the reference point.

### The beam's width

The same map gives the beam's width. With the modeled beam at its fitted center, the model is widened along the
antenna's X and Y axes until it fits the map best (least squares of the logarithms, over the noise floor of the
measurement, with one scale), and `rad.footprint['width']` holds the two factors: the measured footprint's width
over that of the modeled two-way beam. It is a record for the reader and changes nothing in the model. It is
`None` where the clutter stands less than 6 dB over the noise floor in the map. On simulated clutter the factors
come back within 0.15 of 1 under the stated beam, and 1.40 to 1.42 times larger where the file states a pattern
of twice the roll-off in dB, for which 1.41 is expected (`tests/test_calibration.py`).

The width answers one question about a file directly: whether its `AntPattern` is the one-way gain the standard
asks for, applied once on transmit and once on receive. A two-way pattern stored as one-way, or a model that
applied a pattern twice where the antenna applies it once, would make the measured footprint 1.41 times as wide
as the modeled one. In the seven Umbra files the transmit and the receive pattern are two entries with the same
content: the same `GainZero` (46.0 to 49.3 dB), the same gain array in dB whose peak is `GainZero` and whose
half-power width is 0.51 to 0.58 degrees, and the same per-vector frames. The two Capella files name one pattern
for both. The model evaluates the transmit and the receive pattern once each, adds their dB, and refers the sum
to the gain toward the scene reference point, so the absolute level of the arrays drops out.

| Collection | Across track | Along track | From 1,024 pulses |
|---|---|---|---|
| Umbra-07, 2024-09-10 | 1.050 | 1.025 | 1.06, 1.03 |
| UMBRA-06, 2024-05-28 | 1.075 | 1.050 | 1.06, 1.04 |
| UMBRA-06, 2024-07-14 | 1.125 | 1.025 | 1.12, 1.04 |
| UMBRA-06, 2024-07-28 | 1.100 | 1.050 | 1.11, 1.04 |
| UMBRA-06, 2024-10-26 | 1.150 | 1.075 | 1.15, 1.06 |
| UMBRA-08, 2024-11-12 | 0.950 | 1.000 | 0.96, 1.01 |
| UMBRA-05, 2025-01-19 | 1.050 | 1.150 | 1.06, 1.16 |
| Capella, 2025-03-13 | not measured | | 1.29, 1.01 |

The first two columns are `rad.footprint['width']` from the 64 or 128 pulses of the noise measurement (X is
across the track in these files and Y along it); the last is the same fit on 1,024 pulses with both widths
searched together. Over the seven Umbra collections the footprint is 1.07 times as wide as the modeled two-way
beam across the track (standard deviation 0.07) and 1.05 times along it (0.05): the pattern is applied as it
should be, once each way, and the beam is a few percent wider than the stored arrays, whose own polynomial
(`GainPoly`) is 5 to 10% wider along X than the array beside it. The Capella collection's clutter stands 4.7 dB over the
floor; from 1,024 pulses its width across the track is 1.29 with an interval of 1.02 to 1.90, and 1.01 along it
(0.92 to 1.12).

### The pointing and the width against the images

A second measurement of the same beam, which does not use the range-Doppler map, set the tolerance. For each of
the seven Umbra collections the default ground-plane SICD was formed, and its brightness (the noise level taken
out, cells of 200 m, the half of the cells with the least texture) was fitted with the file's two-way pattern
placed at a free center, with the azimuth ambiguities of each cell added.

- The center that fits the image lies 0.02 to 0.35 radii from the reference point and 0.61 to 0.97 radii from the
  file's boresight, in all seven. It lies 0.04 to 0.33 radii (0.18 on average) from the center the range-Doppler
  map gives, which is the scatter of either estimate.
- For the five collections that pass on the reference point, the brightness of the low-texture ground follows the
  modeled gain there with a slope of +0.70 to +0.80 dB per dB (1 for uniform ground under a perfect model; the
  ambiguities, the few percent of width and the residual pointing take the rest). Over all cells, towns
  included, the slope is +0.51 to +0.67: texture attenuates it. The width that fits these images best is 1.15
  to 1.30 across the track and 1.05 to 1.15 along it, from cells that reach 1.3 radii from the center.
- For the two collections that lie halfway, the gain as the file points it gives slopes of +0.45 and +0.31 over
  a modeled span of 22 dB, and the gain on the reference point +1.05 and +0.54. A tolerance of half a radius
  accepted the file's pointing for these two (0.49 and 0.44 radii) and wrote polynomials spanning 23 dB over
  images whose brightness follows a third to a half of that. At 0.4 neither pointing passes.

On the first collection the quality areas change with the pointing as follows (across by along the line of
sight, and the pixels of the SIDD):

| Area | As the file points it | On the reference point |
|---|---|---|
| `'tasked_quality'` | 7.65 x 6.75 km, 51.6 km^2, 192 Mpx (RNIIRS 3.72) | 5.95 x 6.70 km, 39.9 km^2, 148 Mpx (RNIIRS 4.28) |
| `'black'` | 7.90 x 9.45 km, 74.7 km^2, 285 Mpx | 7.75 x 9.05 km, 70.1 km^2, 267 Mpx |

The RNIIRS is the least over the tasked area, which is the threshold of `'tasked_quality'`: with the file's
pointing the modeled gain fell 19.6 dB within the tasked area, on the reference point 9.9 dB. The black area is
now centered on the reference point (-3.8 to +3.95 km across, -4.4 to +4.65 km along) where it reached from -4.7
to +3.2 km and from -3.3 to +6.15 km.

### The black area at -10 dB

Two spotlight collections with tasked areas of 5 by 5 km, at a reference of 0 dB, which was that of 0.2.0, and at
the default, with the pattern pointed as the data decide (both on the reference point; with the files' pointing,
as in 0.2.0, the areas at 0 dB were 8.30 x 11.35 km and 11.85 x 7.95 km). Sizes are across by along the line of sight,
and the pixels are those of the SIDD that `process_cphd(product='sidd', area='black')` plans:

| Collection | `black_sigma0_db=0.0` | `black_sigma0_db=-10.0` (default) |
|---|---|---|
| Umbra, 2024-09-10 | 8.15 x 11.10 km, 90.5 km^2, 348 Mpx | 7.75 x 9.05 km, 70.1 km^2, 267 Mpx |
| Capella, 2025-03-13 | 11.80 x 7.95 km, 93.8 km^2, 1,634 Mpx | 6.00 x 7.90 km, 47.4 km^2, 816 Mpx |

The Umbra area shrinks along the line of sight and hardly across it. Across the beam its edge is where the azimuth
ambiguities reach the signal, and their level follows the scene's own sigma zero, so that edge does not move with
the reference. The Capella collection has a noise-equivalent sigma zero of -13.8 dB at the scene center (an
uncalibrated level, from the measured clutter-to-noise ratio), and its area halves.

## Radiometric calibration

A spotlight SICD from `process_cphd` carries the SICD `Radiometric` block when the CPHD file supports it, and only
then. The product's `notes` say which source was used or why there is no block, and `out['radiometric']` holds
what was written (`None` without a block).

| The CPHD file states | The SICD carries |
|---|---|
| `PTRef` and `PNRef` | `NoiseLevel` and the four scale factor polynomials: a calibrated signal and a calibrated noise level |
| `PTRef` only (Umbra) | the same, with the noise level measured from the phase history: a calibrated signal and a measured noise level, and the note says so |
| no `PTRef` (Capella) | no `Radiometric` block |

No block is ever written from an assumed scene brightness. Two kinds of product have none whatever the file
states: one with `weighting='sva'`, because apodization is not linear in the scene (the peak of an isolated point
survives it, the clutter and noise levels do not, and a reader such as sarpy derives the missing factors from any
one that is present), and one of a moving-beam collection, which the model does not cover.

**What is written.** `NoiseLevel` has the type ABSOLUTE and a constant `NoisePoly`, the thermal noise power of a
pixel in dB. `RCSSFPoly`, `SigmaZeroSFPoly`, `BetaZeroSFPoly` and `GammaZeroSFPoly` are polynomials in SICD's image
coordinates (meters from the scene center point along rows and along columns) whose values, multiplied by pixel
power, give RCS in m^2, sigma zero, beta zero and gamma zero.

**Where the numbers come from.** FastSAR's formers sum the weighted phase history without normalization. Let
PTRef be the per-sample power of a 1 m^2 point at the scene reference point, PN the noise power per sample
(`PNRef / BNRef`, or the measurement), w the aperture weights of the N samples the image uses, rho the density of
those samples in the ground plane's spatial frequencies (N / A when they are spread evenly over a support of area
A, `rgiqe.support_area`) and G(x) the two-way antenna gain at the ground point x relative to the reference point,
with the spreading loss (`Radiometry.gain_db`). Then

```
peak power of a point of 1 m^2             (sum w)^2 PTRef G(x)
mean power of clutter of sigma zero 1      sum(w^2 rho) PTRef G(x)
mean power of thermal noise                sum(w^2) PN

RCSSF(x)       = 1 / ((sum w)^2 PTRef G(x))
SigmaZeroSF(x) = 1 / (sum(w^2 rho) PTRef G(x))
BetaZeroSF(x)  = SigmaZeroSF(x) / cos(SlopeAng)
GammaZeroSF(x) = SigmaZeroSF(x) / sin(GrazeAng)
NoisePoly      = 10 log10(sum(w^2) PN)
```

For `'uniform'` the sums over w are N; for `'taylor'` they are those of the two Taylor windows, and RCSSF /
SigmaZeroSF is about 1.52 times its unweighted value. The density is taken sample by sample
(`rgiqe.clutter_gain`): the pulses of a squinted or curved track are not evenly spaced in angle, which moves the
clutter sum by up to 0.1 dB under a Taylor window. The noise-equivalent sigma zero, the noise power times
SigmaZeroSF, is PN A / (N PTRef G(x)) for samples spread evenly, whatever the weighting. The baseband step,
autofocus, terrain-informed focusing and the resampling to the slant plane do not change the scale, and the
factors belong to the ground point: a ground-plane and a slant-plane SICD of one collection state the same values
at the same place, each in its own coordinates. A product focused with a `dem` carries the block of the product
without one, coefficient for coefficient ([below](#terrain-informed-focusing)).

The image is not corrected for the antenna pattern (`ImageBeamComp` NO), so the factors rise toward the edges as
1 / G(x). Each polynomial is a constant times one fit of 1 / G(x) over the image by least squares in the relative
error, of the lowest order (the same in both coordinates, at most 8) that stays within 0.1 dB of the model on a
lattice finer than the one fitted. The note states the order and the residual. Order 8 follows a gain that falls
by about 30 dB over the image; where an area asked for reaches further down the beam, the residual stated is
larger than 0.1 dB. A file with `PTRef` and no antenna pattern gets constant factors.

**The pattern's pointing is the one the data decide.** The variation of the factors over the image is only as
good as the modeled gain, so the factors follow the pattern as `Radiometry` has pointed it
([the rule](#the-antenna-patterns-pointing)): as the file states it where that agrees with the beam's footprint in
the phase history, with each boresight on its scene reference point where the file's pointing fails and that one
passes, and not at all where neither does. In the last case the scale factors are written as constants, their
values at the scene reference point, as the vendors' own SICDs carry them (`out['radiometric']['constant']`).
The note in the product says which pointing was used and gives the offsets.

**For a consumer.** All four factors are written, so none needs deriving. SICD's relations between sigma zero,
beta zero and gamma zero hold exactly, with the angles of the scene center point. The relation between RCS and
beta zero through the area of a resolution cell, which sarpy computes as 1 / (`Row.ImpRespBW` x `Col.ImpRespBW`)
and takes for a slant-plane area, does not carry over to a ground-plane SICD: those are ground-plane bandwidths
there, and the bounding rectangle of a support that squint shears. Read the polynomial of the quantity wanted.

**Measured.** In `tests/test_calibration.py` a simulated collection flown both ways, with an antenna beam that
peaks off the scene center and with PTRef and PNRef in its file, is processed on both planes with both weightings,
and the polynomials of each written file are read and evaluated by sarpy. Point targets of 0.5 to 8 m^2 come back
within 0.07 dB of their RCS, clutter of -13 dB within 0.02 dB of its sigma zero (beta zero and gamma zero
likewise), and the noise power within 0.02 dB of `NoisePoly`. The fits are of order 5 or 6 and follow 10 to 11 dB
of gain to 0.03 to 0.08 dB. A unit point target and unit-variance noise through the formers give the sum of the
weights to within 0.07 dB and the sum of their squares to 0.03 dB on both planes, with autofocus on and off: the
formers' own accuracy, with no normalization to correct for. sarkit's consistency check passes with the block
present, also beside the terrain step of a product focused with a DEM (`tests/test_sicd_consistency.py`).

Against the vendor's SICD of one Umbra spotlight collection (2024-09-10; PTRef, no PNRef), over the same ground
through each file's own projection and its own block. The vendor's factors are constant over its image (to
0.06 dB). FastSAR's follow the file's pattern with the boresights on the scene reference point, which the data
chose over the file's pointing (0.26 against 1.00 half-power radii); the gain falls 9.9 dB over the tasked
rectangle and the fit is of order 6, to 0.03 dB.

- At the scene center the two agree: over the central 500 m FastSAR's sigma zero is 0.4 dB above the vendor's
  (-8.4 against -8.8 dB).
- Over 64 boxes of 500 m within 2 km of the center FastSAR less vendor is +1.95 dB on average, with a standard
  deviation of 1.18 dB and a range of +0.05 to +4.82 dB (slant-plane product: +1.94, 1.22, +0.03 to +4.72). The
  difference is the modeled gain, which the vendor's constant factors leave in the image. The footprint in the
  phase history has the width of the modeled two-way beam (1.05 and 1.03), and over ground of low texture the
  image darkens by 0.78 dB per dB of the modeled gain
  ([above](#the-pointing-and-the-width-against-the-images)); over all cells, with the town at the scene center,
  the slope is +0.54. The residual pointing (0.26 radii) leaves the factors uncertain by about 1.5 dB at the half-power
  radius.
- With constant factors (`rgiqe.calibration(..., constant=True)`) the same boxes differ by +0.06 dB on average
  (standard deviation 0.26 dB, range -0.85 to +1.00). With the file's pointing they differed by -1.5 to +9.9 dB.
- Noise. The vendor's `NoisePoly` times its sigma zero factor is -71.5 dB, and its `theoretical_nesz_db`
  parameter states -30.8 dB. FastSAR's noise level, measured from the phase history, gives a noise-equivalent
  sigma zero of -24.8 dB at the scene center point. The darkest 100 m cells of the image lie at -18 dB by the
  vendor's block, an upper bound.

A Capella spotlight collection (2025-03-13) states neither PTRef nor PNRef, so its FastSAR SICD has no
`Radiometric` block, the note says why, and the step costs nothing. The vendor's SICD of that collection states a
sigma zero factor of -67.8 dB and a noise level of 49.4 dB at its scene center point, a noise-equivalent sigma
zero of -18.4 dB, from calibration data that the CPHD file does not carry.

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

## Terrain-informed focusing

`process_cphd(..., dem=)` removes the defocus of terrain that lies off the image plane. The correction is
computed from the collection geometry and the DEM (`fastsar.terrain`); nothing is estimated from the image, and
the image stays on its plane.

```python
import fastsar
out = fastsar.process_cphd('scene_CPHD.cphd', 'scene_SICD.nitf',
                           dem='Copernicus_DSM_COG_10_N19_00_W156_00_DEM.tif', geoid=22.4)
out['terrain']                     # applied, quadratic_rad, height, offset, relief, layover, block, order, ...
```

That call writes the default product, a ground-plane SICD. The correction is made on the ground image as formed,
after the baseband step and before the autofocus and any resampling, so `plane='slant'` and `product='sidd'` take
the same corrected image through their own stages. It is for spotlight collections.

**What it corrects.** The fast formers focus on a horizontal plane at the scene height. A scatterer at height
`h` above the plane appears at the plane point that has its range and range rate at the center of aperture
(layover, which `products.geolocate` and `geocode_image` undo when given a height). On a curved track it is
also defocused: its range history differs from that of the plane point by a phase that is quadratic and higher
across the aperture. To first order in `h`, with `t` the time from the center of aperture, `R` the range, `w`
the unit normal of the slant plane, `w_z` its vertical component and `A_w` the antenna's acceleration along `w`
in the scene's frame,

    phi(t) = 2 pi h A_w t^2 / (lambda R w_z)

so that at the ends of an aperture of length `L` flown at speed `v`

    phi_edge = pi / 2 (h / lambda) (L / R)^2 (R / rho) / w_z,        rho = v^2 / A_w

`rho` is the radius of curvature of the track out of the slant plane. A straight track gives zero at any
height. The phase is proportional to the height and to the square of the aperture, and for a given azimuth
resolution `lambda / (2 L / R)` it grows with the wavelength and with `R / rho`. From a circular orbit of radius
`r` looking broadside, `(R / rho) / w_z` is about `R R_e / r^2` (`R_e` the Earth's radius), a tenth from low
orbit.

*A worked number.* An X-band spotlight (31 mm) from 6,965 km orbit radius at 931 km slant range has
`R R_e / r^2 = 0.12`. The 2.9 degrees of aperture that give 0.31 m of azimuth resolution then put 1.65 rad at
the band edge per 100 m of height (the closed form above gives 1.64, the exact model 1.65): the usual tolerance
of pi / 4 is reached 48 m from the plane. With the whole 5.4 degree aperture it is 5.7 rad per 100 m, and the
tolerance 14 m. That collection is of the Kilauea caldera, whose surface lies between 120 m below and 100 m
above the plane in the Copernicus DEM, and since the collapse of 2018 more than 100 m lower than that over 3.7
square kilometers.

**The DEM and its datum.** `dem` is a GeoTIFF path (any CRS; a list of paths for tiles), a function
`dem(lat, lon) -> height` in meters above the WGS-84 ellipsoid (the convention of `products.read_dem`,
`geolocate` and `geocode_image`), or a number, a surface of that ellipsoid height. Most DEMs give heights above
a geoid: Copernicus GLO-30 and GLO-90 above EGM2008, SRTM above EGM96. `geoid` is the height of the geoid above
the ellipsoid at the scene (the undulation `N`), which is added: `h_ellipsoid = H_dem + N`. It is +22 m in
Hawaii and -26 m in North Dakota, and lies between -106 and +85 m. A DEM file is refused without it (`geoid=0.0`
states that the file holds ellipsoid heights). `N` can be read from a public geoid grid, for example PROJ's
`us_nga_egm08_25.tif`, with `products.read_dem(grid)(lat, lon)`. As a check, the median of the DEM over the image
is compared with the image plane, which is at the scene height the collection states: when they differ by more
than 10 m plus three times the DEM's standard deviation over the image, a `terrain.TerrainWarning` is raised and
a note is left in the product. That catches a forgotten offset on flat ground, a wrong tile or wrong units; it
cannot see an offset of 20 m under 200 m of relief.

**How.** `terrain.heights` finds, on a lattice of about 10 m over the image, the height of the surface that
each pixel shows: the point of the DEM on the pixel's circle of constant range and Doppler, by a scan over
height. Where the circle meets the surface more than once (layover: a slope steeper than the line of sight),
the heights of the crossings are averaged with the ground area each contributes as the weight; outside the
DEM nothing is corrected. A list of DEM tiles on one grid (adjacent Copernicus tiles are) is joined into one
raster first, so that heights are interpolated across the seams between tiles.

`terrain.coefficients` gives the phase as a polynomial in azimuth frequency from the antenna positions
themselves, with no small-angle expansion; its constant and linear terms place the scatterer at its layover
point with the phase of its range at the center of aperture, which is where and how the SICD's projection
expects it. The degree is the lowest that fits within a hundredth of a radian: 2 or 3 from orbit, up to 8 for a
short range and a track with a jerk. That is the phase at the center frequency. Across the range band the phase
scales with frequency, and each pulse's azimuth frequency moves with it, so that over the image's 2-D spectrum
the phase also depends on the range frequency: the range walk, about `B / (2 fc)` of the phase at the corners of
the band (from orbit it is close to `-w c2 v^2`, with `w` the range frequency as a fraction of the center
frequency). Left in place it limits the correction to about -34 dB at 3 rad of phase and -28 dB at 6 rad. The
correction fits the phase over both frequencies, to the exact phase at every pulse and five frequencies.

The phase is removed in strips of about 64 m along azimuth, each read with as many rows again on either side as
the phase displaces, on the device that holds the image, with exact zeros kept. Within a strip, column by
column, each pixel is corrected for the height it shows: the strip is corrected for a few heights spanning the
column's heights over the strip (Chebyshev nodes, as many as keep the interpolation in height within 0.025 rad
at the band edge: from one on level ground to five on Etna's steepest slopes), and each pixel takes the
interpolation between those corrections at its own height. Exact backprojection onto the surface does the same,
pixel by pixel. The range walk is proportional to the height; it is removed by a short series (one or two terms
from orbit), each term applied along azimuth with the strip's correction and along range through each row's
spectrum, and scaled by the pixel's height. A strip takes the geometry of its middle row, which from orbit
changes by under a thousandth along 64 m; at a short range the strips are shortened until it holds. The stage
runs on the baseband ground image before the data-driven autofocus, which then sees the residual: what the DEM
got wrong, and the errors that have nothing to do with terrain.

**What the product says.** `out['terrain']` holds what was applied (`applied`, `powers`, `edge_rad`,
`quadratic_rad`: the least and greatest quadratic coefficient in radians at the band edge; `height`; the DEM's
`offset` from the plane and its `relief`; the `layover` and `outside` fractions; `levels`, the most heights a
strip column was corrected for, and `height_rad`, the largest half range of phase across one column's heights
in a strip; `walk_rad`, the largest range walk at the band's corners, `walk_terms` its series, and
`walk_left_rad`, any walk left uncorrected), `out['notes']` a line, and
`out['timing']['terrain']` the stage's seconds. The grid, the geolocation and the SICD's projection are
unchanged: the SICD is still `Grid.Type` PLANE on the same plane, and it records the step as an
`ImageFormation.Processing` element (`TERRAIN PHASE CORRECTION`). The image is not orthorectified; heights
enter geolocation as before.

**The exact route.** Exact backprojection takes any points, so an image can be formed on the surface itself, at
the exact kernel's cost. `terrain.surface_points` gives the surface point each pixel shows, and the result is the
image on the same pixel grid with every scatterer of the surface focused:

```python
import numpy as np
import fastsar
from fastsar import io, terrain

out = fastsar.form_cphd('scene_CPHD.cphd', axes='los')                   # the pixel grid, on the plane
dem = terrain.as_dem('dem.tif', geoid=22.4)
i, j = np.meshgrid(np.arange(4000, 4512), np.arange(6000, 6512), indexing='ij')
points = terrain.surface_points(out, dem, i, j)                          # [512, 512, 3], local frame
col, meta = io.read_cphd('scene_CPHD.cphd', meta=True)
p0, p1 = out['pulses']                                                   # the pulses the image used
chip = fastsar.backproject(col['S'][p0:p1], col['ant'][p0:p1], col['fmin'], col['df'], points,
                           ref=meta['ref'][p0:p1], backend='cuda')
```

`ref` is the range each pulse's samples are referenced to, as the file has it (`meta['ref']`). Without it
`backproject` takes the distance from the antenna to the frame's origin, which is not the reference of every
collection: the samples of a Capella collection are referenced to the mean of the two legs of the path from the
transmit to the receive position, 0.14 mm longer, and leaving the argument out puts 0.055 rad on such an image.
In `tests/test_terrain.py` the route as written here gives the chip of an independent exact reference to -60 dB
of its peak, on a collection referenced to the antenna's distance and on one referenced 0.14 mm beyond it, where
the call without `ref` is off by the 0.055 rad.

**Simulated** (`tests/test_terrain.py`). A squinted spotlight at 10 km on a curved track with a jerk, flown to
either side, with point targets 30, 100 and 300 m above and below the plane: phases of 12, 40 and 120 rad at
the band edge, the cubic term the largest. On a straight track the phase is zero at any height. Uncorrected,
the targets are 1.8 to 2.1, 2.7 to 2.9 and 4.1 to 4.2 times as wide in azimuth as a target on the plane, with
0.50, 0.34 and 0.24 of its peak, and the phase of their azimuth spectra is the model's within 0.01 rad at 30 m
and 0.06 rad at 100 m. Corrected, the targets at 30 m have the on-plane width within 0.4%, its peak sidelobe
within 0.16 dB, 0.99 of its peak or more, and lie within 0.003 pixel of their layover point; as complex chips
they differ from exact backprojection at their true height by -39 to -51 dB of the peak (-19 to -26 dB with
0.3.0's correction, the phase at the center frequency alone), from -1 to -4 dB uncorrected. At 100 m, where the
range walk reaches 5 rad at the band's corners, three targets have the on-plane widths within 0.7% and 0.99 of
the peak, -34 to -40 dB from exact backprojection (-11 to -14 dB before); the fourth lies in a strip whose walk
is beyond the series (widths within 4.4%, 0.93 of the peak, -14 dB). At 300 m the walk is 13 to 19 rad, two
range cells, and is left: widths within 38 to 50% and 0.65 to 0.71 of the peak, as before.

A cliff 300 m high across azimuth (the surface rises 3 m per meter, from 150 m below the plane to 150 m above
it), covered with scatterers of random phase, on the same track bent a twentieth as much (0.008 rad of quadratic
phase per meter of height at the band edge, as from orbit): over a chip of 128 by 128 pixels that holds the
whole cliff the corrected image differs from exact backprojection onto the surface by -35 dB, the plane's image
by -2.5 dB, and 0.3.0's correction by -9 dB. The fast former is within -42.6 dB of exact backprojection on the
plane there, and the exact phase applied pixel by pixel through the local spectrum reaches -35 dB as well: at
10 km the curvature of the wavefronts over the correction's kernel is what is left.

The same collection written with `PTRef`, `PNRef` and an antenna beam, as the default ground-plane SICD and as
the slant-plane one: the SICD carries the terrain step and the `Radiometric` block together, the block is that of
the product without a DEM, coefficient for coefficient, and the energy of the image is unchanged (the correction
is a phase over the azimuth spectrum). Through the written `RCSSFPoly` the target on the plane reads +0.03 dB
(+0.06 on the slant plane) of its RCS, and the target 30 m above it reads -5.9 dB in the plane's image and -0.1
dB once focused with the DEM. A raised target is seen with the beam's gain at its true position and scaled by
the factor at the plane point it lays over onto, which the comparison takes out.

**Measured on Kilauea.** Two Capella spotlights of the caldera, formed with the default square-resolution
aperture: one broadside at 931 km (2.9 degrees of aperture, -1.65 rad per 100 m of height, a 17,489 by 17,093
pixel ground image), one squinted 30 degrees at 611 km (2.3 degrees, -0.83 rad per 100 m). Heights are from
Copernicus GLO-30 with the EGM2008 undulation at the scene (22.4 m, PROJ's `us_nga_egm08_25.tif`). The test of
the model is the image itself: the quadratic coefficients that map drift estimates on 512-pixel blocks of the
uncorrected image, against the model's predictions for the same blocks.

- Over the whole first scene (1,886 blocks, weighted as the spatially variant autofocus weights them) the
  estimates correlate with the predictions at 0.76, with a slope of 1.35 +- 0.11, an intercept of +0.57 rad and
  an rms residual of 1.27 rad (the errors are a jackknife over sixteen parts of the scene). The excess is the
  DEM's age: it predates the 2018 collapse of the caldera floor. On the blocks where a 2019 lidar DEM (USGS,
  doi:10.5066/P9F1ZU8O) agrees with GLO-30 within 10 m (620 blocks) the slope is 1.05 +- 0.24 (1.06 +- 0.05
  unweighted; correlation 0.56, intercept +0.16 rad, rms residual 1.2 rad, the size of the scatter of the block
  estimates). With the lidar DEM's heights the slope is 1.10 +- 0.13 outside the pit that lava has been filling
  since 2020. Scaling the DEM's heights for the highest contrast of the unchanged low ground (15 million
  pixels, every eighth column) gives a factor of 0.99.
- On the second scene the predictions span only -0.7 to +0.7 rad against a block scatter of 0.8 rad: correlation
  0.35, slope 1.37 +- 0.28 over all blocks and 1.35 +- 0.27 on unchanged terrain (1.27 +- 0.08 unweighted),
  intercept -0.2 rad. The sign is the model's; the scale is not resolved there.
- With GLO-30 the correction raises the contrast of the first image by 0.8% and narrows 88 bright compact
  points by 4.8% in azimuth (median; 5.3% and 3.8% more peak amplitude where the predicted phase is 1 to 2 rad).
  With the lidar DEM the points where the phase is 2 to 9 rad go from 0.53 to 0.46 m, with 10% more amplitude,
  and the rms of the block estimates falls from 1.94 rad to 0.74 rad (1.47 rad with GLO-30).
- The default quadratic autofocus declines the uncorrected image because its parts disagree (-2.1 to +2.8 rad
  over eight parts of the scene). After the GLO-30 correction they still do, in the collapsed area. After the
  lidar correction the parts agree on a residual of -0.35 +- 0.08 rad, which the contrast test then declines:
  it is too small to raise the contrast of an image whose pit is corrected for the wrong depth.
- Against exact backprojection onto the DEM's surface on the four 256-pixel chips the validation harness picks
  from the first image with GLO-30: 75 m above the plane (1.2 rad), -54.9 dB (0.3.0's correction: -37.1 dB);
  68 m below it (1.1 rad), -53.7 dB (-39.3); near the plane, -53.7 dB (-49.9); 122 m below it (2.0 rad), -52.8 dB
  (-35.0). The plane's image is within -53.2 to -55.2 dB of exact backprojection on the plane there.
- Against exact backprojection onto the DEM's surface, on three 256-pixel chips of the first image (590,000
  points; the rms of the complex difference over the rms of the chip): the plane's image differs from it by -11.5
  dB on high ground (+74 m) and the corrected image by -48 dB; in the collapsed area with the lidar surface (-190
  to -230 m) by -4.5 dB and -38 dB; near the plane (-30 to +5 m) by -26 dB and -50 dB. The fast former itself
  is within -52 dB of exact backprojection on the plane. The exact route took 215 s for those points on the
  shared GPU, 0.003 megapixels per second, where formation and the correction make 3.3: the surface costs about
  a thousand times the plane with its correction.
- The stage takes 1.9 to 5.8 s for the 206 and 299 megapixel images on a GPU shared with other work (50 to 140
  megapixels per second) and 3 to 15 s on the host's 48 shared cores (19 to 72 megapixels per second), of which
  0.4 to 1.0 s is the height map; formation took 40 and 84 s on the same GPU.
- A flat control, an Umbra spotlight of Fargo, North Dakota: with the EGM2008 undulation (-26.2 m) the DEM's
  median is 1.6 m below the image plane with 1.8 m of relief, the predicted phase stays under 0.012 rad, nothing
  is applied and the product is the one without a DEM, sample for sample; the stage takes 0.6 s for its height
  map. With the undulation left at 0 the DEM lies 25 m above the plane and the warning says so.

**Measured on Etna.** A Capella spotlight of Mount Etna (2024-07-12, 500 MHz at 9.65 GHz, an 11,109 by
10,871 pixel ground image of 0.45 m), whose surface lies between 1,120 m below and 570 m above the image plane
in Copernicus GLO-30 (two tiles, the EGM2008 undulation of 43.6 m): 0.0057 rad of quadratic phase per meter of
height at the band edge. Against exact backprojection onto the DEM's surface, on the four 256-pixel chips of the
validation harness (the rms of the complex difference over the rms of the exact chip):

- 500 m above the plane (496 to 537 m, 2.8 to 3.0 rad): -51.6 dB; 0.3.0's correction -33.1 dB, uncorrected +0.7.
- Below it, 588 to 296 m within one chip (1.7 to 3.3 rad): -49.0 dB; before -29.3.
- Near the plane (-28 to +33 m): -51.6 dB; before -47.6.
- 1,050 m below it (5.9 to 6.1 rad): -49.6 dB; before -25.2.
- The plane's image against exact backprojection on the plane: -50.9 to -52.4 dB, which bounds the comparison.

Over the whole image, on 72 chips of 128 pixels drawn three to a cell from every combination of phase (the
chip's median, 0 to 8 rad at the band edge) and steepest slope inside the chip (under 0.2 to over 2.5 m per
meter, up to 225 m of relief within one chip), the corrected image differs from exact backprojection onto the
surface by -50.2 dB in the median and -44.7 dB at worst (0.3.0: -35.0 and -25.2 dB):

| phase at the band edge | 0.3.0, median of the cells | now, median of the cells | now, worst chip |
| --- | --- | --- | --- |
| under 0.5 rad | -45 to -49 dB | -51 to -53 dB | -47 dB |
| 0.5 to 1.5 rad | -36 to -45 dB | -48 to -52 dB | -47 dB |
| 1.5 to 3 rad | -34 to -35 dB | -49 to -51 dB | -45 dB |
| 3 to 5 rad | -28 to -31 dB | -48 to -50 dB | -46 dB |
| 5 to 8 rad | -25 to -27 dB | -48 to -50 dB | -46 dB |

Slope costs little once each pixel has its own height: the steepest cells (over 2.5 m per meter, 70 to 225 m of
relief inside a 58 m chip) are at -48 to -53 dB, within about 3 dB of the gentlest at the same phase, where
0.3.0's correction lost up to 9 dB to them.

What limited 0.3.0's correction there, measured on the same chips with the correction's parts varied: the range
walk. With each pixel's exact height but the phase along azimuth alone the chips reach only -34.6, -36.0, -50.7
and -27.6 dB; with the phase over both frequencies and a constant height per 64-pixel tile -43.4 dB on the
first and -45.6 dB on the deepest; with both, -52.2, -52.2, -50.9 and -50.1 dB. On the chip below the plane
the height averaged over strips of 64 m also cost 4 dB (-30.8 against -34.6 dB). Shorter strips did not help
the old correction (-24 dB at 14 m). Layover played no part: 0.04% of the lattice. The two DEM tiles' seam,
where 0.3.0's mosaic left a band without heights (1.1% of the image uncorrected), is now interpolated across.

**Limitations.**

- *The DEM.* The correction is as good as the heights: 10 m of error is 0.17 rad on the first collection above
  and 0.6 rad with its whole aperture. A DEM older than the terrain (the caldera above), a surface model where
  the radar sees the ground or the reverse, and the geoid offset all enter directly. The data-driven autofocus
  runs after the correction and sees what is left.
- *One height per pixel.* Each pixel is corrected for the height it shows, as exact backprojection onto the
  surface is formed. In layover the heights that share a pixel are averaged (as `surface_points` takes them), so
  each scatterer is corrected for the wrong height by up to the relief folded into the pixel. The interpolation
  in height stops at eight heights per strip column, which covers 4.3 rad of phase across one column's heights
  within a strip at the band edge; Etna's steepest strips need 1.6 rad and five heights. `terrain.focus(block=)`
  sets the rows of a strip.
- *Range walk.* The walk is removed by a series of at most eight terms. Beyond about 4.5 rad of walk at the
  band's corners the series would not converge, and the walk of that strip is left as 0.3.0 left it everywhere
  (`walk_left_rad` says how much): from orbit the walk is about `B / (2 fc)` of the phase, 0.25 rad on Etna's
  deepest ground, while the simulated 10 km collection reaches 13 to 19 rad at 300 m.
- *Short range.* The correction treats the image's spectrum near a pixel as plane waves. At 10 km the
  wavefronts' curvature over the correction's kernel limits the simulated 300 m cliff to -35 dB against exact
  backprojection onto the surface (the exact phase applied pixel by pixel through the local spectrum reaches the
  same), where the former itself is within -42 dB on the plane; from orbit the effect is negligible.
- *Spotlight only.* A moving-beam collection is refused. The model holds for it pixel by pixel, with the pulses
  that illuminate the pixel and their own center of aperture; the correction would run per patch of the mosaic.
- The image is not moved: layover stays, and geolocation needs the height as before.

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
| `plane` | ground (the default) or slant | ground (`'slant'` is refused) |
| `weighting='uniform'`, `'taylor'` | on the phase history | frequency window on the phase history, azimuth window per pixel over its processed band |
| `weighting='sva'` | on the slant plane | on the ground grid, where the spectrum is a rectangle on its axes (see the limits below) |
| `autofocus='quadratic'` | one error for the image | blocks along the track, tied together by a smooth fit |
| `autofocus='spatially_variant'` | blocks and a polynomial over the scene | the same blocks along the track, with a parabola across the swath |
| `autofocus='pga'`, `'spatially_variant_pga'` | yes | refused: one phase function per aperture, and every block has its own |
| `area` other than `'tasked'` | in both directions | in ground range; the tasked extent along the track |
| `dem` | terrain-informed focusing | refused |
| `Radiometric` block of the SICD | where the file states `PTRef` | none, with a note |

The defaults are the same for both: an unweighted SICD and an apodized SIDD, on the ground plane, with quadratic
autofocus and the tasked area. One rule is the moving beam's own. Where apodization cannot be applied to a
collection (the limits are [below](#weighting)) and `weighting` was not given, the SIDD is Taylor weighted
instead, and a note in the product, which begins with `weighting:`, gives the reason; `weighting='sva'` given by
name raises a `ValueError` with that reason. `out['weighting']` states the weighting the image has.

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
t_c is solved at knots every 32 pixels, and F is evaluated per pixel in double on the host's cores or in a CUDA
kernel. Every `process_cphd` product of a moving beam is at baseband, and its SICD states a band centered at
zero.

JAX's device works in float32 (a TPU has no fast float64), which cannot hold 4 pi fc / c times a range of 600 km
to a thousandth of a radian. The host therefore solves psi in double along the center row and the center column
of the grid, at every pixel of both, and the device adds the two vectors, reduced modulo 2 pi. What is left
vanishes on that row and that column and is small and smooth: zero for a straight track and a constant squint,
15 rad over 30 by 6 km of a stripmap from an orbit at 600 km, 115 rad over the 5 by 5 km of a sliding spotlight.
Its part in the product of row and column is taken in integer turns, as `planes.deramp` takes its own, and the
rest is solved at the knots, split into its singular vectors (one or two terms) and carried to the pixels by a
cubic spline of each; the device sums the terms. Against the phase in double at every pixel of the first, the
middle and the last rows of such grids the float32 phase is off by at most 2e-5 rad
(`tests/test_moving.py::test_baseband_phase_in_float32`), and the host's part takes about two seconds for a grid
of 500 megapixels.

The support is a rectangle on the grid's axes only approximately. The band along the track grows in proportion to
the frequency (a trapezoid, 3% wider at the top of a band of 3% of the carrier), a squint moves its center with
frequency and shears the range band by tan(squint) of the azimuth band, and cos(graze) changes the ground range
band across the swath.

### Weighting

`'uniform'`, the default of a SICD, is a rectangle over each pixel's processed azimuth band and no frequency
window. `'taylor'` is the Taylor window (nbar 4, 35 dB) along frequency and, at each pixel, over its processed
azimuth band, zero outside it: what Taylor weighting means for a spotlight, applied to the pulses a pixel takes.
`form_cphd` itself keeps the Hann window it has always applied there (`azimuth_window='hann'`), whose main lobe
is 22% wider and whose first sidelobe is 3.5 dB higher; `azimuth_window='taylor'` or `'uniform'` selects the
others. The weights enter the
final stage of the factorized backprojection of each patch, per subaperture and tile with their slope across the
tile. The rectangle and the Taylor window's pedestal end in a step at the edge of the band, which the final stage
places by subaperture: a pixel's band ends within one final subaperture of the pulses its beam gives it (24 of
1,000 pulses on the simulated stripmap). Every backend places it at the same pulses, the jax and tpu backends by
laying each patch out as the cpu and cuda backends form it, whatever pulse count and range gate their compiled
program is shared at ([docs/stripmap-burst.md](stripmap-burst.md)). Measured on a simulated broadside stripmap, a
3 degree squinted one and a sliding spotlight (`tests/test_moving.py`; widths in units of the reciprocal band,
azimuth then range):

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

What does not fit the rectangle is not apodized. `weighting='sva'` then raises a `ValueError` that names the
reason, and a SIDD whose weighting was left to the default is Taylor weighted with that reason in its notes:

- a squint that shears the range band by more than 10% of its width across the azimuth band (`moving.SHEAR`);
- a ground range band that changes by more than 5% across the swath (`moving.RANGE`);
- a processed azimuth band that differs by more than 10% over the scene (`moving.VARY`), as under a beam whose
  speed over the ground changes;
- a beam center that does not move smoothly (an SRP that steps).

A `spacing` has the same effect, since apodization sets its own: with `weighting='sva'` it raises, and with the
default of a SIDD the image is Taylor weighted at the spacing asked for.

Inside these limits an isolated point keeps its apodized sidelobes: on an ideal rectangle, a shear of 10% and a
range sampling 5% off twice Nyquist both leave the highest pixel outside the main lobe at -48 dB, where the exact
rectangle has it at -48 dB and the unweighted image at -17 dB. On the simulated
collections of `tests/test_moving.py` the highest pixel within 24 pixels of a target, outside the 7 by 7 pixels of
its main lobe, is at -42 to -53 dB of the peak on the broadside stripmap, -44 to -46 dB on the sliding spotlight
and -42 to -51 dB under a squint of 0.5 degrees, against -17 to -18 dB unweighted; the 3 dB widths are 0.83 to
0.90 of the reciprocal bands (0.886 unweighted) and the peaks keep their unweighted amplitude within 5%. A squint
of 3 degrees shears the range band of that collection by 11%: `weighting='sva'` is refused, and the default SIDD
of that collection is the Taylor-weighted one.

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
  estimate is in radians at the edge of the synthetic aperture. The estimate reads a Taylor-weighted copy of
  those columns (nbar 4, 35 dB: on the two looks along the track, and across range a column with its neighbors
  one resolution cell to either side), and the correction is applied to the image as it was delivered, so the
  estimate does not depend on the product's weighting. In an unweighted image a bright scatterer whose range
  moves over its aperture leaves different parts of its band in two neighboring columns; the two looks of
  either column are then different parts of the scatterer, their correlation has two peaks, and the iteration
  alternates between them. On a Capella sliding spotlight an injected error of 15 rad was not found in the
  unweighted product before this copy was used. A Taylor-weighted product (`weighting='taylor'`) is read as it
  is.
- **Capture range.** An error of c rad at the band edge moves the two looks apart by about c / (pi hf) rows,
  and the correlation peak is sought within a quarter of a block, so a block captures errors up to about
  pi hf B / 4 rad for B rows at a band of hf cycles per row (`info['capture']`): 96 rad for 256 rows of an image
  sampled at 0.96 of its band. Beyond it another peak is found and the second test below returns the image as
  it came.
- **Guard, first test.** The block's iteration must have settled (its last update under ten times the
  tolerance, `info['settled']`): an estimate that still alternates between two peaks is that of neither. Then
  `quadratic`'s, on every block: the correlation peaks of all columns and of at least half
  of the sets must be significant, the estimate of all columns must lie within three standard errors of the sets'
  median, and no more than a quarter of the sets may lie far from it. A block that passes counts, also when its
  estimate is small: it says that its part of the strip is focused.
- **Fit.** Along the track the estimates of the blocks that count are tied by a robust local line (tricube
  weights over four block steps, Tukey's biweight); a block that does not count takes the fit of its neighbors.
  Across the swath the departures of a block's agreeing sets from their median give a linear trend where it is
  three times its standard error (`'spatially_variant'`: a parabola, always), held at its end values beyond the
  columns the sets cover; where there is a trend, the block's estimate of all columns is made again with the
  trend removed column by column, because the columns in best focus would otherwise carry it.
- **Guard, second test.** `quadratic`'s rule on the corrected rows: their contrast must rise, the median gain of
  the parts must not be below minus twice its standard error, and one of the two gains must be twice its
  standard error, or the image is returned as it came. For a spotlight the parts are 16 sets of columns, because
  one estimate serves the image. Here the blocks along the track are separate measurements, so the parts are
  cells of 128 rows by 32 columns (16 sets of columns at least), as many as the corrected rows hold, and the
  standard errors are measured over all of them: a narrow image is then not declined for the few sets of
  columns it holds. A single block whose own contrast does not rise, or whose cells lose, is given back as it
  came. A fit below 0.02 rad everywhere changes nothing.
- **Correction.** `autofocus.spatially_variant`'s: strips of one block, extended by the length of the blur,
  corrected column by column and cross-faded along the track with weights that sum to one. Exact zeros stay
  exact zeros.

`info` holds the blocks' estimates and their standard errors, which blocks counted, the fit, the slope across the
swath, the blocks corrected and their contrast gains. The default for a moving beam is `'quadratic'`, as for a
spotlight.

In `tests/test_moving.py`, on a simulated strip of three blocks with 36 point targets and Taylor weighting, whose
main lobe the product's own sampling resolves, a quadratic error of 2, 5
and 15 rad at the edge of the synthetic aperture, injected into the phase history, is estimated as 2.00, 5.03 and
15.35 rad. The azimuth width goes from 1.358, 2.902 and 7.764 reciprocal bands back to 1.190, 1.189 and 1.192
(1.187 focused) and the peak sidelobe from -32.8, -25.2 and -0.6 dB to -34.1, -34.1 and -33.8 dB (-34.4 focused).
An error that changes from 3.1 to 4.9 rad along the strip is followed by the fit (3.67, 4.05 and 4.38 rad at the
three blocks), and the widest response comes out at 1.199. Targets 40 m above the focus plane, seen from a track
that curves with a radius of 2.9 km, are 2.4 times as wide as in focus (2.838) and return to 1.188; a height that
grows from -6 to 46 m across the swath is fitted with 6.0 rad per half swath, and the widest response goes from
3.12 to 1.22. A focused collection comes back bit for bit. A slow-time error that grows along the strip also
moves the scatterers along the track (its linear part over each aperture); autofocus does not undo that.

The unweighted image, the default SICD, is corrected the same way: errors of 2 and 15 rad are estimated as 1.97
and 15.32 rad, and the median azimuth width goes from 0.991 and 7.713 reciprocal bands back to 0.895 and 0.897
(0.892 focused), with the peak sidelobe at -13.3 dB (-13.2). Its main lobe is narrower than a pixel of the
default spacing, so the width of a single target is measured to a tenth there and the median to a percent. On
the focused strip the estimate of the unweighted image is not zero but -0.03 to -0.04 rad in every block (a band
that ends in a step, sampled at 0.96 of its width); it passes the guard with a contrast gain of 0.015%, and the
correction changes the image by 1% of its peak. With Taylor weighting the same strip is estimated at under 0.01
rad and left alone.

### Areas in ground range

`area='tasked_quality'`, an RNIIRS value and `'black'` keep the tasked extent along the track and extend the
product across it, in ground range, as far as the criterion holds; the criteria, the two reference scenes
(`black_sigma0_db`, -10 dB by default, for `'black'`; the RGIQE's 0 dB for an RNIIRS) and the order of the noise
sources are those of [Processed area by image quality](#processed-area-by-image-quality). The model is
evaluated for the moving beam: at seven positions along the track, on the line across the track through each,
with the pulses that the pixels there are formed from. Each pulse's reference is its own SRP, so the gain of a
ground point is the antenna's two-way pattern in elevation and in azimuth, averaged in amplitude over those
pulses, with the spreading loss; the delay windows are those pulses' TOA1..TOA2 (TOAE1..TOAE2 for `'black'`).
The noise level at the beam center comes once, from the pulses of the middle position, taken from the phase
history in memory. The mask is read between the seven lines, and pixels outside the area are 0.

The antenna pattern's pointing is the one the data decide ([the rule](#the-antenna-patterns-pointing)), checked
once, on the same pulses of the middle position, and used at all seven. Where neither the file's frames nor the
boresights on the reference points agree with the footprint, the areas are refused with the `ValueError` of a
spotlight; where the scene shows no footprint the file's pointing is used and a note says that it is unverified.

An area that reaches less than one step of the model's lattice beyond the tasked swath is the tasked area
itself, on the tasked grid, with a note that says whether the receive window or the criterion ends it there. A
file without an antenna pattern is refused as for a spotlight. In
`tests/test_moving.py` a simulated stripmap carries an antenna block whose one-way gain falls by 10^5 dB per squared
direction cosine across the track, and the areas end where the two-way gain computed from that formula meets the
criterion, to one step of the model's lattice across the track (3 m there, 50 m by default): `'black'` at +-56 m
of ground range (gain -9.6 and -9.7 dB for a noise-equivalent sigma zero of -20 dB, 10 dB under the reference of
-10 dB) and at +-47 m with `black_sigma0_db=-12.0` (-7.7 and -7.8 dB for the -8 dB it needs), an RNIIRS of 4.8 at
-44 and +44 m (-7.1 and -7.2 dB for the -7.5 dB it needs at a noise-equivalent sigma zero of -10 dB, whatever
`black_sigma0_db` is), `'tasked_quality'` at the tasked +-25 m. With an extended window of +-45 m `'black'` ends
there, and with a window that ends at the tasked swath all three give the tasked area. With the frames of the
file turned 8 m off in ground range the area moves by 8 m as the file points them and stays where it was with
the boresights on the reference points.

### Real collections

Two public Capella collections were processed on an Nvidia RTX PRO 6000 that other jobs shared.

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

Autofocus (`'quadratic'`, the default) on the Taylor-weighted image of the collection as it came: six blocks of
4,096 rows (3.2 km; the footprint is 10.4 km) give -0.27,
-0.25, -0.25, -0.14, -0.67 and -0.72 rad (standard errors 0.04 to 0.37), the fit runs from -0.24 to -0.59 rad,
three of the six blocks are corrected, the contrast goes from 21.203 to 21.212 and the median azimuth width of 30
points from 1.146 to 1.142 m: the collection is in focus to half a radian, and autofocus does no harm. With a
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
beyond it. `'black'` opens the swath to the extended window TOAE1..TOAE2 and, at its default of -10 dB, finds a
signal-to-noise ratio of at least 1 from -3,550 to +2,650 m about the swath center, 6.2 km of ground range for
the 5.1 km tasked. With `black_sigma0_db=0.0`, the reference of 0.2.0's spotlight areas, it reaches from -3,550
to +3,300 m, 6.9 km (75.6 Mpx for 55.5): the near edge is the end of the extended window either way, and the far
edge follows the reference. The antenna block passes the check of its pointing as the file states it
([above](#the-antenna-patterns-pointing)).

The default products of the whole strip, on the same GPU while other jobs held about 60 GB of its memory. The SICD,
unweighted: 107,568 by 4,388 pixels of 0.767 by 1.192 m (472 Mpx) in 183 s, of which formation 165 s, the
baseband step 1.2 s, autofocus 12 s (a correction was applied) and the write 2.5 s. The SIDD, apodized: 202,356
by 8,423 pixels of 0.408 by 0.621 m (1,704 Mpx) in 535 s, of which formation 397 s, the baseband step 21 s,
autofocus 48 s, apodization 40 s and the write with its remap 28 s; the device had 35.9 GB for stages that take
36.1 GB with the image on it, so they ran on the host, and the note says so. Both files pass sarkit's
consistency checks.

**Sliding spotlight** (2024-08-29, HH, 300 MHz, 111,132 pulses in 10.7 s, grazing angle 61 degrees, tasked area 5
by 5 km; a 1 by 3 km cut at the center, formed from the whole band). The footprint moves at 1.3 km/s under a
platform at 7.2 km/s: a pixel takes 86,878 pulses (8.4 s), its azimuth band is 6.50 cycles/m, 5.8 times
2 dsin / lambda, and the default product is sampled at 0.144 m along the track (0.2.0 sampled it at 0.84 m). The
beam center's squint runs from -2.7 to +2.8 degrees over the collection, and the baseband phase follows it. The
product is 6,929 by 3,013 pixels (20.9 Mpx) and takes 27.5 GB of host memory at its peak, against 27.0 GB for
0.2.0's `form_cphd` of the same cut: the phase history is 16.8 GB as complex64 and the mosaic's range profiles
are the rest.

- Weighting. `'taylor'`: contrast 14.33, median azimuth width of 30 points 0.253 m (1.65 reciprocal bands) and
  range width 1.23 m (1.18). `'uniform'`: contrast 11.42, azimuth width 0.206 m (1.34). `'sva'`: 12,928 by 5,783
  pixels of 0.077 by 0.519 m, formed with a rectangle 7.8% wider than the processed band; contrast 15.15,
  azimuth width 0.209 m (1.35), range width 0.924 m (0.89) with a peak range sidelobe of -19.7 dB. The squint at
  the ends of the cut shears the range band by about 7% of its width, within the limit; a cut much longer along the
  track, or half the bandwidth, passes it and is refused.
- Autofocus on the Taylor-weighted image of the collection as it came: three blocks give -1.64, -1.65 and -1.54
  rad, each consistent over its
  sets, and the fit runs from -1.65 to -1.58 rad. The contrast gain of the whole image, 0.4%, is within its
  jackknife scatter (0.8%, a few bright points carry it), but the median gain of the 5,000 cells is 0.08% with a
  standard error of 0.01%: the correction is applied (one block of three is given back, its own contrast not
  rising), the contrast goes from 14.33 to 14.42 and the azimuth width from 0.253 to 0.250 m. With a quadratic
  error injected:

  | Injected | Contrast, blurred | Azimuth width, blurred | Fit over the three blocks | Contrast after | Azimuth width after |
  |---|---|---|---|---|---|
  | none | 14.33 | 0.253 m | -1.65 to -1.58 rad | 14.42 | 0.250 m |
  | 2 rad | 14.37 | 0.249 m | 0.34 to 0.45 rad | declined | |
  | 5 rad | 14.01 | 0.311 m | 3.31 to 3.38 rad | 14.39 | 0.245 m |
  | 15 rad | 13.20 | 0.618 m | 13.23 to 13.30 rad | 14.37 | 0.245 m |

  The fits are the injected error less the 1.6 rad the collection has of its own. With 2 rad injected the image
  is sharper than as it came, 0.4 rad are left, and neither gain of that correction stands out of its scatter:
  the guard returns the image as it came. With the guard's parts as 16 sets of columns, as for a spotlight, the
  5 rad case was declined on half the band (a 1,507 column image: a gain of 3.3% against a scatter of 2.5%, and
  a median over 16 sets within its error); over the cells its median gain is ten standard errors.
- Areas, with the same noise source as the stripmap. `'tasked_quality'` (a model minimum of RNIIRS 5.62): -1,650
  to +1,500 m of ground range about the swath center for the tasked +-1,500 m. An RNIIRS of 4.5: -1,650 to
  +2,150 m. `'black'`: -3,150 to +4,100 m, 7.3 km of ground range, where the extended window ends, at the default
  of -10 dB as at 0 dB. The beam is steered along the track, so the pointing of its antenna block is not
  checked against the footprint, and the note in the product says that the pattern is used unverified.

**Speed.** Seconds per stage for the stripmap cut, second call, with megapixels per second in parentheses, on an
Nvidia RTX PRO 6000 that other jobs were using and on 16 of the host's 48 cores:

| Product | Arrays | Form | Baseband | Autofocus | Apodization | Mask |
|---|---|---|---|---|---|---|
| Taylor SICD (`weighting='taylor'`), 55.5 Mpx | CuPy | 8.0 | 0.058 (954) | 0.66 (84) | | |
| | JAX on the same GPU | | 0.63 (88) | 0.49 (112) | | |
| | NumPy | | 2.7 (20) | 3.2 (17) | | |
| Apodized SICD, 201.8 Mpx | CuPy | 17.5 | 0.38 (527) | 1.21 (166) | 0.097 (2,077) | |
| | JAX | | 2.25 (90) | 0.90 (224) | 0.163 (1,235) | |
| | NumPy | | 8.2 (25) | 19.7 (10) | 9.8 (20) | |
| Taylor SICD of the `'black'` area at `black_sigma0_db=0.0`, 75.6 Mpx | CuPy | 5.6 | 0.066 (1,147) | 0.24 (322) | | 0.109 (694) |
| | JAX | | 0.98 (77) | 0.37 (203) | | 0.132 (574) |
| | NumPy | | 1.6 (47) | 1.9 (40) | | 0.24 (319) |

Formation is the mosaic's on the GPU (the 36,000 pulses of the cut, 8,194 samples each); the stages after it are
the new ones. The baseband step includes the upload of the mosaic, which the former assembles on the host, and on
JAX the host's part of the phase (the two vectors and the knots). Autofocus is the estimate on the blocks, the
cross-faded correction and the guard's sums before and after; it was applied in every row of the table (5 rad
injected). The three kinds of array give the same product to 3e-7 of its peak, zeros included. The times moved
by a factor of two between runs with the load of the machine.

The image goes to the device when its memory plan (`memory.cuda_plan`) holds it with the arrays the stages make
beside it. A CUDA device without that room leaves the stages to the host; a JAX device (a TPU) keeps the image on
the host and takes it through in blocks, stage by stage, and a note says so. With `FASTSAR_JAX_BLOCK_MB=4096`
the 201.8 Mpx product, which would take 7.7 GB of the device, crosses it in blocks of at most 1 GB and holds 1.6 GB
of it at its peak: baseband 3.9 s (51 Mpx/s), autofocus 4.9 s (41), apodization 3.6 s (56). With the image on the
device the same stages hold 8.6 GB at their peak.

## Display orientation and SIDD output

A spotlight SIDD from `process_cphd` is detected on the slant plane (after apodization, when that is the
weighting) and projected with a cubic kernel to square ground pixels on axes along the line of sight, so that
range runs straight down the rows. The default pixel is 0.7 of the finer ground resolution. The SIDD of a moving
beam is its ground image on axes along the track, apodized there when that is the weighting, detected and laid
out by the rule below ([Moving-beam collections](#moving-beam-collections)).

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
