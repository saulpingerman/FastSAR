# Several collections of one scene

`process_stack` takes two or more collections of one scene from their CPHD files to one product: the incoherent
average of all of them, which has less speckle than any one image at the same resolution, or the coherence and
the change between two. `fastsar.stack` holds the steps: a common ground grid, registration, and the stack
products.

```python
import fastsar
from fastsar import stack

files = ['a_CPHD.cphd', 'b_CPHD.cphd', 'c_CPHD.cphd']
out = fastsar.process_stack(files, 'mean_SIDD.nitf')                       # the average, as a display product
coh = fastsar.process_stack(files[:2], 'coherence.tif', product='coherence')
chg = fastsar.process_stack(files[:2], 'change.tif', product='change')

stack.baseline(files[0], files[1])                                         # can this pair be coherent at all?

grid = stack.ground_grid((46.9043, -96.8288, 247.7), extent=2000.0, spacing=0.65, azimuth=247.4)
outs = [fastsar.form_cphd(f, grid=grid) for f in files]                    # the same pixels for every collection
outs = stack.register(outs)                                                # one translation each, against the first
avg = stack.mean(outs, equalize='mean')
coh = stack.coherence(outs[0], outs[1], window=7)
```

The figures on this page marked as measured come from six X-band spotlights of one agricultural and built-up
site, taken between May 2024 and January 2025 by three small satellites at grazing angles of 60 to 61 degrees:
three pairs of nearly equal look direction, 12 to 30 degrees apart from one another. They were formed on a grid of
2 km by 2 km at 0.656 m (9.3 megapixels).

## The common grid

A grid is a plane fixed to the Earth: a dict with `origin` (ECEF, pixel (0, 0)), two orthonormal axes `e1` and
`e2` (ECEF, `e1` x `e2` upward), the spacings `spx` and `spy` and the `shape`. Pixel (i, j) lies at
`origin + i spx e1 + j spy e2`.

- `stack.ground_grid(center, extent, spacing, azimuth=0.0)` builds one that is horizontal at its center.
  `center` is (latitude, longitude, height above the ellipsoid), `extent` and `spacing` are in meters along `e1`
  and `e2`, and `azimuth` is the direction of `e2` in degrees clockwise from north; `e1` lies 90 degrees clockwise
  of it. A radar that looks along `e2` has its shadows down the rows of the display.
- `stack.as_grid(grid)` accepts that dict, the same quantities under other names (a `center` in ECEF or `lat`,
  `lon` and `height`; `azimuth` in place of the axes; `extent` in place of the shape), or a product of
  `form_cphd`, `process_cphd` or this module, which stands for its own pixels.
- `form_cphd(cphd, grid=grid)` and `process_cphd(cphd, output, grid=grid)` form a collection directly on the
  grid. Each collection has its own local frame, so the grid's origin and axes are converted into it
  (`io.ecf_to_local`); the image is formed there by the same backprojection as on the collection's own grid. The
  product records the grid (`out['grid']`, and `out['axes'] == 'grid'`), and `grid=out` gives a later collection
  the same pixels.

With `grid=`, `spacing`, `extent` and `height` are given by the grid and refused as arguments. `process_cphd`
also refuses the quality areas (the grid is the processed area), the slant plane and apodization, which runs on
the slant plane. Taylor and uniform weighting apply as usual. The image autofocus works along the first axis of
the image, so it applies where the grid's second axis lies within 5 degrees of the collection's line of sight,
as it does for a grid taken from an earlier product of a similar pass; on another grid it is left out by default
and refused when asked for.

The plane is flat and the Earth is not. At a distance d from the center the plane stands d^2 / (2 R) above the
surface of the center's height: 0.7 m at 3 km, 8 m at 10 km, 71 m at 30 km. Ground at the center's height then
appears displaced toward the radar by that height over the tangent of the grazing angle, and by a different
vector for each look direction. Over tens of kilometers a stack is better cut into grids of a few kilometers,
each horizontal at its own center.

## Registration

Collections formed on one grid are registered exactly as far as their geometry is known. For simulated tracks
that is exact. For real repeat collections it is not: measured on the common grid, the five other images of the
site lay 9 to 22 m from the reference, which is the accuracy of the orbit and timing knowledge in the files, far
more than a pixel. `stack.register(products, reference=0)` removes it:

- The estimate is one translation per image, from the peak of the cross-correlation of the detected images
  (intensity averaged over 5 by 5 pixels, in decibels, clipped at the 1st and 99.5th percentiles, structure larger
  than 32 pixels left out), refined on the correlation upsampled 16 times. It runs where the images are.
- `registration['significance']` is the peak over the rms of the correlation surface (measured: 69 to 129), and
  `registration['found']` is False when it is under 5 or the peak lies on the border of the search; such an
  image is left where it is, with a note.
- `registration['residual_median']` and `['residual_rms']` (m, with `residual_worst` and the tiles' own shifts)
  are what 4 by 4 tiles of the image still disagree by after the translation. They tell whether one translation
  was enough. Measured: a median of 0.15 m (0.26 m rms) for the pair with equal look directions; medians of 0.6
  to 1.1 m and rms of 0.7 to 2.3 m between look directions 12 to 30 degrees apart, where single tiles are off by
  5 to 6 m and set the rms.

What remains between different look directions is height, not a warp of the ground: a scatterer above the grid
plane lays over toward each radar, by 0.5 to 0.9 m per meter of height between these directions, and no smooth
map of the plane removes that. A first-order warp fitted to the tiles' shifts lowered their rms by 5 to 26%.

The translation is applied in one of two ways, which agree to -49 dB away from the edges
(`tests/test_stack.py`):

- `apply='fourier'` (the default, `stack.shift_image`): the whole pixels by moving the array and the fraction of
  a pixel by a phase ramp on the spectrum, with no interpolation kernel. The image is taken to baseband first
  (`planes.deramp`, exact at every pixel), where its spectrum is centered at zero wherever the pixel and whatever
  the squint, and an image on its carrier is put back on the carrier of the displaced grid. Where the spectrum
  wraps around the sampling rate, each sample of the transform takes the phase of its alias inside the
  collection's support. Pixels shifted in from outside the image are 0, and pixels that were exactly 0 stay 0.
- `apply='reform'`: the collection is formed again on the grid displaced by the translation
  (`form_cphd(grid=, shift=)`). This is exact up to the image's edges and costs a second read and formation.

Either way the registered product is recorded on the grid itself, and its frame (`meta['origin']`) is moved by
the translation, as if the collection's orbit had been corrected by it. Its pixels keep their ECEF positions,
`products.geolocate`, the SICD and the SIDD stay consistent, and every image of the stack has the reference's
geolocation. `out['shift']` records the translation in meters along `e1` and `e2`.

## The average

`stack.mean(products, equalize=None, weights=None)` is the mean of the intensities at full resolution. It
returns a product whose image is the amplitude of that mean, with the reference's geometry.

The equivalent number of looks (mean squared over variance of the intensity) is measured on homogeneous ground:
tiles of 32 by 32 pixels, without those whose brightest pixel exceeds 6 times their mean, of which the 15% that
vary least after smoothing over 9 by 9 pixels are kept. Measured on the six spotlights, for the average of the
first n images:

| n | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| looks | 0.99 | 1.88 | 2.69 | 3.64 | 4.60 | 5.41 |

That is 0.91 to 0.95 of n times one image's. The speckle of different dates and directions is uncorrelated
(intensity correlation under 0.01, and 0.03 for the pair with equal look directions), so nothing is lost there.
The shortfall is brightness: looks of intensities with means m_k add to (sum m_k)^2 / sum m_k^2 of one image's,
which is n only for equal means, and the same tile differed by 4.2 dB between its brightest and faintest date
(the median over the tiles). The product reports this prediction as `looks_independent`; it gave 5.45 against
the 5.41 measured.

For the same speckle, one image had to be averaged over 5 by 5 pixels (3.3 m), which widened point responses
from 1.3 m to 3.7 m, while the six-image average kept 1.3 m. The contrast between adjacent fields that differ by
0.3 to 1.5 dB, over its standard deviation, doubled.

`equalize='mean'` or `'median'` scales each image's intensity by one factor that brings its brightness (the mean
of the pixels under their 99th percentile, or the median) to the reference's. The average is then in the
reference image's units, and a difference in brightness between dates is removed whether it came from the
radar's gain or from the ground (moisture, crops, snow). Without it the average is the plain mean of the images
in their own units; CPHD carries no radiometric calibration. A pixel that is 0 in one image is left out of that
pixel's average.

## Coherence and change

Two images share spectrum only where their ground wavenumbers coincide. `stack.baseline(a, b)` tells from the
per-pulse parameters of two files, without reading their samples, whether they do: the difference in grazing
angle against the critical one, (B / fc) / tan(graze), at which the range bands part; the difference in azimuth
against the angle the aperture itself spans on the ground; and the fraction of the band left in each. Of the
fifteen pairs of the six spotlights, three were inside both (0.3, 0.7 and 1.2 degrees apart in azimuth against
apertures of 1.5 to 1.9 degrees).

`stack.coherence(a, b, window=7)` estimates |<a b*>| / sqrt(<|a|^2> <|b|^2>) over the window, after two steps
that keep the estimate from reading low:

- Each image is cut at baseband to the part of its spectrum the other shares, the sector that the other's pulse
  directions and band span on the ground. `process_stack` goes further for a pair: it forms both collections
  from the pulses that look from one interval of azimuth, so the common band keeps the full resolution.
- A linear fringe is removed in tiles of 384 pixels, at the peak of each tile's interferogram spectrum. A
  residual baseline or registration error leaves one, and so does ground tilted against the grid plane.

The estimator is biased upward, and by more than the window's pixel count suggests. An incoherent pair gives
about 0.89 / sqrt(n) for n independent samples, but pixels closer than a resolution cell are not independent.
`floor` is therefore measured on the pair itself: the same estimate with the second image displaced by 29
pixels, where nothing is coherent. Measured at 0.7 of a resolution cell per pixel with 7 by 7 windows: a median
of 0.25, against the 0.13 of 49 independent samples (the window holds about 11.5).

Measured on the pair two weeks apart with 0.07 degrees between its grazing angles and 0.29 degrees between its
azimuths:

| | median | above 0.5 |
|---|---|---|
| floor (the second image displaced) | 0.25 | 4% |
| the images as each collection is formed alone | 0.28 | 11% |
| both formed on one interval of azimuth, common band | 0.32 | 17% |

The coherence above the floor lies on roads and buildings. Fields are at the floor after two weeks in July. The
other two pairs inside the critical baseline, 68 and 151 days apart, were at the floor altogether.

`stack.change(a, b)` returns that coherence and the ratio of the two intensities averaged over the window, in
decibels with its median removed. Low coherence means change only where the surface stays coherent and only
against the floor: vegetation, water, snow and tilled or wet soil decorrelate without changing, and so does a
pair outside the critical baseline or badly registered. For data like these, coherent change detection is
meaningful on built surfaces only. The intensity ratio needs no coherence; its speckle is that of the window's
independent samples, about 6.1 / sqrt(n) dB (`ratio_sigma`), less where the pair is coherent.

## One call

`process_stack(cphds, output=None, *, product='mean', grid=None, reference=0, ...)`:

- Without a `grid`, the reference collection's tasked area is used on axes along its line of sight, so that its
  shadows fall down the rows. The pixels are square, 0.7 of the coarsest collection's ground range resolution
  c / (2 B cos(graze)), and finer where some collection's spectrum on those axes needs it. Every collection's
  geometry is read from its per-pulse parameters for this, before any samples.
- The collections are read and formed one at a time, the reference first, so one phase history is in memory at a
  time. Each image goes to the backend's device for registration and the product, or stays on the host when the
  device lacks the room, with a note.
- `register='fourier'` (default), `'reform'` or `None`; `equalize='mean'` by default here, since a display
  product gains most from equal brightness; `weighting='taylor'` or `'uniform'`.
- The average is written as a SIDD (`remap='nrl'`, shadows down). Coherence is written as a GeoTIFF of one band,
  change as one of two bands (coherence, intensity ratio), in the UTM zone of the grid's center. The grid is
  written without resampling, with the affine transform that places its pixels on the map, when that is within a
  tenth of a pixel everywhere (grids of a few kilometers); a larger grid is resampled north up.

The SIDD of an average says what it is. `ProductCreation` has `ProductClass` 'Detected Image', `ProductType`
'Multi-image incoherent average' and extension parameters for the number of collections, the reference, the
combination, the translations applied, whose geolocation the product has, and the equivalent number of looks.
`ExploitationFeatures` holds one `Collection` per input with its own sensor, date and time, geometry and
phenomenology on the product's plane. `Measurement` carries the reference collection's aperture and time, so the
SIDD's own projection is the reference's. sarkit's consistency check passes it (`tests/test_stack.py`).

Time for the six collections (3.0 to 4.7 GB each) to the SIDD of the average, on an Nvidia RTX PRO 6000 and a
48-core host that other jobs kept fully loaded: 36 s, or 1.5 megapixels of formed image per second end to end.
Reading took 7.5 s; windowing, the range gate and formation 17.8 s (3.1 megapixels per second); registration
9.9 s (the detected images, 17 correlations and the translation of each image: 4.7 megapixels per second); the
average with its looks 0.25 s (over 200 megapixels per second); writing 0.2 s. The coherence of a pair to its
GeoTIFF took 20 to 29 s, of which 1.5 to 3.4 s for the common band, the coherence and its floor.

## The range gate

A grid of a few kilometers is small against the swath a wide-band collection's samples span, and the first
level of the factorized former then decimates in range by a large factor (81 to 99 for these collections). Its
filter did not fit the shared memory of the GPU's kernel, and 0.2.0 refused the image. `ImageFormer` now gates
such a history in range first: each pulse is padded by an eighth of the band on either side, transformed to
range, cut to the bins the image spans over the aperture with a margin, and transformed back over the band and
its guards, which are tapered to 0. `range_gate='auto'` does so where the first level would not fit, `True`
wherever the gate keeps less than half of the samples, and `False` never; `process_stack` uses `True`. The image
equals that of the whole history: -68 dB on a simulated collection with clutter outside the gate (cpu and cuda),
and -63 dB for a 2 km image of a 76,032-sample collection on the cuda backend against the cpu backend ungated.

## Tests

`tests/test_stack.py` simulates passes over one scene from different tracks and writes each as a CPHD displaced
as a whole by meters:

- on the common grid each target lies within 0.04 pixel of where the grid says, displaced by the error written
  into its file;
- `register` recovers the errors to 0.02 pixel for a pass on the reference's track and to 0.07 pixel from
  another direction (other speckle, on images of 241 pixels), and the phase ramp agrees with formation on the
  displaced grid to -49 dB;
- a moving-beam collection formed as a mosaic on a grid turned 45 degrees from its track has its targets within
  0.1 pixel of where the grid says;
- four images with independent speckle average to 3.80 looks, against 3.78 for four times one image's;
- the same track with noise added gives a coherence of 0.948 against the 0.943 its signal-to-noise ratio implies,
  and a pass from another direction over other clutter gives the floor (0.26 against 0.25);
- `process_stack` gives the same products with the images on the host, on a CUDA device and on JAX's (the
  average's intensity to -87 and -93 dB), and sarkit's consistency check passes the SIDD of an average.
