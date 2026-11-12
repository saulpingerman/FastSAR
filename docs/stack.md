# Several collections of one scene

`process_stack` takes two or more collections of one scene from their CPHD files to one product: the incoherent
average of all of them, which has less speckle than any one image at the same resolution, or a product of two
as a coherent pair: their coherence, their interferogram, their coherent sum, or what changed between them.
`fastsar.stack` holds the steps: a common ground grid, registration, the common support of a pair, and the
stack products.

```python
import fastsar
from fastsar import stack

files = ['a_CPHD.cphd', 'b_CPHD.cphd', 'c_CPHD.cphd']
out = fastsar.process_stack(files, 'mean_SIDD.nitf')                       # the average, as a display product
coh = fastsar.process_stack(files[:2], 'coherence.tif', product='coherence')
chg = fastsar.process_stack(files[:2], 'change.tif', product='change')
ifg = fastsar.process_stack(files[:2], 'phase.tif', product='interferogram')
both = fastsar.process_stack(files[:2], 'sum_SIDD.nitf', product='coherent_sum')

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
  directions and band span on the ground. A pair whose spectra already coincide to 2% is taken as it is.
  `process_stack` does better for a pair: it forms both collections from their common support, cut in the
  phase history ([coherent pairs](#coherent-pairs)).
- A linear fringe is removed in tiles of 384 pixels. A residual baseline or registration error leaves one, and
  so does ground tilted against the grid plane. A tile takes its own fringe, the peak of its interferogram's
  spectrum, only where that peak stands clear of what a spectrum without a fringe gives; elsewhere it takes
  the image's. A tile of water or vegetation has no fringe, and the highest sample of its noise is not fitted.

The estimator is biased upward, and by more than the window's pixel count suggests. An incoherent pair gives
about 0.89 / sqrt(n) for n independent samples, but pixels closer than a resolution cell are not independent.
`floor` is therefore measured on the pair itself: the same estimate with the second image displaced by 29
pixels, where nothing is coherent. Measured at 0.7 of a resolution cell per pixel with 7 by 7 windows: a median
of 0.25, against the 0.13 of 49 independent samples (the window holds about 11.5).

Measured on the pair two weeks apart with 0.07 degrees between its grazing angles and 0.29 degrees between its
azimuths:

| | median | above 0.5 |
|---|---|---|
| floor (the second image displaced) | 0.26 | 5% |
| the images as each collection is formed alone | 0.28 | 11% |
| both formed on their common support | 0.31 | 17% |

The coherence above the floor lies on roads and buildings. Fields are at the floor after two weeks in July. The
other two pairs inside the critical baseline, 68 and 151 days apart, were at the floor altogether.

`stack.change(a, b)` returns that coherence and the ratio of the two intensities averaged over the window, in
decibels with its median removed. Low coherence means change only where the surface stays coherent and only
against the floor: vegetation, water, snow and tilled or wet soil decorrelate without changing, and so does a
pair outside the critical baseline or badly registered. Noise does too, so `change` also returns `allowed`, the
coherence that the two brightnesses allow at each pixel, 1 / sqrt((1 + 1 / snr_a)(1 + 1 / snr_b)) with each
image's noise bounded by its darkest ground, and counts the change where it can be seen: of the pixels allowed
more than 0.9 (`capable_fraction`), those under 0.4 (`changed_fraction`). For data like these, coherent change
detection is meaningful on built surfaces only. The intensity ratio needs no coherence; its speckle is that of
the window's independent samples, about 6.1 / sqrt(n) dB (`ratio_sigma`), less where the pair is coherent.

## Coherent pairs

A pair is processed coherently when its two images are to be compared in phase: an interferogram, a coherent
sum, coherent change. Both collections are then cut to the part of the ground spectrum they share before the
images are formed, registered to a few hundredths of a pixel, and compared after their common fringe is removed.

**Before forming anything.** `stack.baseline(a, b)` reads the per-pulse parameters of two files and returns what
decides the matter: `overlap_range` and `overlap_azimuth` (the fraction of the band both have at equal look
direction, and the fraction that the apertures each image would use alone share), `azimuth_available` (the
interval of look directions both whole apertures cover, over what a square resolution cell needs: 1 or more
means the pulses can be cut to full overlap), `critical_graze` and `critical_azimuth` (the differences at which
nothing is left), `height_of_ambiguity`, and `days` between the two. A pair worth forming has both overlaps
well above zero and few days. The height of ambiguity then says what its phase will show: under a few meters,
every building and tree is its own fringe and tall structures decorrelate within a resolution cell; at hundreds
of meters, relief is nearly invisible and the phase is left to motion and propagation. It is only as good as
the two orbits: a perpendicular baseline of 8 m with orbits known to a meter is uncertain by an eighth.

**The common support.** A pulse that looks at the scene from the direction phi samples, at frequency f, the
ground wavenumber of radius 2 f cos(graze) / c in that direction. `stack.common_support(a, b)` returns the
interval of directions both apertures cover (its central part where it is wider than a square cell needs) and,
at each direction, the radii both bands reach, as the selection of each collection that covers it: `pulses`,
the `aperture=(first, last + 1)` of `form_cphd`, and `bands`, its `band=`, the first and last frequency of
every pulse. A difference in grazing angle moves the two bands against each other, so one collection keeps the
lower part of its band and the other the upper part; along a squinted aperture the grazing angle changes, so
the limits follow the pulse. `form_cphd(band=)` gives the samples outside no weight and applies the Taylor
window over the band of each pulse, and along the aperture as a function of the look direction, so that two
collections cut to one support carry one weighting of it.

That last point is why the cut belongs in the phase history. Cutting the two images to their common band
afterwards (`coherence(common_band=True)`) removes what one has alone, but each image was weighted over its own
aperture, and the same part of the spectrum then carries the left side of one window and the right side of the
other. Measured on an exactly repeating pair by displacing the two apertures against each other by 30% of
their width, on the surfaces that are coherent when nothing is displaced (0.79 with Taylor weighting, 0.76
without):

| | Taylor | uniform |
|---|---|---|
| each aperture alone | 0.56 | 0.53 |
| the two images cut to their common band | 0.63 | 0.77 |
| both formed on the common support | 0.81 | 0.77 |

The simulated pairs of the tests say the same with a known answer, stable clutter whose noise allows 0.943: an
aperture displaced by 30% gives 0.52 alone, 0.68 after the image cut and 0.950 on the common support; grazing
angles 0.6 degrees apart (a quarter of the critical difference) give 0.74, 0.79 and 0.951. Unweighted, the
image cut reaches 0.94 in both, as the common support does.

**Registration.** `register(..., coherent=True)` refines the translation found on the detected images by
maximizing the complex correlation of tiles within half a pixel of it, and reports what that added (`gain`).
On the pairs measured it moved the translation by 0.01 to 0.1 pixel and raised the tiles' correlation by 0.2 to
0.8%; tiles each given their own shift gain another 0.2 to 0.5%, so one translation is kept.

**Products.** All take a registered pair formed on its common support, and `process_stack` forms it that way.

- `stack.interferogram(a, b, window=7, dem=None)`: the phase of a b* averaged over the window, and the
  coherence. Both images lie on one plane, so ground in the plane has no phase and there is no flat-earth
  fringe; one linear fringe, which a difference of orbit errors leaves, is removed where the image has one, and
  the topographic phase of a surface `dem(lat, lon)` if one is given. A point h above the plane turns the phase
  by 2 pi h / `height_of_ambiguity`; motion d toward the radar by 4 pi d / wavelength (`height_per_radian`,
  `displacement_per_radian`). `flatness` summarizes what is left at the scale of 128 pixels.
- `stack.coherent_sum(a, b)`: the second image brought to the first's phase (the fringe, and the phase of the
  surroundings of each pixel) and amplitude, and added. Signal adds in amplitude and noise in power, so the
  signal-to-noise ratio gains 1 + coherence, 3 dB at most, at full resolution. `gain` reports it as measured:
  the sum against the first image on the most coherent tenth of the scene and on its darkest hundredth.
- `stack.change(a, b)`: as above.

`process_stack(..., product='interferogram')` writes a GeoTIFF of two bands (phase, coherence);
`product='coherent_sum'` writes a SIDD like the average's, with `ProductType` 'Multi-image coherent sum', both
collections under `ExploitationFeatures` and the measured gain as an extension parameter.

**Measured on three pairs** of X-band spotlights from open archives (Sunny Isles Beach in Florida, the port of
Melbourne, the Black Rock Desert in Nevada), each 2 km by 2 km at 0.39 to 0.41 m, 7 by 7 pixel windows, Taylor
weighting:

| | urban coast | port and suburb | desert playa |
|---|---|---|---|
| satellite, days apart | Capella, 3.0 | Umbra, 6.0 | Capella, 3.0 |
| band common at equal look direction | 1.00 | 0.89 | 1.00 |
| common azimuth over a square cell's | 3.3 | 0.96 | 3.3 |
| difference in grazing angle (critical) | 0.0008 (1.82) degrees | 0.197 (1.86) degrees | 0.0023 (1.81) degrees |
| height of ambiguity | 510 m | 2.5 m | 174 m |
| coherence: median, mean of the top tenth | 0.30, 0.80 | 0.29, 0.65 | 0.28, 0.58 |
| floor (median) | 0.25 | 0.27 | 0.25 |
| top tenth with each formed alone and the images cut | (an exact repeat) | 0.61 | (an exact repeat) |
| best tenth, measured against allowed by noise | 0.79, 0.95 | 0.62, 0.81 | 0.56, 0.57 |
| can show change (allowed above 0.9); of those, changed | 17%; 31% | 12%; 55% | 1%; 36% |
| coherent sum: gain against the first, the second image | 2.6, 2.7 dB | 2.2, 1.2 dB | 2.0, 1.4 dB |

The first and third pairs repeat their orbits to a few meters, so their images formed alone already coincide
and trimming changes nothing. The second pair is the ordinary case: 4% of a square cell's aperture and 11% of
the band are not common, and forming on the common support raises the top tenth from 0.61 to 0.65. Its
coherent sum gains less against the second image because that image is the less noisy of the two.

**What limits coherence**, in the order these pairs show it:

1. Time. Three days after the first collection, water and vegetation are at the estimator's floor and built
   surfaces are not: 4.8% of the urban scene is above 0.8. Nothing in the processing recovers a surface
   that changed.
2. Noise. On the desert scene the best tenth of the pixels reaches what its signal-to-noise ratio allows and
   no more (0.56 against 0.57). `change`'s `allowed` shows where a low coherence is only a dark surface.
3. Geometry. A displaced aperture or a different grazing angle costs coherence in proportion to the support
   lost, and the common support returns it (the tables above). With a height of ambiguity of 2.5 m, what stands
   above the plane decorrelates within its own resolution cell, and no cut of the spectrum returns that.
4. Registration, last. A tenth of a pixel costs under 1%.

Two things on the urban pair are not explained by any of these. Its interferogram is not flat after the linear
fringe: the phase of blocks of 128 pixels spreads over 1.25 rad rms (-2.1 to +2.0 rad between the 5th and 95th
percentiles), with a full cycle across 150 to 250 m of the lower buildings inland. It is not relief: with a
height of ambiguity of 510 m the towers of the scene, laid over across 60 to 100 m of height, would each show a
ramp of -0.7 to -1.2 rad, and the seven measured show -1.5 to +0.6 rad, four of them of the other sign. It is
not a difference in how the two collections were focused or sampled: the cross-spectrum of their coherent tiles
is flat over the common support to 0.13 rad rms, with 0.01 rad of curvature at the band edges and 0.04 pixel of
slope, the delay removed for the troposphere is the same constant in both, and both lie on one plane. The phase
changes by a quarter of a cycle at most along a tower and differs between towers 100 to 300 m apart, so it
belongs to the place on the ground, as a difference in propagation delay between the two dates would (a cycle is
16 mm of path); these data do not say more.

The other is the coherence of the best surfaces, 0.79 where their brightness allows 0.95. The 0.79 is the median
of the top tenth of the measured coherence, which begins at 0.64 by construction. Where the brightness allows
0.98 or more the median is 0.90 and a tenth of the pixels exceed 0.98; cells that one strong scatterer dominates
have 0.91 over 7 by 7 pixels and 0.95 over 5 by 5; and the stable set falls from 0.79 to 0.75 as the window
grows from 7 to 15 pixels. Per-surface registration (0.13 pixel median, worth 0.5%), the two spectra (their
weightings agree to 0.5 dB, and a flat cross-spectrum would add 0.9%) and the estimator (biased upward) do not
account for the rest, which leaves scatterers that decorrelate in three days mixed with stable ones at the scale
of the window.

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
- The two collections of a pair product are formed from their common support (`aperture='auto'`; `'square'`
  forms each as it would be alone), and their translation is refined on the complex images. The product has
  `support` (`common_support`'s figures) and `baseline`.
- The average and the coherent sum are written as a SIDD (`remap='nrl'`, shadows down). Coherence is written as
  a GeoTIFF of one band, change as one of two bands (coherence, intensity ratio) and the interferogram as one of
  two (phase, coherence), in the UTM zone of the grid's center. The grid is written without resampling, with
  the affine transform that places its pixels on the map, when that is within a tenth of a pixel everywhere
  (grids of a few kilometers); a larger grid is resampled north up.

The SIDD of an average says what it is. `ProductCreation` has `ProductClass` 'Detected Image', `ProductType`
'Multi-image incoherent average' and extension parameters for the number of collections, the reference, the
combination, the translations applied, whose geolocation the product has, and the equivalent number of looks.
`ExploitationFeatures` holds one `Collection` per input with its own sensor, date and time, geometry and
phenomenology on the product's plane. `Measurement` carries the reference collection's aperture and time, so the
SIDD's own projection is the reference's. sarkit's consistency check passes it (`tests/test_stack.py`).

Time for the six collections (3.0 to 4.7 GB each) to the SIDD of the average, on an Nvidia RTX PRO 6000 shared
with other jobs and 16 cores of the host: 14.6 s, or 3.8 megapixels of formed image per second end to end.
Reading took 3.3 s; windowing, the range gate and formation 7.9 s (7.1 megapixels per second); registration
3.2 s (the detected images, 17 correlations and the translation of each image: 15 megapixels per second); the
average with its looks 0.11 s; writing 0.08 s. With the host ten times oversubscribed the same call took 36 to
56 s. A coherent pair of 27-megapixel images (two collections of 8.5 GB) took 22 s to its interferogram with
the host and the GPU both saturated by other work: reading 4.7 s, windowing and formation on the common support
5.3 s, registration with its refinement on the complex images 9.3 s, the interferogram 2.4 s.

The stages by themselves on 9.3-megapixel arrays, in megapixels per second (the best of three runs, measured
with the host oversubscribed, so the host column is a lower bound):

| stage | host (48 cores) | GPU |
|---|---|---|
| detected image for the correlation | 10 | 89 |
| translation estimate, the whole image | 5.0 | 40 |
| the same with its 16 tiles | 1.6 | 9.3 |
| translation by a phase ramp | 20 | 440 |
| average of two images with its looks | 12 | 720 |
| coherence over 7 by 7 pixels, fringe removed in tiles | 3.3 | 16 |

## The range gate

A grid of a few kilometers is small against the swath a wide-band collection's samples span, and the first
level of the factorized former then decimates in range by a large factor (81 to 99 for these collections). Its
filter did not fit the shared memory of the GPU's kernel, and 0.2.0 refused the image. `ImageFormer` now gates
such a history in range first: each pulse is padded by an eighth of the band on either side, transformed to
range, cut to the bins the image spans over the aperture with a margin, and transformed back over the band and
its guards, which are tapered to 0. `range_gate='auto'` does so where the first level would not fit, `True`
wherever the gate keeps less than half of the samples, and `False` never; `process_stack` uses `True`. A host
history crosses to the device block by block through one page-locked buffer by synchronous copies, so the buffer
is free when the next block is written into it, and every block is checked across the transfer (a weighted sum
of a subsample of its rows, on the host and on the device): a block that did not arrive as it was raises an
error instead of entering an image. The image equals that of the whole history: -68 dB on a simulated collection with clutter outside the gate (cpu and cuda),
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
  and a pass from another direction over other clutter gives its floor (0.07 against 0.08);
- two apertures displaced by 30% of their length, and two grazing angles 0.6 degrees apart: stable clutter
  formed on the common support has the coherence its noise allows (0.950 and 0.951 against 0.943), formed each
  alone 0.52 and 0.74, and with the images cut to their common band 0.68 and 0.79 under Taylor weighting (0.94
  and 0.94 unweighted);
- ground 0.3 m higher over half of the scene, at a height of ambiguity of -1.217 m: the interferogram's phase
  steps by -1.534 rad against 2 pi h / height of ambiguity = -1.549, and by 0.011 rad with the surface given;
- two images of one scene with independent noise sum to 2.9 dB more signal-to-noise ratio (6.0 dB of signal,
  3.1 dB of noise), an incoherent pair to 0.4 dB;
- a displacement of (0.12, -0.08) pixel is found on the complex images to 0.003 pixel, where the detected
  images leave 0.03;
- a tile takes its own fringe on a tilted coherent pair and none on an incoherent one;
- `process_stack` gives the same products with the images on the host, on a CUDA device and on JAX's (the
  average's intensity to -87 and -93 dB), and sarkit's consistency check passes the SIDD of an average.
