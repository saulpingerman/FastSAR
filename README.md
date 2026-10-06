# FastSAR

SAR image formation for Cloud TPUs, Nvidia GPUs and x86 CPUs from one Python call.

What it covers, with the section that documents and validates each part:

- spotlight phase histories (CPHD, frequency domain), by factorized backprojection with device kernels or polar
  format; the final tile size follows the collection's range (Tile size and range)
- exact backprojection onto any points: DEM surfaces, map grids, bistatic geometry, moving reference points
  (Exact backprojection)
- CPHD reading with per-pulse frequency grids, moving reference points, channel selection and alignment to a
  vendor SICD grid (Reading CPHD)
- wide-angle and circular apertures (Wide-angle and circular apertures)
- phase gradient autofocus (Autofocus)
- stripmap focusing by range-Doppler and omega-k (Stripmap)
- interferograms and coherence for interferometry and change detection (Interferometry and change detection)
- multilooking, terrain-corrected geocoding, GeoTIFF and SICD output (Products and geolocation)

```python
import fastsar
col = fastsar.io.read_cphd('scene_CPHD.cphd', sicd='scene_SICD.nitf')   # phase history and the vendor's grid
img = fastsar.form_image(**col)                                          # complex64 [azimuth, range]
```

For many images of the same geometry, build the plan and compile the kernels once:

```python
former = fastsar.ImageFormer(col['ant'], col['fmin'], col['df'], col['S'].shape[1],
                             col['nx'], col['ny'], col['spx'], col['spy'], col['e1'], col['e2'])
img = former(col['S'])            # each further call pays only the image formation
```

The main algorithm is factorized backprojection (three levels of tiles, Kaiser decimation filters, a 2T by 2T
product per final tile), with a kernel for each kind of device:

| backend | device | kernels |
|---|---|---|
| `tpu` | Cloud TPU (v5e, v6e) | Pallas: fused rotation and decimation, final stage on the matrix units |
| `cuda` | Nvidia GPU | CUDA through CuPy: shared-memory filters, tensor-core final stage for float16 |
| `cpu` | x86-64 with AVX-512 or AVX2 | C++ with OpenMP, compiled with g++ on first use |
| `jax` | anything JAX runs on | the plain JAX program, for checking |

All four use the same plan, filters and float64 geometry, and the CUDA and C++ float32 images agree with each
other to about -87 dB. `backend='auto'` picks the TPU if JAX sees one, else a GPU if CuPy sees one, else the CPU.
Polar format, with the resampling that removes its planar-wavefront displacement, is there as `algorithm='pfa'`.

## Speed

Time to form one Umbra spotlight image of the Panama Canal (12,207 by 8,808 pixels from 15,186 pulses of 14,399
samples), at float32-class accuracy (about -59.5 dB against a float64 exact backprojection), on-demand prices
of October 2026:

Times are the image formation itself (an `ImageFormer` call); a one-off `form_image` call adds planning and
compilation, a few seconds on an image of this size.

| device | seconds per image | dollars per 1000 images |
|---|---|---|
| TPU v6e-1 (three-pass products) | 2.4 | 1.89 |
| TPU v5e-1 (three-pass products) | 4.6 | 1.62 |
| Nvidia L4 (float32) | 3.6 | 0.76 |
| Nvidia L4 (float16 storage, float32 accumulation) | 2.9 | 0.61 |
| 16-vCPU AMD EPYC 9B45 (c4d-highmem-16) | 12.4 | 3.25 |

On the TPUs `precision='single-pass'` (the device default) runs in 1.7 s (v6e) and 2.8 s (v5e) at -45.8 dB, which
leaves the amplitude image unchanged but adds phase error beside bright returns.

## Precision options

- `float32` (default): float32 data and products. On a TPU this means three bfloat16 passes per product.
- `float16`, CUDA only: float16 phase history and intermediates with float32 accumulation; 0.9 dB above float32.
  Needs `T=32`.
- `single-pass`, `three-pass`: the TPU's one- and three-pass products.

## Tile size and range

The final stage treats each T by T tile with a plane-wave model plus an aperture-mean curvature term. What that
leaves out grows as the square of the tile size and falls with range. With `T='auto'` (the default) FastSAR
predicts the error from the collection geometry and takes the largest tile (32 or 16 pixels) that meets
`target_db` (default -40 dB), warning when neither does. The prediction tracks measurement to within about 2 dB
on simulated scenes of 0.5 m pixels:

- 1 km: T=16, -36 dB (T=32 would give -22 dB; the warning suggests coarser pixels or exact backprojection)
- 4 km: T=16, -48 dB (T=32: -35 dB)
- 16 km and beyond: T=32, -47 dB at 16 km, falling to about -62 dB at orbital range

Spaceborne collections therefore keep T=32. CUDA float16 always uses T=32.

## Install

```
pip install -e .                 # numpy, scipy, jax
pip install -e .[cuda]           # adds cupy-cuda12x (the tensor-core kernel also needs nvcc and g++)
pip install -e .[io]             # adds sarpy for CPHD and SICD
```

For a TPU install `jax[tpu]` as Google documents. The CPU backend needs g++; set `CXX` to use another compiler
and `FFBP_CPU_FLAGS` to change the flags (default `-O3 -march=native -mprefer-vector-width=512 -funroll-loops`).
For the CPU, run one thread per core: `OMP_NUM_THREADS=<cores> OMP_PLACES=cores OMP_PROC_BIND=close`.

## Tests

```
python tests/test_api.py           # every backend this machine has against the JAX program, and polar format
python tests/test_ffbp_cpu.py      # C++ kernels against the dense JAX image, two and three levels
python tests/test_ffbp_cuda.py     # CUDA kernels against the dense JAX image
python tests/test_pallas_fused.py  # TPU kernels in interpret mode (runs on a CPU)
python tests/test_autofocus.py     # phase gradient autofocus against injected phase errors (jax; or pass cpu)
python tests/test_stripmap.py      # stripmap omega-k and RDA against float64 backprojection
python tests/test_products.py      # layover projection, geocoding and multilooking on off-plane targets
python tests/test_bp.py            # exact backprojection: backends, bistatic, orbital range, moving reference
python tests/test_io.py            # CPHD helpers: frequency resampling, re-referencing, geodetic conversions
python tests/test_wide_angle.py    # 10 to 360 degree apertures against exact backprojection
python tests/test_insar.py         # change detection and interferometric height (needs finufft)
```

The tests use a small simulated scene and take seconds to a few minutes.

## Coordinates

`ant` holds the antenna phase centers in a frame whose origin is the scene reference point. The output grid is
given by pixel counts `nx, ny`, spacings `spx, spy` (m) and unit vectors `e1` (azimuth) and `e2` (range) of the
image plane; pixel (i, j) sits at `(i - nx/2) spx e1 + (j - ny/2) spy e2`. `read_cphd` builds all of
these from the files, using the SICD's grid when one is given.

## Exact backprojection

`fastsar.backproject(S, ant, fmin, df, points, rcv=None, ref=None)` forms the image at any points [..., 3]: the
plane of `plane_points(nx, ny, spx, spy, e1, e2, height=None)` (the grid of `form_image`, optionally lifted onto a
DEM), a map grid, or scattered points. With `rcv` the geometry is bistatic (`ant` is then the transmitter), and
`ref` gives per-pulse reference ranges for a phase history compensated to a moving point. Its cost is pulses times
points, so for large planar images `form_image` is the fast path. Ranges are computed in float64 at the centers of
blocks of 256 nearby points and in float32 within a block, so the float32 kernels keep their accuracy at orbital
range. Against a float64 reference (`tests/test_bp.py`, upsample 16) the CPU kernel, which works in float64,
agrees to -77 to -82 dB, and the JAX and CUDA kernels to -72 to -76 dB, monostatic and bistatic, at 5 and 600 km;
a phase history compensated to a moving point and imaged with its reference ranges matches the fixed-point image
to -60 dB. Each pulse is range compressed by a zero-padded FFT (`upsample` times its length) and read with linear
interpolation; the image error against a 128x reference falls by 12 dB per doubling of `upsample`, from -39 dB at
2 to -63 dB at 8 (the default) and -88 dB at 32.

## Reading CPHD

`fastsar.io.read_cphd(path, sicd=None, channel=0, meta=False)` reads a frequency-domain CPHD with sarpy and returns
the arguments of `form_image`; with `meta=True` it also returns the transmitter and receiver positions, the
per-pulse reference ranges and the frame (`io.local_to_ecf`, `io.ecf_to_geodetic` and their inverses convert
points). It handles:

- per-pulse frequency grids (FXFixed false), resampled onto a common grid with a 16-tap Kaiser sinc (-80 dB on a
  test signal); on the Panama collection the largest offset is 0.0045 samples and the image changes by -65 to
  -94 dB
- a moving scene reference point, re-referenced to the mid-aperture point when the range change is small and
  otherwise left for `backproject` with `ref=meta['ref']`
- channels by index or identifier (polarizations)
- pulses with invalid positions (trimmed at the ends, interpolated inside) and empty or flagged pulses (reported)
- with a vendor SICD, the vendor's grid: the origin moves onto its pixel grid so that the image is the SICD array
  (transposed when range runs along rows); on Panama our pixels coincide with the vendor's to a quarter pixel near
  the scene center
- optionally (`troposphere=True`) the per-pulse troposphere delay at the scene reference point

## Wide-angle and circular apertures

Factorized backprojection makes no small-angle assumption, so the same call images wide-angle and circular
collections, which polar format cannot. `tests/test_wide_angle.py` forms 256 by 256 images at the resolution of
apertures of 10, 45, 120 and 360 degrees (X band, 1.5 GHz of bandwidth, pixels of 72 to 6 mm) and compares them
with exact backprojection: -53.0, -61.7, -63.8 and -65.2 dB. An axis of the phase history that has become shorter
than its decimation kernel at a late level (a few frequency samples remain at these resolutions) is left
undecimated at that level.

## Interferometry and change detection

Images of two passes formed onto the same grid or the same points are coregistered by construction, and a
scatterer that lies on the grid surface contributes zero interferometric phase, so no flat-earth or topographic
phase needs removing. `products.interferogram` gives the multilooked interferogram and `products.coherence` the
sample coherence over moving windows, the statistic of coherent change detection. `tests/test_insar.py` (needs
finufft for the clutter simulation) checks three cases on simulated X-band clutter at 5 km:

- change detection, same geometry, coherence 0.98 outside two disturbed areas and a vehicle track: median sample
  coherence 0.981 on unchanged ground and 0.275 inside the changes (7 by 7 windows)
- height, passes 3.5 m apart vertically (height of ambiguity 22.3 m), a 6 m block on flat ground, both formed onto
  the ground plane: the phase converts to 5.98 m on the block (at its layover position, 3.5 m toward the radar) and
  0.00 m on the flat ground, with coherence 0.999 there
- the same pair formed by exact backprojection onto the terrain itself: median phase 0.008 rad on the block top and
  0.003 rad on the flat ground

## Products and geolocation

`fastsar.products` turns a formed image into products: `multilook` and `to_db` for detected images, `project` and
`geocode` to place 3-D points (a map grid, optionally on a DEM) in an image and sample it there, `write_geotiff`
(needs rasterio) and `write_sicd` (needs sarpy). A point off the image plane appears where the plane has its range
and Doppler cone angle at the aperture center; `project` solves that pair of conditions in closed form, so terrain
correction is exact under that model.

`tests/test_products.py` forms an image on the ground plane of targets 0 to 30 m above it: each target appears
within 0.5 cm of its projection, with layover up to 21 m. On the Umbra Panama collection, a 2048 by 2048 crop
formed by factorized backprojection and geocoded onto the pixel grid of the vendor's GEC GeoTIFF, and exact
backprojection directly onto the same ground points, both land 6.98 m from the GEC when the surface is taken at the
height of the scene reference point (-0.34 m above the ellipsoid); the offset changes by 1.33 m per meter of
assumed height and vanishes at 5.3 m above that height, where exact backprojection onto the ground points also
focuses best (correlation with the GEC 0.75, against 0.13 at -0.34 m). The vendor's SICD projected through its
own model lands on the GEC with no offset.

`write_sicd` writes an image formed on a vendor SICD's grid (`read_cphd(..., sicd=...)`) with that SICD's
metadata, ImageFormAlgo OTHER and Grid.Type PLANE: the pixels are backprojection's on the template's image plane.
A round trip of the Panama SICD keeps every pixel and passes sarpy's validity check; its projection of the center
pixel agrees with the vendor's exactly and departs from it by up to about 6 m at the corners, where the vendor's
polar-format image is displaced by the distortion its own projection model accounts for.

## Stripmap

`fastsar.stripmap` focuses stripmap data from a straight, constant-velocity track, starting from raw echoes or from
range-compressed ones:

```python
import numpy as np
from fastsar import stripmap as sm, quality
p = sm.make_params(squint_deg=5.0)                          # airborne X-band geometry; see make_params
raw = sm.simulate(p, [[0.0, p.r0], [20.0, p.r0 + 80.0]])    # point targets at (x, zero-Doppler range)
img, r, x = sm.focus_stripmap(raw, p, algorithm='omegak', rwin='taylor')    # or 'rda', 'bp'
ij = np.unravel_index(np.abs(img).argmax(), img.shape)
m = quality.point_target(img, ij, d_az=x[1] - x[0], d_rg=r[1] - r[0])     # resolution, PSLR, ISLR, peak
```

The simulator computes the echoes of each target exactly in float64 in the time domain (up-chirp, two-way sinc^2 or
Gaussian azimuth pattern, optional squint). Omega-k uses the exact two-dimensional reference phase and the exact
Stolt mapping, interpolated with a windowed sinc; the range-Doppler algorithm corrects the exact range migration
r/cos(theta) by sinc interpolation and has optional secondary range compression. Both run in JAX on any device, in
float32 by default or float64, with phases and interpolation positions computed in float64 on the host. Time-domain
backprojection in numpy float64 is the reference. All three return the image on the zero-Doppler grid with the same
complex scale and phase, so they can be compared sample by sample.

`tests/test_stripmap.py` checks them on a 9.6 GHz scene (100 MHz chirp of 2 us sampled at 125 MHz, PRF 650 Hz,
200 m/s, 5 km range, 1.5 m antenna, 1024 pulses of 448 samples) with five targets across a 200 m swath, a Taylor
range window and the sinc^2 pattern as the only azimuth weighting. The error against backprojection on 64 by 64
pixel patches around the targets is:

- broadside: omega-k -64.9 to -65.5 dB, RDA with secondary range compression -66.5 to -67.7 dB, without it -65.4 to
  -66.3 dB
- 5 degrees of squint (Doppler centroid 1116 Hz, 1.7 times the PRF): omega-k -60.2 to -62.5 dB, RDA with secondary
  range compression -61.1 to -67.4 dB, without it -27.4 to -27.7 dB

At broadside every algorithm measures 0.585 m in azimuth and 1.79 m in range against 0.585 m and 1.775 m expected
from the weighting, with peak sidelobe ratios of -39.6 dB (azimuth) and -32.7 dB (range; the chirp's spectral ripple
raises the Taylor sidelobes by about 2 dB) and peak positions within 1.5 mm of the truth. The squinted image is
sheared, so its range cut measures 1.74 m. The float32 and float64 omega-k images differ by -125 dB, and the residual
against backprojection comes from the edge of the processed Doppler band, which backprojection cuts on pulses and
the other two on Doppler bins. On four CPU threads omega-k takes 0.3 s and RDA 0.1 s for this scene.

## Autofocus

`fastsar.autofocus` estimates an unknown phase error per pulse by phase gradient autofocus (Wahl, Eichel,
Ghiglia and Jakowatz, 1994) and removes it from the phase history:

```python
img, phi = fastsar.autofocus.autofocus(S, ant, fmin, df, nx, ny, spx, spy, e1, e2, backend='cpu')
S_corrected = S * np.exp(-1j * phi)[:, None]
```

The image is formed by factorized backprojection, and PGA runs along its azimuth axis (axis 0): the brightest
pixel of each range line is centered, the lines with the highest peak-to-mean intensity are kept and windowed, and
the phase difference between adjacent azimuth frequency bins is estimated jointly over the kept lines. The window
narrows as the image focuses. The estimate on bins is mapped to pulses through each pulse's azimuth spatial
frequency at the center frequency, the phase history is corrected and the image re-formed, twice by default.

A backprojection image keeps the spherical wavefront, so the azimuth spectrum of a scatterer moves with its
position (by 16 of 128 bins for a scatterer 20 m from the center at 5 km range in the test below). Before PGA the
image is therefore multiplied by `exp(-1j * deramp_phase(...))`, the difference between the spherical and planar
wavefronts of the center pulse, which returns every scatterer to the polar-format convention in which one pulse
occupies one bin. `pga` can be called on its own on an image in that convention.

`tests/test_autofocus.py` injects phase errors into a simulated 128 by 128 image of 0.5 m pixels (0.6 m
resolution, 5 km range, 190 pulses) holding 40 point targets over 3,000 clutter scatterers. Image errors are
relative to the error-free image after the best complex gain, and residuals are rms over pulses after removing
the constant and linear terms:

- quadratic, 5.3 rad peak: image error -2.7 dB before, -30.1 dB after; residual 0.036 rad
- polynomial to fifth order, 4.1 rad peak: -6.8 dB before, -29.2 dB after; residual 0.037 rad
- low-pass random walk, 3.0 rad peak: -0.2 dB before, -30.3 dB after; residual 0.036 rad

On the error-free image the procedure changes the image by -30.1 dB (estimated phase 0.038 rad rms), and on the
same targets without clutter by -34.3 dB. This floor is the estimator's error on the scene, set by clutter and
by scatterers that share a range line, rather than a residual of the injected errors; two other random scenes
gave residuals of 0.035 and 0.05 rad. The method assumes an aperture that fits inside the image's azimuth band,
that is, azimuth pixels finer than the resolution, and an image long enough in azimuth for its bin spacing,
1/(nx spx), to follow the variation of the phase error across the aperture. It has been validated with
backprojection images only.

## Paper

The measurements behind the table, the kernels' design and their bounds against each device's units are in the
companion study, Singerman and Braun, *Cost and Precision of Spotlight SAR Image Formation on Google TPUs, an
Nvidia GPU and a CPU* (2026), code at https://github.com/saulpingerman/sar-accel-study.

## License

MIT.
