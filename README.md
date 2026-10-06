# FastSAR

Spotlight SAR image formation for Cloud TPUs, Nvidia GPUs and x86 CPUs from one Python call.

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
```

The tests use a small simulated scene and take seconds to a few minutes.

## Coordinates

`ant` holds the antenna phase centers in a frame whose origin is the scene reference point. The output grid is
given by pixel counts `nx, ny`, spacings `spx, spy` (m) and unit vectors `e1` (azimuth) and `e2` (range) of the
image plane; pixel (i, j) sits at `(i - nx/2) spx e1 + (j - ny/2) spy e2`. `read_cphd` builds all of
these from the files, using the SICD's grid when one is given.

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
