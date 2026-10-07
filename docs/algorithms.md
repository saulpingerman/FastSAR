# Algorithms and API

This page covers spotlight image formation: the coordinate convention, factorized backprojection and its tile
size, exact backprojection, polar format, wide-angle apertures and autofocus. Numbers marked as test results come
from the scripts in `tests/`, which use small simulated scenes; results on real collections are in
[real-data.md](real-data.md) and [precision.md](precision.md).

## Coordinates and arguments

```python
img = fastsar.form_image(S, ant, fmin, df, nx, ny, spx, spy, e1, e2,
                         algorithm='ffbp', backend='auto', precision='float32', T='auto', target_db=-40.0)
```

- `S`: phase history [pulses, samples], complex, frequency domain, motion compensated to the scene reference point.
- `ant`: antenna phase centers [pulses, 3] in a frame whose origin is the scene reference point.
- `fmin`, `df`: first frequency and sample spacing (Hz).
- `nx, ny, spx, spy, e1, e2`: the output grid, pixel counts and spacings (m) along the unit vectors `e1`
  (azimuth) and `e2` (range) of the image plane. Pixel (i, j) sits at `(i - nx/2) spx e1 + (j - ny/2) spy e2`.

The result is a complex64 image [nx, ny]. `fastsar.io.read_cphd` builds all of these from a CPHD file, using
the SICD's grid when one is given ([real-data.md](real-data.md)). A Taylor window (-35 dB, nbar 4) is applied
along both axes unless `window=False`.

`form_image` plans and compiles on every call. For repeated images of one geometry, build an `ImageFormer` once
and call it on each phase history; a different antenna path needs a new former.

## Factorized backprojection

The image is divided recursively into tiles (three levels by default). At each level the phase history of a
parent tile is re-referenced to each child tile's center by a phase ramp, then low-pass filtered and decimated in
frequency and pulse index by a Kaiser-windowed sinc (70 dB design attenuation). At the last level the range to
each pixel is linearized about the tile center, so each final T by T tile is a product of two matrices followed
by a per-pixel quadratic phase correction. Rectangular images of any size are supported.

| backend | device | kernels |
|---|---|---|
| `tpu` | Cloud TPU (v5e, v6e) | Pallas: fused rotation and decimation, final stage on the matrix units |
| `cuda` | Nvidia GPU | CUDA through CuPy: shared-memory filters, tensor-core final stage for float16 |
| `cpu` | x86-64 with AVX-512 or AVX2 | C++ with OpenMP, compiled with g++ on first use |
| `jax` | anything JAX runs on | the plain JAX program, for checking |

All four use the same plan, filters and float64 geometry. The CUDA and C++ kernels use float64 geometry at every
level; the JAX program and the TPU kernels compute the last level's geometry in float32. The CUDA and C++
float32 images agree with each other to about -87 dB. `backend='auto'` picks the TPU if JAX sees one, else a GPU
if CuPy sees one, else the CPU. `fastsar.available_backends()` lists what the machine can run.

The CPU kernels are compiled with `g++` (override with `CXX`) and the flags in `FFBP_CPU_FLAGS`, default
`-O3 -march=native -mprefer-vector-width=512 -funroll-loops`. The compiled library is cached under
`~/.cache/fastsar`.

### Tile size and range

The final stage treats each T by T tile with a plane-wave model plus an aperture-mean curvature term. What that
leaves out grows as the square of the tile size and falls with range. With `T='auto'` (the default) FastSAR
predicts the error from the collection geometry and takes the largest tile (32 or 16 pixels) that meets
`target_db` (default -40 dB), and warns when neither does. The prediction is kept as
`ImageFormer.predicted_error_db`. Spaceborne collections keep T=32. CUDA float16 always uses T=32.

Test results on simulated scenes of 0.5 m pixels, where the prediction tracks measurement to within about 2 dB:

- 1 km: T=16, -36 dB (T=32 would give -22 dB; the warning suggests coarser pixels or exact backprojection)
- 4 km: T=16, -48 dB (T=32: -35 dB)
- 16 km and beyond: T=32, -47 dB at 16 km, falling to about -62 dB at orbital range

## Exact backprojection

```python
img = fastsar.backproject(S, ant, fmin, df, points, rcv=None, ref=None, backend='auto', upsample=8)
pts = fastsar.plane_points(nx, ny, spx, spy, e1, e2, height=None)   # the form_image grid, optionally on a DEM
```

`backproject` forms the image at any points [..., 3]: the plane of `plane_points`, a map grid, or scattered
points. With `rcv` the geometry is bistatic (`ant` is then the transmitter), and `ref` gives per-pulse reference
ranges for a phase history compensated to a moving point. Its cost is pulses times points, so for large planar
images `form_image` is the fast path.

Each pulse is range compressed by a zero-padded FFT (`upsample` times its length) and read with linear
interpolation. Ranges are computed in float64 at the centers of blocks of 256 nearby points and in float32
within a block, so the float32 kernels keep their accuracy at orbital range. The CPU kernel works in float64.
There is no TPU kernel; `backend='tpu'` runs the JAX program.

Test results (`tests/test_bp.py`, simulated, `upsample=16`, against a float64 reference):

- the CPU kernel agrees to -77 to -82 dB, the JAX and CUDA kernels to -72 to -76 dB, monostatic and bistatic,
  at 5 and 600 km
- a phase history compensated to a moving point and imaged with its reference ranges matches the fixed-point
  image to -60 dB
- the image error against a 128x reference falls by 12 dB per doubling of `upsample`, from -39 dB at 2 to
  -63 dB at 8 (the default) and -88 dB at 32

## Polar format

`form_image(..., algorithm='pfa')` resamples the data onto a rectangular wavenumber grid by chirp-z transforms and
forms the image by a two-dimensional FFT. It runs as a JAX program on whatever device JAX uses. `pfa_guard` sets
the margin (m) kept free of wrap-around around the scene; the default 300 m suits orbital scenes of a few
kilometers and must be smaller for small simulated scenes.

Polar format assumes a planar wavefront, so a scatterer away from the scene center is focused at a displaced
position. On the Umbra Panama collection the displacement reaches 18 pixels in azimuth and 12 in range at the
corners. FastSAR fits that displacement from the collection geometry and removes it by a final resampling. After
the resampling the error against the float64 reference is -32.1 dB on Panama (-33.4 and -35.3 dB on the
Melbourne and Iowa collections), and it grows with distance from the scene center:

![Error of corrected polar format and float32 factorized backprojection in four rings by distance from the scene center, for the Panama, Melbourne and Iowa collections](images/rings.png)

*Error relative to the float64 reference in four concentric rings, for polar format and the float32 factorized
image on the L4. The polar-format error rises by about 10 dB from the inner to the outer ring; the factorized
error does not depend on distance from the center.*

## Wide-angle and circular apertures

Factorized backprojection makes no small-angle assumption, so the same call images wide-angle and circular
collections, which polar format cannot. An axis of the phase history that has become shorter than its decimation
kernel at a late level is left undecimated at that level.

Test results (`tests/test_wide_angle.py`, simulated): 256 by 256 images at the resolution of apertures of 10, 45,
120 and 360 degrees (X band, 1.5 GHz of bandwidth, pixels of 72 to 6 mm) agree with exact backprojection to
-53.0, -61.7, -63.8 and -65.2 dB.

## Autofocus

`fastsar.autofocus` estimates an unknown phase error per pulse by phase gradient autofocus (PGA) and removes it from
the phase history:

```python
img, phi = fastsar.autofocus.autofocus(S, ant, fmin, df, nx, ny, spx, spy, e1, e2, backend='cpu')
S_corrected = S * np.exp(-1j * phi)[:, None]
```

The image is formed by factorized backprojection, and PGA runs along its azimuth axis (axis 0): the brightest
pixel of each range line is centered, the lines with the highest peak-to-mean intensity are kept and windowed,
and the phase difference between adjacent azimuth frequency bins is estimated jointly over the kept lines. The
window narrows as the image focuses. The estimate is mapped from bins to pulses through each pulse's azimuth
spatial frequency, the phase history is corrected and the image re-formed, twice by default.

A backprojection image keeps the spherical wavefront, so the azimuth spectrum of a scatterer moves with its
position. Before PGA the image is multiplied by `exp(-1j * deramp_phase(...))`, which returns every scatterer to
the polar-format convention in which one pulse occupies one bin. `pga` can be called on its own on an image in
that convention.

Test results (`tests/test_autofocus.py`, simulated 128 by 128 image of 0.5 m pixels, 40 point targets over 3,000
clutter scatterers at 5 km). Image errors are relative to the error-free image; residuals are rms over pulses
after removing the constant and linear terms:

- quadratic, 5.3 rad peak: image error -2.7 dB before, -30.1 dB after; residual 0.036 rad
- polynomial to fifth order, 4.1 rad peak: -6.8 dB before, -29.2 dB after; residual 0.037 rad
- low-pass random walk, 3.0 rad peak: -0.2 dB before, -30.3 dB after; residual 0.036 rad

On the error-free image the procedure changes the image by -30.1 dB, a floor set by clutter and by scatterers that
share a range line. The method assumes azimuth pixels finer than the resolution and an image long enough in
azimuth for its bin spacing, 1/(nx spx), to follow the phase error across the aperture. It has been tested on
simulated backprojection images only.
