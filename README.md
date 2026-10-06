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
```

The tests use a small simulated scene and take seconds to a few minutes.

## Coordinates

`ant` holds the antenna phase centers in a frame whose origin is the scene reference point. The output grid is
given by pixel counts `nx, ny`, spacings `spx, spy` (m) and unit vectors `e1` (azimuth) and `e2` (range) of the
image plane; pixel (i, j) sits at `(i - nx/2) spx e1 + (j - ny/2) spy e2`. `read_cphd` builds all of
these from the files, using the SICD's grid when one is given.

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
