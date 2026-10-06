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
```

The tests use a small simulated scene and take seconds to a few minutes.

## Coordinates

`ant` holds the antenna phase centers in a frame whose origin is the scene reference point. The output grid is
given by pixel counts `nx, ny`, spacings `spx, spy` (m) and unit vectors `e1` (azimuth) and `e2` (range) of the
image plane; pixel (i, j) sits at `(i - (nx - 1)/2) spx e1 + (j - (ny - 1)/2) spy e2`. `read_cphd` builds all of
these from the files, using the SICD's grid when one is given.

## Paper

The measurements behind the table, the kernels' design and their bounds against each device's units are in the
companion study, Singerman and Braun, *Cost and Precision of Spotlight SAR Image Formation on Google TPUs, an
Nvidia GPU and a CPU* (2026), code at https://github.com/saulpingerman/sar-accel-study.

## License

MIT.
