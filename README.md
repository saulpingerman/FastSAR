# FastSAR

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

FastSAR forms synthetic aperture radar (SAR) images from phase history data on x86 CPUs, Nvidia GPUs and Google
Cloud TPUs, with the same Python call on each.

![FastSAR image of the Pacific entrance of the Panama Canal from an Umbra spotlight collection, showing the Cocolí Locks, the channel with a ship, and the Port of Balboa](docs/images/hero_panama.jpg)

*FastSAR's image of the Pacific entrance of the Panama Canal (Umbra open data, 2023-07-18): 12,207 by 8,808
pixels formed in 3.6 s on an Nvidia L4 by float32 factorized backprojection, downsampled for display. Its error
against a float64 exact backprojection is -59.5 dB.*

## What it does

- Spotlight image formation by factorized backprojection, with hand-written kernels for each device: C++/OpenMP
  for the CPU, CUDA for Nvidia GPUs, Pallas for TPUs. Polar format with its wavefront-curvature correction is
  also available.
- Exact backprojection onto any set of points: DEM surfaces, map grids, bistatic geometry, moving reference points.
- CPHD reading, with output on the vendor's SICD pixel grid when a SICD is given, and SICD and GeoTIFF writing.
- Stripmap (range-Doppler, omega-k, and patch mosaics for arbitrary tracks), ScanSAR and TOPS bursts, phase
  gradient autofocus, interferograms, coherence and Pauli decomposition.

## Results

On the Panama collection above, compared against a float64 exact backprojection, FastSAR's float32 factorized
backprojection forms the full image in 12.4 s on a 16-vCPU AMD EPYC 9B45 instance and 3.6 s on an Nvidia L4.
Its error over three image regions is -55.8 to -61.2 dB on the CPU and -55.7 to -61.2 dB on the L4. The fastest open-source backprojection that forms an image of
this collection, ISCE3, would take an estimated 4.8 h on the same CPU instance and 27 min on the same GPU.

![Cost per 1000 Panama Canal images against image error for FastSAR on TPU v5e, TPU v6e, Nvidia L4 and CPU, and for the open-source implementations RITSAR, bpBasic, ISCE3, torchbp and GRDL](docs/images/teaser.png)

*Cost per 1000 Panama images (on-demand prices, October 2026) against error relative to the float64 image.
Details and the open-source comparison: [docs/comparison.md](docs/comparison.md) and
[docs/performance.md](docs/performance.md).*

## Install

FastSAR is installed from source. The base install pulls in numpy, scipy and jax.

```bash
pip install "fastsar @ git+https://github.com/saulpingerman/FastSAR"           # CPU
pip install "fastsar[cuda] @ git+https://github.com/saulpingerman/FastSAR"     # + CuPy for Nvidia GPUs
pip install "fastsar[io] @ git+https://github.com/saulpingerman/FastSAR"       # + sarpy for CPHD and SICD
```

Or from a clone: `pip install -e ".[cuda,io]"`, or `uv sync` with the lock file.

| Backend | Needs |
|---|---|
| CPU | `g++` with OpenMP; the kernels compile on first use |
| CUDA | `cupy-cuda12x` (the `cuda` extra); `nvcc` and `g++` for the float16 tensor-core kernel |
| TPU | `jax[tpu]`, installed as the JAX documentation describes |

For the CPU, run one thread per core: `OMP_NUM_THREADS=<cores> OMP_PLACES=cores OMP_PROC_BIND=close`.

## Quickstart

```python
import fastsar

out = fastsar.form_cphd('scene_CPHD.cphd')          # any mode: spotlight, stripmap, sliding spotlight
img = out['image']                                  # ground-plane grid: out['origin'], out['e1'], out['e2'], spacing
```

`form_cphd` picks the mode, window and grid from the file (or from the vendor's SICD with `sicd=`). For control over
each step:

```python
col = fastsar.io.read_cphd('scene_CPHD.cphd', sicd='scene_SICD.nitf')   # phase history and the vendor's grid
img = fastsar.form_image(**col)                     # complex64 [nx, ny]; backend='auto' picks TPU, GPU, then CPU
img_pfa = fastsar.form_image(**col, algorithm='pfa')                     # polar format
```

To form many images of one geometry, plan and compile once with `ImageFormer`:

```python
former = fastsar.ImageFormer(col['ant'], col['fmin'], col['df'], col['S'].shape[1],
                             col['nx'], col['ny'], col['spx'], col['spy'], col['e1'], col['e2'],
                             backend='cuda', precision='float32')
img = former(col['S'])
```

Exact backprojection onto the same grid, optionally lifted onto a DEM:

```python
pts = fastsar.plane_points(col['nx'], col['ny'], col['spx'], col['spy'], col['e1'], col['e2'])
img = fastsar.backproject(col['S'], col['ant'], col['fmin'], col['df'], pts, backend='cpu')
```

[`examples/form_umbra.py`](examples/form_umbra.py) forms an Umbra image from the command line, and
[`examples/form_capella_stripmap.py`](examples/form_capella_stripmap.py) a Capella stripmap crop.

## Devices and algorithms

| | CPU (`cpu`) | Nvidia GPU (`cuda`) | Cloud TPU (`tpu`) | Any JAX device (`jax`) |
|---|---|---|---|---|
| Factorized backprojection | C++/OpenMP, AVX-512 or AVX2 | CUDA via CuPy | Pallas | reference program |
| Precision | float32 | float32, float16 | three-pass (default), single-pass | float32, float16 |
| Exact backprojection | float64 | float32, float64 block centers | none (falls back to `jax`) | float32, float64 block centers |

The factorized backends share one plan, one set of filters and float64 geometry. Polar format
(`algorithm='pfa'`) runs as a JAX program on whatever device JAX uses.

## Documentation

- [Algorithms and API](docs/algorithms.md): coordinates, factorized and exact backprojection, polar format, tile size, wide-angle apertures, autofocus
- [Stripmap, long apertures and burst modes](docs/stripmap-burst.md)
- [Real data](docs/real-data.md): CPHD reading, SICD grids, Umbra and Capella collections, vendor quirks
- [Products](docs/products.md): interferometry, change detection, geocoding, SICD and GeoTIFF output
- [Precision](docs/precision.md): precision options and image quality on real data
- [Performance](docs/performance.md): time and cost per image by device
- [Comparison](docs/comparison.md) with RITSAR, the AFRL toolbox, ISCE3, torchbp and GRDL

## Citing

Citation: see CITATION.cff (coming with the preprint).

The measurement scripts and records behind the results are in the
[sar-accel-study](https://github.com/saulpingerman/sar-accel-study) repository.

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to run the tests.

## License

MIT, see [LICENSE](LICENSE).
