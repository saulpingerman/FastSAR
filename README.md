# FastSAR

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

FastSAR forms synthetic aperture radar (SAR) images from phase history data on x86 CPUs, Nvidia GPUs and Google
Cloud TPUs, with the same Python call on each. It reads frequency-domain CPHD files, forms spotlight, stripmap and
sliding spotlight images by factorized backprojection, and turns them into products (autofocus, multilooking,
interferometry, geocoding, SICD and GeoTIFF output).

![FastSAR image of the Pacific entrance of the Panama Canal from an Umbra spotlight collection, showing the Cocolí Locks, the channel with a ship, and the Port of Balboa](docs/images/hero_panama.jpg)

*FastSAR's image of the Pacific entrance of the Panama Canal (Umbra open data, 2023-07-18): 12,207 by 8,808
pixels formed in 3.6 s on an Nvidia L4 by float32 factorized backprojection, downsampled for display. Its error
against a float64 exact backprojection is -59.5 dB.*

On the same collection the float32 image takes 12.4 s on a 16-vCPU AMD EPYC 9B45 instance. ISCE3, the fastest
open-source backprojection that forms an image of this collection, would take an estimated 4.8 h on that instance
and 27 min on the L4.

![Cost per 1000 Panama Canal images against image error for FastSAR on TPU v5e, TPU v6e, Nvidia L4 and CPU, and for the open-source implementations RITSAR, bpBasic, ISCE3, torchbp and GRDL](docs/images/teaser.png)

*Cost per 1000 Panama images (on-demand prices, October 2026) against error relative to the float64 image.
Details: [docs/performance.md](docs/performance.md) and [docs/comparison.md](docs/comparison.md).*

## Install

FastSAR is installed from source. The base install pulls in numpy, scipy and jax.

```bash
pip install "fastsar @ git+https://github.com/saulpingerman/FastSAR"           # CPU
pip install "fastsar[cuda] @ git+https://github.com/saulpingerman/FastSAR"     # + CuPy for Nvidia GPUs
pip install "fastsar[io] @ git+https://github.com/saulpingerman/FastSAR"       # + sarpy for CPHD and SICD
```

From a clone: `pip install -e ".[cuda,io]"`, or `uv sync` with the lock file. The CPU kernels need `g++` with
OpenMP and compile on first use. The CUDA backend needs `cupy-cuda12x` (the `cuda` extra), plus `nvcc` for the
float16 tensor-core kernel. TPUs need `jax[tpu]`, installed as the JAX documentation describes. On the CPU, run one
thread per core: `OMP_NUM_THREADS=<cores> OMP_PLACES=cores OMP_PROC_BIND=close`.

## Quickstart

```python
import fastsar

out = fastsar.form_cphd('scene_CPHD.cphd')          # spotlight or moving beam, grid and window from the file
img = out['image']                                  # complex64 [nx, ny] on a ground plane at the scene height
# pixel (i, j) lies at out['origin'] + i*out['spx']*out['e1'] + j*out['spy']*out['e2'] in read_cphd's local frame
lat, lon, h = fastsar.products.geolocate(out, i, j, height=0.0)         # pixel -> ground (products.locate back)
fastsar.products.write_geotiff('amp.tif', **fastsar.products.geocode_image(out))   # amplitude on a UTM map grid
fastsar.products.write_sicd('scene.nitf', out)                          # complex image with its geometry

col = fastsar.io.read_cphd('scene_CPHD.cphd', sicd='scene_SICD.nitf')    # spotlight phase history, vendor's grid
img = fastsar.form_image(**col, backend='cuda')                          # 'auto' tries tpu, cuda, then cpu
former = fastsar.ImageFormer(col['ant'], col['fmin'], col['df'], col['S'].shape[1], col['nx'], col['ny'],
                             col['spx'], col['spy'], col['e1'], col['e2'])   # plan and compile once
img = former(col['S'])                                                   # then one call per image
```

The [processing chain](docs/processing-chain.md) takes these images on to autofocus, multilooking, geocoding and
output. [`examples/form_umbra.py`](examples/form_umbra.py) forms an Umbra image from the command line, and
[`examples/form_capella_stripmap.py`](examples/form_capella_stripmap.py) a Capella stripmap crop.

## Supported data, modes and devices

| Data | Modes | Status |
|---|---|---|
| Capella CPHD (frequency domain) | stripmap, spotlight, sliding spotlight | formed by `form_cphd` and compared with the vendor's SICD ([real-data.md](docs/real-data.md)); one 2022 dynamic stripmap collection is correct near the scene center only |
| Umbra CPHD (frequency domain) | spotlight | three open-data collections formed on the vendor's grid |
| ICEYE CPHD (frequency domain) | dwell spotlight | read (mode EXPERIMENTAL accepted, SICD metadata from the `.xml`); the CPU former reads its 28 GB history in place and the CUDA former streams it; no published comparison with the vendor image |
| CPHD in the time-of-arrival (TOA) domain | | not supported; `read_cphd` raises |
| Raw Level-0 data (Sentinel-1, NISAR, ALOS) | | not supported yet; no decoder |
| Simulated echoes | stripmap, ScanSAR, TOPS, any track | `fastsar.stripmap`, `fastsar.burst`, `fastsar.patches` |

| | CPU (`cpu`) | Nvidia GPU (`cuda`) | Cloud TPU (`tpu`) | Any JAX device (`jax`) |
|---|---|---|---|---|
| Factorized backprojection | C++/OpenMP, AVX-512 or AVX2 | CUDA via CuPy | Pallas | reference program |
| Precision | float32 | float32, float16 | three-pass (default), single-pass | float32, float16 |
| Exact backprojection | float64 | float32, float64 block centers | none (falls back to `jax`) | float32, float64 block centers |

Polar format (`algorithm='pfa'`) runs as a JAX program on whatever device JAX uses.

## Documentation

[docs/README.md](docs/README.md) lists every page. The [API reference](docs/api.md) gives each public function's
signature and purpose; the docstrings in `fastsar/` give every argument.

The measurement scripts and records behind the results are in the
[sar-accel-study](https://github.com/saulpingerman/sar-accel-study) repository. A citation file will come with the
preprint.

## Contributing and license

Bug reports and pull requests are welcome; [CONTRIBUTING.md](CONTRIBUTING.md) explains how to run the tests.
MIT license, see [LICENSE](LICENSE).
