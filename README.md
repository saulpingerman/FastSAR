# FastSAR

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/pyproject.toml)

FastSAR forms synthetic aperture radar (SAR) images on x86 CPUs, Nvidia GPUs and Google Cloud TPUs from Python. It
reads frequency-domain CPHD files, forms spotlight, stripmap and sliding spotlight images by factorized
backprojection, and provides autofocus, interferometry, geolocation, map GeoTIFFs and SICD output.

![FastSAR image of the Panama Canal's Pacific entrance from an Umbra spotlight collection](https://raw.githubusercontent.com/saulpingerman/FastSAR/v0.1.1/docs/images/hero_panama.jpg)

*Panama Canal, Pacific entrance (Umbra open data, 2023-07-18): 12,207 by 8,808 pixels in 3.8 s on an Nvidia L4
(warm former, transfers included; float32, -59.9 dB against the float64 reference of
[precision.md](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/docs/precision.md)), downsampled. A
c4d-highmem-16 CPU instance (16 vCPUs, 8 cores) takes 11.3 s. ISCE3, the fastest open-source code that forms this
collection, would take an estimated 4.8 h on that CPU and 27 min on the L4.*

![Cost per 1000 images against error for FastSAR and five open-source implementations](https://raw.githubusercontent.com/saulpingerman/FastSAR/v0.1.1/docs/images/teaser.png)

*Cost per 1000 Panama images (October 2026 prices) against error over the lock, port and ship regions, relative to
the float64 reference ([performance](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/docs/performance.md), [comparison](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/docs/comparison.md)).*

## Install

Linux on x86-64, Python 3.10 to 3.13. The CPU kernels compile on first use with `g++` and OpenMP (Debian and
Ubuntu: `apt install g++`); the `cuda` extra needs an Nvidia driver for CUDA 12 and the `tpu` extra a Cloud TPU VM.

```bash
pip install fastsar                  # CPU (and JAX on whatever device it has)
pip install "fastsar[io]"            # io: sarpy for CPHD and SICD
pip install "fastsar[cuda]"          # cuda: CuPy for Nvidia GPUs
pip install "fastsar[tpu]"           # tpu: jax[tpu] for Cloud TPUs
pip install "fastsar[geo]"           # geo: rasterio for GeoTIFFs and DEMs
pip install "fastsar[cuda,io,geo]"   # extras combine
```

From a clone: `pip install -e ".[cuda,io]"` or `uv sync`. The float16 CUDA kernel needs `nvcc`. On the CPU set
`OMP_NUM_THREADS=<physical cores> OMP_PLACES=cores OMP_PROC_BIND=close`. Importing `fastsar` sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` unless it is already set, so that JAX does not reserve most of a GPU's memory
before the CUDA kernels run ([performance.md](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/docs/performance.md#environment-variables)).

## Quickstart

The chain below reads a CPHD file and writes a GeoTIFF and a SICD, so it needs `pip install "fastsar[io,geo]"`.

```python
import fastsar
from fastsar import products

out = fastsar.form_cphd('scene_CPHD.cphd')        # mode, grid and window from the file
img = out['image']                                # complex64 [nx, ny] on a ground plane at the scene height
lat, lon, h = products.geolocate(out, 100, 200)   # pixel (100, 200) on the ground
products.write_geotiff('amp.tif', **products.geocode_image(out))   # amplitude on a north-up UTM grid
products.write_sicd('scene.nitf', out)            # complex image with its geometry
```

[`examples/chain.py`](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/examples/chain.py) runs this chain from the command line (`--simulate` needs no data);
[`form_umbra.py`](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/examples/form_umbra.py) and [`form_capella_stripmap.py`](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/examples/form_capella_stripmap.py) form
vendor grids.

## Supported data, modes and devices

| Data | Modes | Status |
|---|---|---|
| Capella CPHD | stripmap, spotlight, sliding spotlight | checked against vendor SICDs; a 2022 dynamic stripmap is correct near the center only |
| Umbra CPHD | spotlight | three open-data collections on the vendor's grid |
| ICEYE CPHD | dwell spotlight | formed (28 GB history); not compared with the vendor image |
| TOA-domain CPHD, raw Level-0 | | not supported |
| Simulated echoes | stripmap, ScanSAR, TOPS, any track | `fastsar.stripmap`, `burst`, `patches` |

| | CPU (`cpu`) | Nvidia GPU (`cuda`) | Cloud TPU (`tpu`) | Any JAX device (`jax`) |
|---|---|---|---|---|
| Factorized backprojection | C++/OpenMP, AVX-512 or AVX2 | CUDA via CuPy | Pallas | reference program |
| Precision | float32 | float32, float16 | three-pass (default), single-pass | float32, float16 |
| Exact backprojection (`ExactFormer`) | C++/OpenMP, float32 with float64 tile centers | CUDA, the same | falls back to `jax` | float32, float64 block centers |

Polar format (`algorithm='pfa'`) runs as a JAX program on any device.

## Documentation

[docs/README.md](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/docs/README.md) lists every page; start with the [processing chain](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/docs/processing-chain.md).
Measurement records are in [sar-accel-study](https://github.com/saulpingerman/sar-accel-study). A citation file
will come with the preprint.

## Contributing and license

[CONTRIBUTING.md](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/CONTRIBUTING.md) explains how to run the tests
(GitHub Actions runs the CPU suite; the CUDA and TPU tests run on those machines before a release). MIT license ([LICENSE](https://github.com/saulpingerman/FastSAR/blob/v0.1.1/LICENSE)).
