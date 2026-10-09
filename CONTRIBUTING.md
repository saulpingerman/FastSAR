# Contributing to FastSAR

Bug reports, questions and pull requests are welcome. Please open an issue first for a change that touches a
kernel or the public API, so the approach can be agreed before the code is written.

## Development setup

```bash
git clone https://github.com/saulpingerman/FastSAR
cd FastSAR
uv sync --group test            # the versions pinned in uv.lock, with the test packages; --group dev adds nox
```

or, with pip, `pip install -e ".[io,test]"` (add `cuda` on a machine with an Nvidia GPU: `".[cuda,io,test]"`).
The CPU backend needs `g++` with OpenMP. The `test` dependency group (and the `test` extra) holds pytest and the
packages some tests need that are not dependencies of the library: `sarpy` (`tests/test_chain.py`), `finufft`
(`tests/test_insar.py`), `rasterio` (`products.write_geotiff`, `read_dem` and map projections other than
latitude/longitude) and `sarkit` (consistency checks in `tests/test_chain.py`).

## Running the tests

The tests are pytest modules on small simulated scenes; the CPU suite takes about four minutes on eight cores.
From the repository root:

```bash
uv run pytest                          # every test this machine can run
uv run pytest tests/test_exact.py -s   # one module, with the measurements each test prints
uv run pytest -k "bp and cpu"          # tests by name
```

Tests that need hardware are marked `cuda` (CuPy and an Nvidia GPU) or `tpu` (a Cloud TPU) and are skipped on a
machine without it; `--require cuda`, `--require tpu` or `--require cuda,tpu` makes a missing device a failure, for
runs on those machines. Tests that compare backends run once per backend (`cpu`, `jax`, and `cuda` or `tpu` where
present). Tests that need a package of the `test` group are skipped without it; `--require io` makes that a
failure.

Each test's peak resident memory is checked against `--max-rss-gb` (default 12, or `FASTSAR_TEST_MAX_GB`): a test
above it fails, and a process that reaches 1.5 times the limit is stopped so that a runaway test cannot exhaust the
machine. The summary lists the peak of each module. `--timeout` (pytest-timeout) limits the time of each test.
For the CPU kernels, set `OMP_NUM_THREADS` to the number of physical cores.

[nox](https://nox.thea.codes) runs the suite in fresh environments made by uv (`uv tool install nox`, or
`uv sync --group dev`):

```bash
nox                  # tests on Python 3.10 to 3.13, and lowest: numpy, scipy and jax at their lower bounds
nox -s tests-3.12    # one Python
nox -s cuda          # on a machine with an Nvidia GPU: installs the cuda extra, runs with --require cuda
nox -s tpu           # on a Cloud TPU VM: installs the tpu extra, runs with --require tpu
nox -s tests-3.12 -- -k exact -s     # arguments after -- go to pytest
```

The nox sessions pass `--require io` and a timeout of 600 s per test. GitHub Actions runs `tests` on each Python
and `lowest` on every push and pull request (CPU only). `python tests/run_all.py` remains for scripts written for
the former runner: it runs pytest with the same options (`-t` seconds per test, `--max-rss-gb`, `--require`,
module names to select, `-v` for every test's output) and, as before, fails when a test package is missing.

| Module | Checks |
|---|---|
| `test_api.py` | every backend this machine has against the JAX program, and polar format |
| `test_ffbp_cpu.py` | C++ kernels against the dense JAX image, two and three levels |
| `test_ffbp_cuda.py` | CUDA kernels against the dense JAX image (needs a GPU) |
| `test_pallas_fused.py` | TPU level kernel in interpret mode (runs on a CPU) |
| `test_pallas_final.py` | TPU final-stage kernel in interpret mode (runs on a CPU) |
| `test_pallas_e2e_tpu.py` | TPU kernels end to end in interpret mode (runs on a CPU) |
| `test_autofocus.py` | phase gradient autofocus on the JAX and CPU backends, and CUDA and TPU where present |
| `test_stripmap.py` | stripmap omega-k and RDA against float64 backprojection |
| `test_patches.py` | patch mosaics for stripmap and non-linear tracks |
| `test_burst.py` | ScanSAR and TOPS burst focusing |
| `test_products.py` | layover projection, geocoding, multilooking, Pauli |
| `test_bp.py` | exact backprojection: backends, bistatic, orbital range, moving reference |
| `test_io.py` | CPHD helpers: frequency resampling, re-referencing, geodetic conversions |
| `test_wide_angle.py` | 10 to 360 degree apertures against exact backprojection |
| `test_insar.py` | change detection and interferometric height (needs finufft) |
| `test_chain.py` | simulated CPHD to geolocation, map GeoTIFFs, SICD and autofocus (needs sarpy, rasterio, sarkit) |
| `test_exact.py` | ExactFormer: 1 to 600 km and a wide near-range grid against backprojection, validation |
| `test_units.py` | filters, program cache, aperture weights, mosaic pieces, input checks |
| `test_cphd.py` | read_cphd and form_cphd on simulated collections (stand-in CPHD reader) |
| `test_planning.py` | bucket planning, memory models, TPU out-of-memory retry, program cache |
| `test_accuracy.py` | tile-size and oversampling errors against exact backprojection |

`test_ffbp_cpu.py`, `test_ffbp_cuda.py` and the three `test_pallas_*.py` modules compare each error with a limit
set about 5 dB above the value measured when the limit was set. A test that checks several cases reports every
case that failed.

## Pull requests

- Keep a pull request to one change, and state in the description which tests ran and on which devices.
- A change to a kernel should report the error against the JAX program or exact backprojection before and after,
  as the tests print it.
- A change that adds a result to the documentation should say what data it comes from. Results from simulated
  data are labeled as test results.
- Do not commit data files (CPHD, SICD, NITF, `.npy`) or build artifacts.

## Reporting a bug

Use the bug report template. Include the FastSAR commit, the backend, the device, the Python, JAX and CuPy
versions, and if possible a script that reproduces the problem on a simulated scene (`fastsar.sim`), since radar
data files are often large or not shareable.
