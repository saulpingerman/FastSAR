# Contributing to FastSAR

Bug reports, questions and pull requests are welcome. Please open an issue first for a change that touches a
kernel or the public API, so the approach can be agreed before you write it.

## Development setup

```bash
git clone https://github.com/saulpingerman/FastSAR
cd FastSAR
pip install -e ".[io]"          # add cuda on a machine with an Nvidia GPU: ".[cuda,io]"
```

or `uv sync`, which installs the versions pinned in `uv.lock`. The CPU backend needs `g++` with OpenMP. Some
tests need packages that are not dependencies of the library: `finufft` (`tests/test_insar.py`), `rasterio`
(`products.write_geotiff`, `read_dem` and map projections other than latitude/longitude) and `sarkit` (optional
consistency checks in `tests/test_chain.py`).

## Running the tests

The tests are plain scripts, not pytest functions. Each prints its measurements, stops with an `AssertionError`
when a check fails, and exits 0 when it passes. Run them from the repository root:

```bash
python tests/test_api.py           # every backend this machine has against the JAX program, and polar format
python tests/test_ffbp_cpu.py      # C++ kernels against the dense JAX image, two and three levels
python tests/test_ffbp_cuda.py     # CUDA kernels against the dense JAX image (needs a GPU)
python tests/test_pallas_fused.py  # TPU level kernel in interpret mode (runs on a CPU)
python tests/test_pallas_final.py  # TPU final-stage kernel in interpret mode (runs on a CPU)
python tests/test_pallas_e2e_tpu.py  # TPU kernels end to end in interpret mode (runs on a CPU)
python tests/test_autofocus.py cpu # phase gradient autofocus; the argument is the backend (default jax)
python tests/test_stripmap.py      # stripmap omega-k and RDA against float64 backprojection
python tests/test_patches.py       # patch mosaics for stripmap and non-linear tracks
python tests/test_burst.py         # ScanSAR and TOPS burst focusing
python tests/test_products.py      # layover projection, geocoding, multilooking, Pauli
python tests/test_bp.py            # exact backprojection: backends, bistatic, orbital range, moving reference
python tests/test_io.py            # CPHD helpers: frequency resampling, re-referencing, geodetic conversions
python tests/test_wide_angle.py    # 10 to 360 degree apertures against exact backprojection
python tests/test_insar.py         # change detection and interferometric height (needs finufft)
python tests/test_chain.py         # simulated CPHD to geolocation, map GeoTIFFs, SICD and autofocus (needs sarpy)
```

`test_autofocus.py` is the only script that takes a backend argument. The others pick the backends this machine
has (`fastsar.available_backends()`) or use the CPU. The tests use small simulated scenes and take seconds to a few
minutes each. For the CPU kernels, set `OMP_NUM_THREADS` to the number of physical cores.

`test_ffbp_cpu.py`, `test_ffbp_cuda.py` and the three `test_pallas_*.py` scripts compare each error with a limit
set about 5 dB above the value measured when the limit was set, and exit with a list of the failed cases.

## Pull requests

- Keep a pull request to one change, and say in the description which tests you ran and on which devices.
- A change to a kernel should report the error against the JAX program or exact backprojection before and after,
  as the tests print it.
- A change that adds a result to the documentation should say what data it comes from. Results from simulated
  data are labeled as test results.
- Do not commit data files (CPHD, SICD, NITF, `.npy`) or build artifacts.

## Reporting a bug

Use the bug report template. Include the FastSAR commit, the backend, the device, the Python, JAX and CuPy
versions, and if possible a script that reproduces the problem on a simulated scene (`fastsar.sim`), since radar
data files are often large or not shareable.
