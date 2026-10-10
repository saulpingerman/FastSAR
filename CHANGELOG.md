# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The date of a version is the day it
was published on PyPI.

## [0.1.2] - unreleased

### Added

- `CITATION.cff` with the software citation and the paper as the preferred citation.

## [0.1.1] - 2026-10-10

### Added

- `[dependency-groups]` in `pyproject.toml`: `test` (pytest, pytest-timeout, pytest-xdist and the packages the
  tests import) and `dev` (the test group and nox); pytest and pytest-timeout in the `test` extra.
- `noxfile.py` (uv environments): `tests` on Python 3.10 to 3.13, `lowest` with numpy, scipy and jax at the lower
  bounds of `pyproject.toml`, and `cuda` and `tpu` for GPU and TPU machines.
- GitHub Actions workflow running the `tests` and `lowest` sessions on the CPU for each push and pull request.
- `form_cphd(..., troposphere=)`, passed to `read_cphd`; `sim.write_cphd(..., tropo=)` writes `TDTropoSRP`.
- `FASTSAR_CACHE_DIR` (else `$XDG_CACHE_HOME/fastsar`, else `~/.cache/fastsar`) for the compiled CPU kernels, with a
  clear error when the directory cannot be created.
- A packaging job in the workflow: sdist and wheel built and checked, the wheel installed with the `io` and `geo`
  extras and its C++ kernel compiled from site-packages, `examples/chain.py --simulate` run.
- Python version classifiers and lower bounds for the `io`, `geo` and `tpu` extras; the `test` extra now matches
  the `test` dependency group.

### Changed

- `ExactFormer` (and `form_image(..., algorithm='bp')`) oversamples the range profiles 8 times by default with
  cubic interpolation, as with linear interpolation (4 before). On three regions of the Umbra Panama collection,
  against a float64 backprojection with profiles oversampled 64 times, the error falls from -59.8 to -69.6 dB to
  -77.5 to -81.7 dB; the profile buffers double. Pass `upsample=4` for the previous behavior.
- `read_cphd` and `form_cphd` remove the troposphere delay at the scene reference point (PVP `TDTropoSRP`) by
  default when the file gives a nonzero delay (`troposphere=None`; before, only with `troposphere=True`). An uncorrected delay displaces scatterers in range, by 1.4 m on an Umbra collection at Silver Peak, Nevada, and
  by 3.7 m on a Capella stripmap. With the correction the Silver Peak image lies 2 to 3 m closer to its position in
  Sentinel-2 and NAIP imagery, and the Capella image registers to Capella's SICD within a pixel (6 pixels without). Umbra's SICD images keep the delay, so
  `troposphere=False` reproduces their pixel grid.
- A missing optional package raises `ImportError` naming the extra that installs it (`pip install "fastsar[io]"`
  for sarpy in `read_cphd`, `form_cphd`, `write_sicd`, `sim.write_cphd`; `"fastsar[geo]"` for rasterio in
  `write_geotiff`, `read_dem` and map projections) instead of a bare `ModuleNotFoundError`; the CPU backend says
  so when it is not on x86-64 Linux instead of failing in the compiler.
- Documentation remeasured for the release: the exact-backprojection rows at the 8x default, the crossover with
  factorized backprojection, and the accuracy margins against the 64x float64 computation (`docs/performance.md`,
  `docs/comparison.md`, `docs/precision.md`).
- The `sarkit` test dependency is required on Python 3.11 and later only.
- The tests are pytest modules (`uv run pytest`) instead of scripts, with the same checks and limits;
  shared scenes are module fixtures and backend comparisons run once per backend. Tests marked `cuda` or `tpu`
  are skipped without the device, or fail with `--require cuda,tpu`; tests needing a package of the test group
  are skipped without it, or fail with `--require io`. Each test's peak resident memory is checked against
  `--max-rss-gb` (default 12 GB, or `FASTSAR_TEST_MAX_GB`). `tests/run_all.py` runs pytest with its former
  options.

### Fixed

Found by forming collections not used in development (Capella, ICEYE and Umbra open data):

- `ImageFormer` and `form_image` add factorization levels when three cannot split a long grid ("3 levels cannot
  split ..."); an explicit `levels` is kept.
- `form_cphd` without a SICD or `spacing` forms square resolution from the central pulses of a spotlight aperture
  longer than that needs, instead of a grid tens of thousands of pixels long in azimuth, and says so in `notes`.
- `form_cphd` checks host memory before forming and raises a `MemoryError` naming `extent` and `spacing`.
- The CPHD image area comes from `ImageAreaCornerPoints`: Capella writes `ImageArea` in grid lines, which read as
  meters made a 10 by 50 km stripmap footprint 50 by 249 km.
- `read_cphd` reads in pulse blocks into one complex64 array and applies its corrections in place; a 16.5 GB ICEYE
  file needed over 128 GB of memory.
- `backproject` windows the history in row blocks of one complex64 copy; whole-array products took over 128 GB for a
  320,360-pulse Capella spotlight.
- `ImageFormer` on the cpu, jax and tpu backends windows a host history the same way (one complex64 copy, row
  blocks); the whole-array product held three copies of the 54 GB history of that spotlight.
- `products.geolocate` on terrain steeper than the radar's line of sight (mine pits, cliffs) no longer divides by a
  vanishing Newton derivative; `geocode_image` failed with an infinite map extent on such a DEM.

## [0.1.0] - 2026-10-09

First release.

### Added

- `fastsar.ExactFormer`: exact backprojection onto the `form_image` grid, set up once per geometry, with CUDA and
  C++ kernels (float64 tile centers, a third-order expansion within each tile, the tile size chosen from the
  expansion's predicted error, cubic or linear interpolation of cropped range profiles);
  `form_image(..., algorithm='bp')` with `interp`, `upsample` and `center`.
- `fastsar.form_cphd`: one call from a CPHD file to an image; spotlight or moving beam chosen from the scene
  reference point, ground-plane grid at the scene reference height, SICD optional.
- `products.geolocate` and `products.locate`: latitude, longitude and height of `form_cphd` pixels on the image
  plane, at a height or on a DEM, and the inverse; `products.read_dem` for DEM GeoTIFFs.
- `products.geocode_image`: a `form_cphd` image, or data from it, on a north-up map grid (UTM, any rasterio CRS,
  or latitude/longitude) with terrain correction; `write_geotiff` writes complex and multi-band data.
- `products.write_sicd(path, out)` and `products.sicd_meta`: SICD from `form_cphd`'s output with its geometry
  (PLANE grid, aperture polynomial), whose projection matches `geolocate`.
- `form_cphd(..., autofocus=True)` for spotlight collections; the result also carries the band, the spatial
  bandwidths, the windows and the autofocus phase estimate.
- `sim.write_cphd`, `sim.to_ecf`: simulated collections as CPHD 1.0.1 files placed on the Earth.
- `ref=` on `ImageFormer` and `form_image`: per-pulse reference ranges other than `|ant|` (bistatic half paths,
  vendor reference points). Polar format re-references the samples to `|ant|`.
- `ImageFormer.stage(S)`: the checks, scaling and upload of a history ahead of formation on JAX and TPU.
- `fastsar.MemoryWarning` when a former falls back to a slower path for lack of memory (smaller first-level
  groups, CPU pulse blocks, CUDA streaming, a GPU window of mosaic range profiles), stating the memory full speed
  needs; `ImageFormer.memory()`. A TPU that cannot hold the history raises `MemoryError`. `MemoryWarning`
  subclasses `UserWarning`, so `python -W error` or pytest's `filterwarnings = error` turns a fallback into an
  exception.
- Stripmap and sliding spotlight mosaics apply the azimuth window in the final stage of factorized backprojection
  (C++, CUDA and JAX); range profiles are computed once per mosaic; the next patches are prepared on worker threads.
- `patches.gate_bins`, `patches.choose_buckets` and `patch_history(gate_len=)`: JAX and TPU mosaics plan the pulse
  counts and range gates of their compiled programs for the whole mosaic.
- The CUDA former streams a host phase history whose device planes exceed half the free GPU memory through the
  first level; the CPU former reads a complex64 history in place and sizes its first-level groups from the memory
  available.
- JAX and TPU formers cache compiled programs and the plan's device arrays by plan signature (up to 32).
- Polar format keeps its compiled program and geometry arrays per collection geometry (the last four), and applies
  the window, the spectral weighting and the scaling on the device.
- `ExactFormer` on CUDA halves its pulses per chunk on an out-of-memory error and issues a `MemoryWarning`.
- Environment variables `FASTSAR_SHARED_PROFILES`, `FASTSAR_WEIGHT_TERMS`, `FASTSAR_WEIGHT_GRAD`,
  `FASTSAR_MOSAIC_PREFETCH`, `FASTSAR_COMPILE_PATCHES`, `FASTSAR_PULSE_SLACK`, `FASTSAR_CPU_GROUP_GB`,
  `FASTSAR_CUDA_GROUP`, `FASTSAR_TPU_GROUP`, `FASTSAR_CUDA_STREAM` and `FASTSAR_TIMING`
  ([performance.md](docs/performance.md#environment-variables)); numeric values out of range raise `ValueError`.
- `examples/chain.py`.

### Changed

- `read_cphd`'s local frame takes z along the ellipsoid normal at the scene reference point instead of the
  geocentric radial (up to 0.19 degrees apart). `form_cphd`'s ground plane is now horizontal at the scene, as
  documented; its comparisons with vendor images in `docs/real-data.md` were measured before this change.
- `read_cphd`'s meta carries the collection start, collector and core name.
- `form_cphd` references a spotlight history to |ant - c|, the range to the grid center, so its images change.
- JAX and TPU mosaics widen range gates onto shared lengths, so their images differ slightly from before.
- `FASTSAR_MOSAIC_PREFETCH` defaults to up to 4 worker threads on a GPU or TPU (1 before; still 1 on the CPU).
- The JAX program cache holds 32 programs instead of 16.
- CUDA float32 histories are no longer scaled to their peak.
- The CUDA former windows and checks a host history on the GPU as its row blocks are uploaded, without a full
  complex copy on the device (host-side windowing cost a 4-vCPU instance more than the formation); `ExactFormer`
  scans for NaN and inf on the GPU.
- Importing FastSAR sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` unless already set, so that JAX does not take 75% of
  a GPU's memory that the CUDA formers share.
- CPU first-level groups follow the available memory (MemAvailable), so they vary on a busy host.
- `form_image`, `ImageFormer`, `backproject`, `patches.form_mosaic`, `io.read_cphd` and `form_cphd` check their
  inputs: a phase history that is not 2-D or not complex, has NaN or inf samples, fewer than 2 pulses or a pulse
  count different from the antenna path; antenna positions not [pulses, 3]; pixel counts that are not positive
  integers, non-positive spacings, axes that are not orthonormal; unknown backends, and `cuda` or `tpu` on a
  machine without one, raise `ValueError` or `TypeError` with the reason. complex128 and non-contiguous histories
  are converted.

### Fixed

- An all-zero phase history gave a NaN image on the JAX and TPU paths and with polar format, and a
  ZeroDivisionError on the CPU path; it now gives a zero image.
- `api.final_weights` gave weight zero to the final subapertures centered beyond the collection, which hold the
  decimation filters' tails of the edge pulses: a unit `aperture_weight` changed the image by -32 dB. They now
  take the weight of the nearest pulse, and a unit weight leaves the image unchanged.
- `ffbp2.decimator_fir` differed from the column of `ffbp.decimator` by up to 3e-16; it is now equal bit for bit.
- `backproject` with a `ref` shorter than the pulse count read past its end in the C++ kernel.
- `patches.fill_gaps` divided by zero for a single position or a platform at rest.
- `sim.simulate_brute` held a [pulses, samples, scatterers] array (5.8 GB for the dropped-pulse case of
  `tests/test_patches.py`); it now sums in blocks of scatterers.
- A failure while the mosaic prefetches the next patch now shuts its thread down.

### Tests

- `tests/run_all.py` runs every test script with a timeout and a memory limit and prints a summary.
- `tests/test_units.py`: decimation kernels, the JAX program cache, the C++ input paths and group sizes, aperture
  weights, shared range profiles, pulse spans, gap filling, mosaic prefetch, input checks, `ref=`.
- `tests/test_cphd.py`: `read_cphd` and `form_cphd` end to end on simulated spotlight and stripmap collections
  through a stand-in for sarpy's CPHD reader.
- `tests/test_chain.py`: a simulated CPHD through geolocation, map GeoTIFFs, SICD and autofocus.
- `tests/test_planning.py`: bucket planning, the CUDA and TPU memory models, the TPU out-of-memory retry, the
  program cache under concurrent formers, and environment variable checks.
- `tests/test_accuracy.py`: the tile-size and oversampling figures of `docs/algorithms.md`.

### Documentation

- Processing-chain page and API reference; README reduced to a landing page with a supported-data table.
- Documentation matches the geolocation, map GeoTIFF, SICD and autofocus calls for `form_cphd` output; each fact
  stated on one page.
- Shorter README with figures; the long-form material moved to `docs/`.
- Contributing guide, issue and pull request templates, and package metadata for PyPI-style tools.

### Initial functionality

- Spotlight image formation by factorized backprojection with kernels for x86 CPUs (C++/OpenMP), Nvidia GPUs
  (CUDA through CuPy) and Cloud TPUs (Pallas), and the plain JAX program; `form_image` and `ImageFormer`.
- Final tile size chosen from the predicted error of the collection geometry.
- Polar format with the planar-wavefront displacement correction.
- Exact backprojection onto arbitrary points, monostatic or bistatic, with per-pulse reference ranges.
- CPHD reader: per-pulse frequency grids, moving reference points, channels and polarizations, SICD pixel-grid
  alignment, troposphere delay, Capella phase sign, a warning for delay windows beyond the unambiguous range.
- Phase gradient autofocus.
- Stripmap focusing (range-Doppler, omega-k, backprojection), patch mosaics for long apertures and arbitrary
  tracks, ScanSAR and TOPS burst modes.
- Products: multilook, interferogram, coherence, Pauli decomposition, range-Doppler projection and geocoding,
  GeoTIFF and SICD output.

[0.1.2]: https://github.com/saulpingerman/FastSAR/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/saulpingerman/FastSAR/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/saulpingerman/FastSAR/releases/tag/v0.1.0
