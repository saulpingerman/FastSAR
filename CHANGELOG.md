# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). FastSAR has no tagged release yet;
`fastsar.__version__` is 0.1.0.

## Unreleased

### Added

- `fastsar.ExactFormer`: exact backprojection onto the `form_image` grid, set up once per geometry, with CUDA and
  C++ kernels (float64 tile centers, a second-order expansion within each tile, cubic or linear interpolation of
  cropped range profiles); `form_image(..., algorithm='bp')`.
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
  stated on one page, about 7,000 words in all.
- Shorter README with figures; the long-form material moved to `docs/`.
- Contributing guide, issue and pull request templates, and package metadata for PyPI-style tools.

## 0.1.0 (October 2026, untagged)

First public version.

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
