# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The date of a version is the day it
was published on PyPI.

## [0.1.3] - unreleased

### Added

- `FASTSAR_PINNED` (`0` returns the cuda image by a plain copy instead of through pinned memory) and `FASTSAR_FINAL`
  (`fp32` keeps the cuda float32 final stage on the CUDA cores). Testing settings for the cuda kernels' launch
  parameters: `FASTSAR_FIRP`, `FASTSAR_CUDA_PB`, `FASTSAR_CUDA_JT`, `FASTSAR_FIRP_SMEM`, `FASTSAR_FINAL_PPS`,
  `FASTSAR_FINAL_COLB`, `FASTSAR_TC_KERNEL`, `FASTSAR_TC_PPS`, `FASTSAR_TC_MW`, `FASTSAR_TC_SPEC`, `FASTSAR_TC_NH`
  ([performance.md](docs/performance.md#environment-variables)). Settings with fixed choices reject other values
  with a `ValueError` that names the variable.

### Changed

- CUDA float32: the final stage runs on the tensor cores as three bfloat16 products (high by high, high by low, low
  by high), as on the TPU, on cards of compute capability 8.0 and later when its tables fit the shared memory. The
  image agrees with the CUDA-core kernel's to -87 dB and its error is unchanged (-59.94 dB on Panama). This kernel
  needs `nvcc`; without it float32 stays on the CUDA cores, with a warning.
- CUDA final stage: a new tensor-core kernel builds the next pulses' tables while the current products run, and builds
  each table by columns, reading each phase-history sample once instead of once per pixel row. The CUDA-core kernel
  builds its tables by columns too (7 to 17% faster). Aperture weights of mosaics are applied in the kernel, which
  makes the weighted float16 image 12 dB closer to float32.
- CUDA uploads: the phase history goes to the GPU through two pinned staging buffers, the copy of the next block
  overlapping the transfer of the current one, and the image comes back through pinned host memory.
- CUDA float16: the scale of the half-precision planes is the first uploaded block's peak, taken on the device,
  instead of a pass over the whole history on the host (0.3 to 0.5 s per image). A zero first block, or a later block
  that would overflow the half range, falls back to the whole history's peak.
- CUDA pulse filter: cards with at least 100 streaming multiprocessors read the filter's rows from global memory,
  which was faster on the A100 and H100 and slower on the L4.
- Mosaics: the decimation filters are memoized across patches, the band-center phases and the final stage's
  per-pulse geometry run as one fused kernel each on `cuda`, and the aperture weights' quadrature is summed in place
  of a BLAS product (which on large hosts started a thread pool per prefetch worker).
- `form_cphd`: the host work runs on all cores: the reader's troposphere and spotlight re-reference phases (one pass
  over the whole history each), the window, and in `cuda` mosaics the gather of a patch's range profiles when they
  exceed the GPU window, now done outside the lock that serializes the GPU. The phases are float64 cosines and
  sines rounded to float32 and differ from 0.1.2's in the last bit of some samples (-193 dB relative).
- Speed on the Panama image of the paper: float32 from 3.86 to 2.9 s on an L4, from 1.87 to 1.07 s on an A100 and
  from 1.04 to 0.57 s on an H100; the float16 final stage from 0.41 to 0.20 s on an L4. The 84-patch Capella
  stripmap goes from 6.9 to 5.0 s on an H100 and from 26.9 to 22.2 s on an L4 with a 4-vCPU host. End to end from
  the file on a g2-standard-32 host with an L4, a Capella spotlight (17 GB in memory) goes from 188 to 46 s and a
  sliding spotlight (24 GB) from 1450 to 331 s; before, the L4 was slower end to end than the host's CPU backend.

### Fixed

- CUDA out of memory in the later levels: the first-level group is sized with room for one of its children carried
  through the later levels, a later level that does not fit is formed over parts of its children, and one that runs
  out of memory for a single child halves the group. A 12.3 GB UMBRA-10 spotlight that failed on an L4 in both
  precisions now forms in 157 s (float32) and 72 s (float16). When nothing fits, a `MemoryError` names the remedy
  instead of CuPy's `OutOfMemoryError`.
- CUDA mosaics return the pinned memory of their patch gathers after the mosaic, and the pinned gather of a patch is
  complete before its buffer is reused.
- `ImageFormer.memory()` and the full-speed figure of the CUDA `MemoryWarning` include the later levels.

## [0.1.2] - 2026-10-14

### Added

- `io.read_nisar`: NISAR L0B RRSD granules (HDF5) read as a CPHD is, one frequency and polarization at a time: the
  echoes decoded (BFPQ), range compressed with the file's chirp replica and taken to the frequency domain
  (`fastsar.raw`), compensated to each pulse's zero-Doppler ground point at mid swath, the antenna from the file's
  orbit state vectors (Hermite). `form_cphd` accepts the `.h5` file (`io.read_collection` dispatches) and forms it
  in moving mode with the swath as footprint. `sim.write_nisar` writes a look-alike for tests. Extra `raw` (h5py).
- `io.read_palsar`: ALOS PALSAR level 1.0 products (CEOS raw signal data, a directory, zip or `IMG-` file with its
  leader): 8-bit I and Q lines range compressed with the leader's chirp, line times from the 1PPS microsecond
  counter, the antenna from the platform position record; the same frequency-domain output as `read_cphd`, formed
  by `form_cphd` in moving mode. `sim.write_palsar` writes a look-alike for tests.
- `io.read_sentinel1`: Sentinel-1 Level-0 products (a .SAFE directory, its zip, or one measurement file): the
  instrument source packets parsed (`fastsar.sentinel1`), the echoes decoded from FDBAQ, BAQ or bypass user data by a
  C++ decoder checked bit for bit against a public decoder on real data, range compressed with the replica the
  headers describe, the antenna from the sub-commutated position and velocity records; the same frequency-domain
  output as `read_cphd`, formed by `form_cphd` in moving mode. Stripmap and wave modes; the bursts of IW and EW are
  not split yet. `sim.write_sentinel1` writes a look-alike for tests. `form_cphd(reader_options=...)` passes
  `pulses=(first, last)`, `polarization` and the like to the raw-data readers.
- `form_image(pfa_support='inscribed')`: polar format on the rectangle common to all pulses instead of the union
  of their bands (the default), a clean window at the cost of the bandwidth outside the rectangle.
- `read_cphd(troposphere='model')` and `form_cphd(troposphere='model')`: the troposphere delay of a standard
  atmosphere (`io.troposphere_delay`, Saastamoinen hydrostatic zenith delay mapped by the elevation of each antenna)
  removed at the scene reference point, for files that give no delay. ICEYE's CPHD files write `TDTropoSRP` zero
  while ICEYE's images include the correction; an uncorrected image lay 6 m in ground range from ICEYE's on a
  26 degree incidence dwell.
- `CITATION.cff` with the software citation and the paper as the preferred citation.
- The raw readers remove the instruments' range delays, as the agencies' processors do: `read_nisar(range_delay=)`
  subtracts JPL's calibration common delay (`io.NISAR_RANGE_DELAY`, 57.125 m for frequency A and 19.65 m for B,
  the values of JPL's products processed from September 2026; a number, or an RSLC or GCOV path to read the
  product's value, `io.nisar_product`); `read_sentinel1(internal_delay=)` subtracts the internal time delay
  estimated from the product's calibration packets (`sentinel1.internal_delay`, about 0.43 us, within 4 ns of
  ESA's annotated value on a 2014 datatake). Compared in the agencies' own radar coordinates, the images agree
  with JPL's RSLC of a NISAR pass to 0.2 m and with ESA's SLC of a Sentinel-1 datatake to 0.5 m in slant range
  (1.4 m and 1.3 to 1.5 m along track), from 100 to 180 m before the delays were removed. The Sentinel-1 packet
  time stamp is taken as latched 1.439 us before the packet's own transmit pulse (`sentinel1.T_LATCH`, ESA's Level 1
  Detailed Algorithm Definition 9.23.2) and the fine time code as the protocol interprets it (code plus a half);
  ESA's processor takes the stamp as the receive time of the window's first sample by default, which puts the
  annotated azimuth times of its products about 0.8 m of track before the physical ones (the reader placed its
  image 0.9 to 1.1 m along track from the SLC when it mimicked that convention, as a check).
- `read_nisar(orbit=)`: JPL's orbit from an RSLC or GCOV product of the pass in place of the L0B's forecast orbit
  (10 m off radially on a September 2026 pass); `read_sentinel1(orbit=)`: an ESA precise or restituted orbit file
  (`sentinel1.orbit_from_file`) in place of the packets' records, which agree with the precise orbit to 1.4 m on a
  2014 datatake. The raw readers remove the standard atmosphere's troposphere delay by default (`troposphere=None`
  keeps it; the look-alike writers put the delay into their echoes) and take `reference_range=`, the slant range of
  the zero-Doppler reference points, since the receiver's position is taken at the reference point's echo time
  (2.5 m along track per 100 km of range from the reference). Against the surveyed corner reflectors of the
  Rosamond site, a NISAR image with JPL's orbit falls 1.4 m along track and 5.7 m across track from the survey and
  within 0.2 m of slant range of JPL's RSLC of the pass; the remainder is the ionosphere neither models.
- `read_nisar` resamples dithered pulse timing onto uniform pulse times for the factorized former (-60 dB against
  exact backprojection on a dithered granule, from -38 dB); `read_palsar` uses the downward chirp and the sample
  count from the record length, as JAXA's files need.
- `fastsar/par.hpp`: the CPU kernels' parallel loops on OpenMP where the kernels are built with it and on
  `std::thread` otherwise; macOS builds with Apple's clang and `-fopenmp-simd`, with no OpenMP runtime (a second
  runtime next to the one finufft's wheel bundles deadlocked the test suite) and no Homebrew libomp.
- `backend='metal'` of `ExactFormer`: exact backprojection on Apple GPUs (`fastsar.metal`), the CUDA tile kernel
  as a Metal compute kernel with the tile centers' float64 terms computed by numpy per chunk (Apple GPUs have no
  float64), through a small Objective-C++ bridge compiled on first use with Apple's clang; the shader compiles at run
  time, no Xcode needed. On an M2 MacBook Air the 1024 by 1024 image of 5,794 pulses takes 1.4 s (3.4 s on its CPU
  cores) and 2048 by 2048 takes 3.2 s (11.9 s), at 7.6 billion pixel-pulse pairs per second, agreeing with the CPU
  kernel to 2e-7 of the peak; `backend='auto'` picks it for `ExactFormer` on a Mac. The factorized former has no
  Metal kernels and runs on the CPU cores.
- ARM and Apple silicon: the factorized CPU kernel's vectors are four lanes on ARM (NEON) with the final tile
  stage templated on the vectors per table row (the x86 code is unchanged), the exact CPU kernel's sines are
  vectorizable polynomials (3.7e-9 maximum error) on every target (15% faster than glibc's vector library on x86,
  the only vector option on macOS and ARM), macOS builds with `-mcpu=native`, and the automatic tile choice takes
  16 pixels on ARM. On a c4a-highmem-16 (Axion) the factorized Panama image takes 20.3 s, from 43.7 s. On an M2 MacBook Air the synthetic 4096 by 4096 image of
  `oss/bench_cpu_synth.py` (5,794 pulses) forms in 3.7 s factorized and 3.5 s exact at 1024 by 1024, from 6.3 and
  8.0 s before; eight Zen 4 threads with AVX-512 take 2.7 s for each.

### Changed

- The options of `form_image` (after `e2`), `form_cphd` (after `sicd`) and `backproject` (after `points`) are
  keyword-only, as those of `ImageFormer` and `ExactFormer` already were. A call that passed `algorithm`, `mode` or
  `rcv` by position raises a TypeError; name the argument.

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

[0.1.3]: https://github.com/saulpingerman/FastSAR/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/saulpingerman/FastSAR/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/saulpingerman/FastSAR/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/saulpingerman/FastSAR/releases/tag/v0.1.0
