# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). FastSAR has no tagged release yet;
`fastsar.__version__` is 0.1.0.

## Unreleased

### Added

- `fastsar.form_cphd`: one call from a CPHD file to an image; spotlight or moving beam chosen from the scene
  reference point, ground-plane grid at the scene reference height, SICD optional.
- Stripmap and sliding spotlight mosaics apply the azimuth window in the final stage of factorized backprojection
  (C++, CUDA and JAX); range profiles computed once per mosaic; the next patch prepared on a second thread.
- CUDA former streams a host phase history larger than 30% of the free GPU memory through the first level; the
  CPU former reads a complex64 history in place and sizes its first-level group from the memory available.
- JAX and TPU formers cache compiled programs and the plan's device arrays by plan signature.
- Environment variables `FASTSAR_SHARED_PROFILES`, `FASTSAR_WEIGHT_TERMS`, `FASTSAR_MOSAIC_PREFETCH`,
  `FASTSAR_CPU_GROUP_GB`, `FASTSAR_CUDA_STREAM` and `FASTSAR_TIMING`.

### Documentation

- Processing-chain page and API reference; README reduced to a landing page with a supported-data table.

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
