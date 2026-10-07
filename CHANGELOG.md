# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). FastSAR has no tagged release yet;
`fastsar.__version__` is 0.1.0.

## Unreleased

### Added

- `products.geolocate` and `products.locate`: latitude, longitude and height of `form_cphd` pixels on the image
  plane, at a height or on a DEM, and the inverse; `products.read_dem` for DEM GeoTIFFs.
- `products.geocode_image`: a `form_cphd` image, or data from it, on a north-up map grid (UTM, any rasterio CRS,
  or latitude/longitude) with terrain correction; `write_geotiff` writes complex and multi-band data.
- `products.write_sicd(path, out)` and `products.sicd_meta`: SICD from `form_cphd`'s output with its geometry
  (PLANE grid, aperture polynomial), whose projection matches `geolocate`.
- `form_cphd(..., autofocus=True)` for spotlight collections; the result also carries the band, the spatial
  bandwidths, the windows and the autofocus phase estimate.
- `sim.write_cphd`, `sim.to_ecf`: simulated collections as CPHD 1.0.1 files placed on the Earth.
- `examples/chain.py` and `tests/test_chain.py`.

### Changed

- `read_cphd`'s local frame takes z along the ellipsoid normal at the scene reference point instead of the
  geocentric radial (up to 0.19 degrees apart). `form_cphd`'s ground plane is now horizontal at the scene, as
  documented; its comparisons with vendor images in `docs/real-data.md` were measured before this change.
- `read_cphd`'s meta carries the collection start, collector and core name.

### Documentation

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
