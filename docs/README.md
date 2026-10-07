# FastSAR documentation

Using FastSAR:

- [Processing chain](processing-chain.md): from a CPHD file to products, one call per step, and the steps not
  covered yet
- [API reference](api.md): signatures and purpose of the public functions and classes
- [Real data](real-data.md): CPHD reading, `form_cphd`, SICD grids, Umbra, Capella and ICEYE collections, vendor
  conventions

Methods and results:

- [Spotlight image formation](algorithms.md): coordinates, factorized and exact backprojection, tile size, polar
  format, wide-angle apertures, autofocus
- [Stripmap, long apertures and burst modes](stripmap-burst.md): omega-k, range-Doppler, patch mosaics, ScanSAR
  and TOPS
- [Products](products.md): interferometry, change detection, geocoding, SICD and GeoTIFF output
- [Precision and image quality](precision.md)
- [Performance](performance.md): time and cost per image by device, large collections, environment variables
- [Comparison with open-source implementations](comparison.md)

Results on real data come from Umbra and Capella open-data collections. Numbers labeled as test results come from
the scripts in `tests/`, which use small simulated scenes; [CONTRIBUTING.md](../CONTRIBUTING.md) explains how to
run them.
