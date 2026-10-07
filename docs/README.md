# FastSAR documentation

- [Algorithms and API](algorithms.md): coordinates and arguments, factorized backprojection and tile size,
  exact backprojection, polar format, wide-angle apertures, autofocus
- [Stripmap, long apertures and burst modes](stripmap-burst.md)
- [Real data](real-data.md): CPHD reading, SICD grids, Umbra and Capella collections, vendor conventions
- [Products](products.md): interferometry, change detection, geocoding, SICD and GeoTIFF output
- [Precision and image quality](precision.md)
- [Performance](performance.md): time and cost per image by device
- [Comparison with open-source implementations](comparison.md)

Results on real data come from Umbra and Capella open-data collections. Numbers labeled as test results come
from the scripts in `tests/`, which use small simulated scenes; see [CONTRIBUTING.md](../CONTRIBUTING.md) to run
them. The function docstrings in `fastsar/` give every argument.
