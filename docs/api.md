# API reference

`...` marks keywords the docstrings describe. Positions are in meters in `io.read_cphd`'s local frame
([real-data.md](real-data.md#reading-cphd)). Invalid inputs raise `ValueError` or `TypeError`.

## `fastsar`

- `form_cphd(cphd, sicd=None, mode='auto', backend='auto', window=True, spacing=None, channel=0, patch=1024,
  azimuth_fraction=0.8, extent=None, height=None, precision='float32', target_db=-40.0, info=None,
  autofocus=False, troposphere=None)`: the image of a CPHD file ([processing-chain.md](processing-chain.md#2-form)).
  `mode`: `'auto'`, `'spotlight'`, `'moving'`. `troposphere`: passed to `read_cphd` (default: the delay the file
  gives is removed; `'model'` removes a standard-atmosphere delay for files that give none). `sicd`: path, `.xml` or sarpy SICDType. `spacing`, `extent`: (along, across
  track) in m. `height`: grid height above the ellipsoid. Returns a dict: `image` [nx, ny] complex64, `origin`
  (pixel (0, 0)), `e1`, `e2`, `spx`, `spy`, `mode`, `band` (first and last frequency), `bandwidth` (spatial
  frequency support along e1, e2), `window`, `phase_error` (or None), `meta` (from `read_cphd`), `notes`.
- `form_image(S, ant, fmin, df, nx, ny, spx, spy, e1, e2, algorithm='ffbp', backend='auto', precision='float32',
  window=True, T='auto', ..., target_db=-40.0, ref=None, interp=None, upsample=None, center=None)`: a spotlight
  image on a grid centered on the frame origin ([algorithms.md](algorithms.md)). `algorithm`: `'ffbp'`, `'pfa'` or
  `'bp'` (exact backprojection, `ExactFormer`). `ref` [P]: the one-way range each pulse's samples are referenced to
  when it is not `|ant|` (a bistatic half path, a vendor's reference point). `precision`, `T`, `levels`, `pmax` and
  `target_db` apply to `'ffbp'`, `pfa_guard` to `'pfa'`; `interp` (`'cubic'` or `'linear'`), `upsample` and
  `center` (the point the grid is centered on) apply to `'bp'` only and raise `ValueError` with the others.
- `ImageFormer(ant, fmin, df, K, nx, ny, spx, spy, e1, e2, *, backend='auto', precision='float32', ...,
  aperture_weight=None, ref=None)`: plans and compiles once; `former(S)` forms each image. Arguments after `e2` are
  keyword-only. Attributes `T`, `predicted_error_db`. `aperture_weight(points, pulses) -> W [n, m]` weights pulses
  per pixel. `memory()`: the bytes full speed needs and the bytes free, as
  `dict(backend, needed, available, full_speed, parts)`. `stage(S)`: the checks, scaling and upload of `S` ahead
  of formation on the JAX and TPU backends; `former(former.stage(S))` equals `former(S)`.
- `ExactFormer(ant, fmin, df, K, nx, ny, spx, spy, e1, e2, *, backend='auto', window=True, interp='cubic',
  upsample=None, ref=None, center=None, chunk=1024)`: exact backprojection onto the `form_image` grid (centered on
  `center`, default the origin), set up once; `former(S)` forms each image (`S` may be a CuPy array on the cuda
  backend). Arguments after `e2` are keyword-only. `interp`: `'cubic'` (default) or `'linear'`, both at
  `upsample=8` by default. Attributes `tile` (the pixel tile chosen from the range expansion's predicted error,
  [algorithms.md](algorithms.md#exact-backprojection)) and `predicted_error_db`; a `UserWarning` when even the
  smallest tile misses 3e-4 rad. `memory()`: as for `ImageFormer`, the bytes of the per-chunk buffers and the
  image. A grid wider in range than `c / (2 df)` raises `ValueError`. Monostatic; for bistatic geometry or arbitrary
  points use `backproject`. One former is not safe to call from two threads at once.
- `MemoryWarning`: a `UserWarning` subclass issued when a former falls back to a slower path for lack of memory
  ([performance.md](performance.md#memory)). A TPU that cannot hold the history raises `MemoryError`.
- `backproject(S, ant, fmin, df, points, rcv=None, ref=None, backend='auto', upsample=8, window=True, chunk=256)`:
  exact backprojection at points [..., 3]; `rcv` for bistatic, `ref` for per-pulse reference ranges.
- `plane_points(nx, ny, spx, spy, e1, e2, height=None)`: the `form_image` grid as points [nx, ny, 3].
  `available_backends()`: backends this machine runs, in `'auto'` order.

Backends: `'cpu'`, `'cuda'`, `'tpu'`, `'jax'`, `'auto'`. Precision: `'float32'`, `'float16'` (CUDA, JAX),
`'single-pass'`, `'three-pass'` (TPU) ([precision.md](precision.md)).

## `fastsar.io`

- `read_cphd(cphd, sicd=None, channel=0, meta=False, regrid_tol=1e-3, drop_flagged=False, troposphere=None,
  phase_sign=None)`: one channel as `dict(S, ant, fmin, df)`, plus `nx, ny, spx, spy, e1, e2` with a SICD.
  `channel`: index, identifier or polarization. `meta=True` also returns `tx`, `rcv`, `ref`, `R`, `origin`, `srp`,
  `tx_time`, `rcv_time`, `pulses`, `polarization`, `channel`, `mode`, `start`, `collector`, `core_name`, `notes`.
- `troposphere_delay(tx, rcv, srp)`: two-way delay [P] (s) of a standard atmosphere at the scene reference point
  (Saastamoinen hydrostatic zenith delay, cosecant mapping); what `read_cphd(troposphere='model')` removes.
- `sicd_points(sicd, rows, cols, meta, hae=None)`: local positions of SICD pixels on any grid type.
- `local_to_ecf(points, meta)`, `ecf_to_local(points, meta)`, `ecf_to_geodetic(ecf)`, `geodetic_to_ecf(lat, lon,
  h)`: frame and WGS-84 conversions (degrees, meters).
- `rereference(S, fmin, df, dref)`: moves each pulse's reference point by `dref` m of range.

## `fastsar.products`

- `multilook(img, la=1, lr=1)`, `to_db(power)`, `interferogram(a, b, la=1, lr=1)`, `coherence(a, b, w1=5, w2=5)`,
  `pauli(hh, hv, vv, vh=None, ...)`.
- `geolocate(out, i, j, height=None, iterations=10, tol=1e-4)`: latitude, longitude, ellipsoid height of pixels of
  a `form_cphd` output, on the image plane or on a surface (number or `dem(lat, lon)`).
- `locate(out, lat, lon, height=0.0)`: fractional pixel coordinates [..., 2] of ground points.
- `geocode_image(out, data=None, spacing=None, crs=None, height=None, order=1)`: data on a north-up map grid (default
  the scene's UTM zone) as `dict(data, transform, crs)`, the arguments of `write_geotiff`.
- `read_dem(path, offset=0.0)`: a DEM GeoTIFF as `dem(lat, lon)`; `offset` adds the geoid height.
- `project(points, ant, nx, ny, spx, spy, e1, e2)`, `sample(img, ij, order=1)`, `geocode(img, grid, ant, points,
  order=1)`: the same model on a `form_image` grid.
- `write_sicd(path, out)`, `write_sicd(path, img, template, transpose=None)`: SICD of a `form_cphd` output, or of
  an image on a template SICD's grid. `sicd_meta(out)`: the metadata and the SICD-ordered array.
- `write_geotiff(path, data, transform, crs='EPSG:4326', nodata=nan)`: [rows, cols] or [bands, rows, cols].

SICD needs sarpy; GeoTIFFs, DEMs and map projections other than `EPSG:4326` need rasterio.

## `fastsar.autofocus`

- `autofocus(S, ant, fmin, df, nx, ny, spx, spy, e1, e2, rounds=2, pga_kwargs=None, **form_kwargs)`: returns
  `(image, phi)`, phi per pulse. `pga(img, ...)` works on an image in the polar-format convention; `deramp_phase`
  and `pulse_bins` support it.

## `fastsar.patches`

- `form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2, patch=(128, 128), beam=None, umax=1.0, awin=None, ...,
  backend='cpu', exact=False, ...)`: image on the grid `origin + i spx e1 + j spy e2`. `fx`: `dict(S, fmin, df,
  ref, band)`. `beam(idx, points) -> u`: normalized azimuth coordinate; pulses with |u| <= `umax` serve a patch.
- `form_stripmap(data, p, ...)`: mosaic of straight-track echoes; returns `(image, r, x)`.
- `echoes_to_fx`, `stripmap_beam`, `straight_track`, `simulate`, `backproject`; the mosaic steps `range_profiles`,
  `patch_history`, `tile_plan`, `beam_span`, `weight_terms`, `fill_gaps`; `report_timing()` (with
  `FASTSAR_TIMING=1`).

## `fastsar.stripmap`, `fastsar.burst`

- `stripmap.make_params(...)` (X-band airborne defaults), `stripmap.simulate(p, targets, amp=None)`,
  `stripmap.focus_stripmap(data, p, algorithm='omegak', ...)` with `'omegak'`, `'rda'` or `'bp'`, returning
  `(image, r, x)`; also `omegak`, `rda`, `backproject`, `window(spec)`.
- `burst.make_bursts(mode='tops', nburst=1, ...)`, `burst.simulate(bp, targets, amp=None)`,
  `burst.focus_burst(raw, bp, algorithm='omegak', ...)` returning `(image, r, x)`, `burst.mosaic(images, bursts,
  ...)`.

## `fastsar.quality`, `fastsar.sim`

- `quality.point_target(img, ij, d_az=1.0, d_rg=1.0, ...)`: resolution, PSLR, ISLR, peak. `quality.upsample`.
- `sim.make_collect(...)`, `sim.simulate(col, pos, amp)` (needs finufft), `sim.simulate_brute(col, pos, amp)`.
- `sim.write_cphd(path, col, S, lat, lon, height=0.0, heading=0.0, ...)`: a simulated collection as a CPHD 1.0.1
  file placed on the Earth. `sim.to_ecf(points, lat, lon, height=0.0, heading=0.0)`.
