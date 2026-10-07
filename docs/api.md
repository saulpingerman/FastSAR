# API reference

Public functions and classes, by module. Arrays are numpy unless stated. Positions are in meters in the local frame
of `io.read_cphd` (x along track, y along ground range away from the radar, z up). The docstrings give every
argument.

## Top level (`fastsar`)

- `form_cphd(cphd, sicd=None, mode='auto', backend='auto', window=True, spacing=None, channel=0, patch=1024,
  azimuth_fraction=0.8, extent=None, height=None, precision='float32', target_db=-40.0, info=None)`:
  forms the image of a CPHD file in one call. `mode`: `'auto'`, `'spotlight'` or `'moving'`. `sicd`: path,
  `.xml` metadata or sarpy SICDType, optional. `spacing`, `extent`: (along track, across track) in meters.
  `height`: grid height above the ellipsoid. Returns a dict with `image` [nx, ny] complex64, `origin` (pixel
  (0, 0)), `e1`, `e2`, `spx`, `spy`, `mode`, `meta` and `notes`.
- `form_image(S, ant, fmin, df, nx, ny, spx, spy, e1, e2, algorithm='ffbp', backend='auto', precision='float32',
  window=True, T='auto', levels=3, pmax=0.4, pfa_guard=300.0, target_db=-40.0)`: forms a spotlight image
  [nx, ny] complex64 on a grid centered on the frame origin. `algorithm`: `'ffbp'` or `'pfa'`.
- `ImageFormer(ant, fmin, df, K, nx, ny, spx, spy, e1, e2, backend='auto', precision='float32', window=True,
  T='auto', levels=3, pmax=0.4, target_db=-40.0, aperture_weight=None)`: plans and compiles factorized
  backprojection once for one geometry; `former(S)` forms each image. Attributes `T`, `predicted_error_db`.
  `aperture_weight(points, pulses) -> W [n, m]` weights pulses per pixel in the final stage.
- `backproject(S, ant, fmin, df, points, rcv=None, ref=None, backend='auto', upsample=8, window=True,
  chunk=256)`: exact backprojection at any points [..., 3]. `rcv`: receiver positions (bistatic). `ref`: per-pulse
  reference ranges for a moving reference point.
- `plane_points(nx, ny, spx, spy, e1, e2, height=None)`: the points [nx, ny, 3] of the `form_image` grid,
  optionally lifted by a height map.
- `available_backends()`: the backends this machine can run, in the order `'auto'` tries them.

Backends: `'cpu'`, `'cuda'`, `'tpu'`, `'jax'`, `'auto'`. Precision: `'float32'`, `'float16'` (CUDA),
`'single-pass'`, `'three-pass'` (TPU). See [precision.md](precision.md).

## `fastsar.io`

- `read_cphd(cphd, sicd=None, channel=0, meta=False, regrid_tol=1e-3, drop_flagged=False, troposphere=False,
  phase_sign=None)`: reads a frequency-domain CPHD channel into `dict(S, ant, fmin, df)`, plus the SICD grid
  `nx, ny, spx, spy, e1, e2` when `sicd` is given. `channel`: index, identifier or polarization. With `meta=True`
  also returns `tx`, `rcv`, `ref`, `R`, `origin`, `srp`, `tx_time`, `rcv_time`, `pulses`, `polarization`,
  `channel`, `mode` and `notes`.
- `sicd_points(sicd, rows, cols, meta, hae=None)`: local positions of SICD pixels on any grid type.
- `local_to_ecf(points, meta)`, `ecf_to_local(points, meta)`: frame conversions.
- `ecf_to_geodetic(ecf)`, `geodetic_to_ecf(lat, lon, h)`: WGS-84 conversions (degrees, meters).
- `rereference(S, fmin, df, dref)`: moves each pulse's motion-compensation point by `dref` meters of range.

## `fastsar.patches` (long apertures and any track)

- `form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2, patch=(128, 128), crop=0, beam=None, umax=1.0,
  awin=None, pulses=None, margin=None, guard=None, backend='cpu', T='auto', levels='auto', target_db=-40.0,
  sub='auto', wtol_db=None, exact=False, info=None)`: image [nx, ny] on the grid `origin + i spx e1 + j spy e2`,
  formed patch by patch. `fx`: `dict(S, fmin, df, ref, band)`. `beam(idx, points) -> u`: normalized azimuth
  coordinate; pulses with |u| <= `umax` serve a patch. `awin`: azimuth window (`stripmap.window` names).
  `exact=True`: exact backprojection per patch.
- `form_stripmap(data, p, compressed=False, rwin=None, awin=None, umax=1.0, rows=None, cols=None, patch=(128, 128),
  crop=0, backend='cpu', **kw)`: patch mosaic of straight-track echoes on the zero-Doppler grid; returns
  `(image, r, x)`.
- `echoes_to_fx(data, p, compressed=False, rwin=None)`: time-domain echoes to a frequency-domain phase history
  with per-pulse reference ranges.
- `stripmap_beam(p, ant, direction=(1, 0, 0))`: the `beam` function of a stripmap antenna on any track.
- `straight_track(p, height=0.0)`, `simulate(p, ant, targets, amp=None, beam=None)`,
  `backproject(data, p, ant, points, ...)`: track, simulator and float64 reference for any track.
- `range_profiles(fx, guard=None)`, `patch_history(fx, ant, center, pts, lo, hi, margin=None, guard=None,
  prof=None)`: the shared range profiles and one patch's gated phase history.
- `tile_plan(...)`, `beam_span(beam, P, pts, umax, step=64)`, `weight_terms(W, wtol_db=-50.0, most=8)`,
  `fill_gaps(a)`: final tile choice, pulse span, separable weight terms, dropped-pulse filling.
- `report_timing()`: the step times collected with `FASTSAR_TIMING=1`, as text.

## `fastsar.stripmap` (straight-track stripmap)

- `make_params(fc=9.6e9, B=100e6, Tp=2e-6, fs=125e6, prf=650.0, v=200.0, r0=5000.0, La=1.5, beamwidth_deg=None,
  squint_deg=0.0, swath=200.0, na=1024, pattern='sinc2', extent=None)`: a `StripParams` with airborne X-band
  defaults.
- `simulate(p, targets, amp=None)`: raw echoes of point targets at (x, zero-Doppler range).
- `focus_stripmap(data, p, algorithm='omegak', compressed=False, rows=None, cols=None, **kw)`: focuses with
  `'omegak'`, `'rda'` or `'bp'`; returns `(image, r, x)`. Keywords `rwin`, `awin`, `umax`, `r_ref`, `dtype`,
  `taps`, `src`.
- `omegak(...)`, `rda(...)`, `backproject(...)`: the three algorithms called directly.
- `window(spec)`: window function from `None`, `'taylor'`, `'hann'`, `'kaiser'`, a tuple or a callable.
- `range_compress`, `matched_filter`, `axes`, `doppler`, `pattern`, `resolution`, `irf_width`: helpers.

## `fastsar.burst` (ScanSAR and TOPS)

- `make_bursts(mode='tops', nburst=1, n=None, gap=None, kpsi=None, alpha=0.25, subswaths=None, el_width=None,
  **kw)`: a list of `BurstParams`.
- `simulate(bp, targets, amp=None)`: one burst's echoes.
- `focus_burst(raw, bp, algorithm='omegak', compressed=False, rwin=None, awin=None, umax=1.0, r_ref=None,
  dtype='float32', taps=16, up=None, margin=32)`: focuses one burst; returns `(image, r, x)`.
- `mosaic(images, bursts, r_ref=None, umax=1.0)`: combines burst images of one subswath.
- `backproject`, `coverage`, `resolution`, `upsampling`, `elevation`: reference and helpers.

## `fastsar.autofocus`

- `autofocus(S, ant, fmin, df, nx, ny, spx, spy, e1, e2, rounds=2, pga_kwargs=None, **form_kwargs)`: forms,
  estimates and removes a per-pulse phase error; returns `(image, phi)`.
- `pga(img, iterations=20, ...)`: phase gradient autofocus on an image in the polar-format convention.
- `deramp_phase(ant, fmin, df, K, nx, ny, spx, spy, e1, e2)`, `pulse_bins(ant, fmin, df, K, nx, spx, e1)`:
  the deramp phase and the azimuth bin of each pulse.

## `fastsar.products`

- `multilook(img, la=1, lr=1)`, `to_db(power, floor=1e-30)`: detected power and decibels.
- `interferogram(a, b, la=1, lr=1)`, `coherence(a, b, w1=5, w2=5)`, `pauli(hh, hv, vv, vh=None, la=1, lr=1)`:
  interferometric, change detection and polarimetric products.
- `project(points, ant, nx, ny, spx, spy, e1, e2)`, `sample(img, ij, order=1)`, `geocode(img, grid, ant, points,
  order=1)`: range-Doppler projection, sampling and terrain-corrected geocoding.
- `write_sicd(path, img, template, transpose=None)`, `write_geotiff(path, data, transform, crs='EPSG:4326',
  nodata=nan)`: output (sarpy, rasterio).

## `fastsar.quality` and `fastsar.sim`

- `quality.point_target(img, ij, d_az=1.0, d_rg=1.0, half=32, up=16)`: resolution, PSLR, ISLR and peak of a point
  target.
- `quality.upsample(patch, up=16)`: band-limited upsampling.
- `sim.make_collect(...)`, `sim.simulate(col, pos, amp)`: a simulated spotlight collection for tests (needs
  finufft).
