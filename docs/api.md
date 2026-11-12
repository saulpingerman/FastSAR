# API reference

`...` marks keywords the docstrings describe. Positions are in meters in `io.read_cphd`'s local frame
([real-data.md](real-data.md#reading-cphd)). Invalid inputs raise `ValueError` or `TypeError`.

## `fastsar`

- `process_cphd(cphd, output=None, *, product='sicd', area='tasked', plane=None, autofocus=None, weighting=None,
  remap='nrl', remap_args=None, spacing=None, extent=None, sicd=None, mode='auto', backend='auto',
  precision='float32',
  shadows_down=True, nesz_db=None, scene_sigma0_db=-10.0, height=None, channel=0, troposphere=None,
  reader_options=None, target_db=-40.0, grid=None)`: from a CPHD file to a SICD or SIDD
  ([product-options.md](product-options.md)). `grid`: as `form_cphd`'s; the quality areas, the slant plane and
  apodization are then refused ([stack.md](stack.md#the-common-grid)). Returns the product dict (`form_cphd`'s entries on the product's
  grid) with `product`, `area`, `weighting`, `autofocus`, `autofocus_info`, `timing` (seconds per stage) and, when
  written, `path` and `metadata`.
- `process_stack(cphds, output=None, *, product='mean', grid=None, reference=0, spacing=None, extent=None,
  register='fourier', equalize='mean', weights=None, window=7, weighting='taylor', aperture='auto', dem=None,
  remap='nrl', remap_args=None, shadows_down=True, backend='auto', precision='float32', max_shift=64,
  range_gate=True, channel=0, troposphere=None, reader_options=None, target_db=-40.0)`: several collections of
  one scene to one product ([stack.md](stack.md)). `product`: `'mean'` (a SIDD), or of two collections as a
  coherent pair `'coherence'`, `'change'`, `'interferogram'` (GeoTIFFs) or `'coherent_sum'` (a SIDD); the pair is
  formed from its common support ([stack.md](stack.md#coherent-pairs)). Returns the product dict of the
  `stack` function of that name with `registrations`, `inputs`, `timing`, for a pair `support` and `baseline`,
  and, when written, `path` and `metadata`.
- `form_cphd(cphd, sicd=None, *, mode='auto', backend='auto', window=True, spacing=None, channel=0, patch=1024,
  azimuth_fraction=0.8, extent=None, height=None, precision='float32', target_db=-40.0, info=None,
  autofocus=False, troposphere=None, reader_options=None, axes='track', aperture='auto', band=None, grid=None,
  shift=None, range_gate='auto')`: the image of a CPHD file ([processing-chain.md](processing-chain.md#2-form)).
  `mode`: `'auto'`, `'spotlight'`, `'moving'`. `troposphere`: passed to `read_cphd` (default: the delay the file
  gives is removed; `'model'` removes a standard-atmosphere delay for files that give none). `sicd`: path, `.xml` or SICD metadata. `spacing`, `extent`: (along, across
  track) in m. `height`: grid height above the ellipsoid. Returns a dict: `image` [nx, ny] complex64, `origin`
  (pixel (0, 0)), `e1`, `e2`, `spx`, `spy`, `mode`, `band` (first and last frequency), `bandwidth` (spatial
  frequency support along e1, e2), `window`, `phase_error` (or None), `meta` (from `read_cphd`), `notes`, and the
  grid's geometry: `plane`, `axes`, `center`, `height`, `graze`, `pulses`. `axes`: `'track'`, `'los'` (a spotlight:
  e2 along the line of sight on the ground) or `'auto'`. `aperture`: `'auto'`, `'square'`, `'full'` or the pulses
  (first, last + 1). `band`: the frequencies to use, (first, last) in Hz or one pair per pulse, weighted over
  that band; with `aperture`, the selection a coherent pair is formed from (`stack.common_support`). `grid`: an Earth-fixed ground grid (`stack.ground_grid`, or an earlier product: its pixels) in
  place of the collection's own; the output then has `grid`, `shift`, `source`, `form_options` and `axes` `'grid'`.
  `shift`: (d1, d2) in m along the grid axes, the grid displaced by it ([stack.md](stack.md#registration)).
  `range_gate`: `ImageFormer`'s.
- `form_image(S, ant, fmin, df, nx, ny, spx, spy, e1, e2, *, algorithm='ffbp', backend='auto', precision='float32',
  window=True, T='auto', ..., target_db=-40.0, ref=None, interp=None, upsample=None, center=None)`: a spotlight
  image on a grid centered on the frame origin ([algorithms.md](algorithms.md)). `algorithm`: `'ffbp'`, `'pfa'` or
  `'bp'` (exact backprojection, `ExactFormer`). `ref` [P]: the one-way range each pulse's samples are referenced to
  when it is not `|ant|` (a bistatic half path, a vendor's reference point). `precision`, `T`, `levels`, `pmax` and
  `target_db` apply to `'ffbp'`, `pfa_guard` and `pfa_support` to `'pfa'`; `interp` (`'cubic'` or `'linear'`), `upsample` and
  `center` (the point the grid is centered on) apply to `'bp'` only and raise `ValueError` with the others.
- `ImageFormer(ant, fmin, df, K, nx, ny, spx, spy, e1, e2, *, backend='auto', precision='float32', ...,
  aperture_weight=None, ref=None, range_gate='auto')`: plans and compiles once; `former(S)` forms each image.
  Arguments after `e2` are keyword-only. Attributes `T`, `predicted_error_db`, `range_gate` (None, or what the
  gate kept). `range_gate`: the history gated in range to the image's swath before formation: `'auto'` where the
  cuda backend's first level could not hold its filter otherwise, `True` wherever the gate keeps under half of
  the samples, `False` never ([stack.md](stack.md#the-range-gate)); `form_image` takes it too. `aperture_weight(points, pulses) -> W [n, m]` weights pulses
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
  points use `backproject`. One former is not safe to call from two threads at once. `backend='metal'` runs the
  kernel on an Apple GPU (`fastsar.metal`; `'auto'` picks it on a Mac), the tile centers' float64 terms computed
  by numpy per chunk since Apple GPUs have no float64.
- `MemoryWarning`: a `UserWarning` subclass issued when a former falls back to a slower path for lack of memory
  ([performance.md](performance.md#memory)). A TPU that cannot hold the history, or a GPU that cannot form one
  first-level child, raises `MemoryError`.
- `backproject(S, ant, fmin, df, points, *, rcv=None, ref=None, backend='auto', upsample=8, window=True, chunk=256)`:
  exact backprojection at points [..., 3]; `rcv` for bistatic, `ref` for per-pulse reference ranges.
- `plane_points(nx, ny, spx, spy, e1, e2, height=None)`: the `form_image` grid as points [nx, ny, 3].
  `available_backends()`: backends this machine runs, in `'auto'` order.

Backends: `'cpu'`, `'cuda'`, `'tpu'`, `'jax'`, `'metal'` (`ExactFormer` on Apple GPUs; the factorized former runs on
the CPU), `'auto'`. Precision: `'float32'`, `'float16'` (CUDA, JAX),
`'single-pass'`, `'three-pass'` (TPU) ([precision.md](precision.md)).

## `fastsar.io`

- `read_cphd(cphd, sicd=None, channel=0, meta=False, regrid_tol=1e-3, drop_flagged=False, troposphere=None,
  phase_sign=None)`: one channel as `dict(S, ant, fmin, df)`, plus `nx, ny, spx, spy, e1, e2` with a SICD.
  Reads CPHD 1.0.x, 1.1.0 and 0.3 in the frequency domain. `channel`: index, identifier or polarization (0.3 files
  have no channel identifiers, so their channels are identified by the index as a string, `'0'`, `'1'`, ...; their
  polarizations are taken from `RadarCollection.RcvChannels` only when it lists one per channel, or for a single
  channel from a `Polarization` or `Polarisation` parameter of `CollectionInfo`). `meta=True` also returns `tx`, `rcv`, `ref`, `R`, `origin`, `srp`, `tx_time`,
  `rcv_time`, `pulses`, `polarization`, `channel`, `mode`, `start`, `collector`, `core_name`, `cphd_version`,
  `notes` (which for a 0.3 file begin with `'CPHD 0.3 file'`). In a 0.3 file the TropoSRP field, whose units 0.3
  does not define, is reported in `notes` and not applied (`troposphere='model'` applies the modeled delay, and
  `troposphere=True` is refused); a 0.3 file whose frequencies are offsets from a reference frequency it does not give
  (`RefFreqIndex` set) is refused.
- `read_nisar(path, frequency=None, polarization=None, meta=False, height=None, band_margin=1.0, block=1024, range_delay=None, troposphere='model', orbit=None, reference_range=None)`: a
  NISAR L0B RRSD granule as `read_cphd` returns a CPHD (range compressed, frequency domain, a moving scene
  reference point at mid swath); `meta` adds `image_area` (the swath's corners). `read_collection(path, **kw)`
  dispatches on the path: `.h5` to NISAR, Sentinel-1 product names (`S1...`, raw or `.SAFE`) to `read_sentinel1`,
  zips, directories and `IMG-`/`LED-` files to `read_palsar`, anything else to `read_cphd`. Needs h5py (`fastsar[raw]`).
- `nisar_product(path)`: the orbit (time, position, velocity, epoch, type) and the common delays per frequency of an
  RSLC, GSLC or GCOV product, for `read_nisar(orbit=, range_delay=)`.
- `read_palsar(path, polarization=None, meta=False, height=None, band_margin=1.0, block=512, troposphere='model', reference_range=None)`: an ALOS PALSAR
  level 1.0 product (CEOS directory, zip, or `IMG-` file beside its `LED-` leader) as `read_cphd` returns a CPHD;
  `read_collection` dispatches directories, zips and `IMG-`/`LED-` files to it.
- `read_sentinel1(path, polarization=None, meta=False, height=None, band_margin=1.0, block=512, pulses=None, swath=None, internal_delay=None, troposphere='model', reference_range=None, orbit=None)`:
  a Sentinel-1 Level-0 product (.SAFE directory, zip, or one measurement .dat file) as `read_cphd` returns a CPHD;
  `pulses=(first, last)` reads a part. `fastsar.sentinel1` holds the packet parser, the user-data decoder and the
  orbit assembly. `orbit=` takes an ESA orbit file (`sentinel1.orbit_from_file`).
- `troposphere_delay(tx, rcv, srp)`: two-way delay [P] (s) of a standard atmosphere at the scene reference point
  (Saastamoinen hydrostatic zenith delay, cosecant mapping); what `read_cphd(troposphere='model')` removes.
- `sicd_points(sicd, rows, cols, meta, hae=None)`: local positions of SICD pixels on any grid type.
- `read_sicd(path, native=False)`: the pixels (a memory map, or an array in memory in the machine's byte order)
  and the metadata of a SICD file. `native=True` gives complex64 for every pixel type. A file that cannot be read raises a
  `ValueError` that starts with its path, as `read_cphd` does.
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

`sicd_meta` and `write_sicd` return the SICD metadata as a light object over the SICD XML (`fastsar._sicdxml.SICD`):
attribute access by the XML's element names (`sm.Grid.Row.SS`, `sm.ImageData.NumRows`, `sm.SCPCOA.GrazeAng`),
`get_array()` on vectors and polynomials (`sm.GeoData.SCP.ECF.get_array()`), `None` for an absent element,
`to_xml_bytes()`, and the SICD's projection (`project_image_to_ground`, `project_ground_to_image`). SICD, SIDD and
CPHD files are read and written by FastSAR itself; GeoTIFFs, DEMs and map projections other than `EPSG:4326` need
rasterio.

## `fastsar.stack`

Several collections of one scene ([stack.md](stack.md)). The functions work where the images are (NumPy, CuPy,
JAX) and return arrays of the same kind.

- `ground_grid(center, extent, spacing, azimuth=0.0)`: an Earth-fixed ground grid, horizontal at `center`
  (latitude, longitude, height): `dict(origin, e1, e2, spx, spy, shape, center, lat, lon, height, azimuth)`, ECEF.
  `as_grid(grid)`: the same from a dict under other names or from a product.
- `register(products, reference=0, *, apply='fourier', max_shift=64, tiles=4, box=5, highpass=32.0,
  coherent=False, form=None)`: one translation per image against the reference; each product gets `registration`
  (`shift`, `shift_px`, `significance`, `correlation`, `found`, `residual_rms`, `residual_median`,
  `residual_worst`, `tile_residuals`, `tiles_used`) and `shift`. `apply`: `'fourier'`, `'reform'` or `None`.
  `coherent`: the translation refined on the complex images; `registration` then has `coherent` (`found`,
  `delta_px`, `tiles_used`, `quality`, `residual_px`, `gain`) and `detected`, `detected_px`.
  `shift_image(out, shift, *, inplace=False)`: a product translated on its grid by a phase ramp on its spectrum.
- `mean(products, *, equalize=None, weights=None, tile=32)`: the incoherent average; `image` (amplitude), `looks`,
  `looks_single`, `looks_by_n`, `looks_ideal`, `looks_independent`, `brightness_spread_db`, `gains`, `inputs`.
- `coherence(a, b, window=7, *, common_band=True, fringe='tiles', tile=384, floor=True)`: `image` (the coherence),
  `median`, `mean`, `p90`, `above_0p5`, `above_0p7`, `floor` (the same of the displaced pair), `samples`,
  `overlap`, `fringes`, `fringes_found`, `fringe`, `baseline`. `change(a, b, window=7, *, capable=0.9, lost=0.4,
  ...)`: `image` (the intensity ratio, dB), `coherence`, `allowed` (the coherence the two brightnesses allow),
  `capable_fraction`, `changed_fraction`, `coherent_fraction`, `ratio_sigma`.
- `interferogram(a, b, window=7, *, fringe=True, dem=None, dem_step=4, floor=True)`: of a registered pair formed
  on its common support: `image` (the phase, rad), `coherence`, `fringe`, `fringe_per_km`, `fringe_found`,
  `phase_offset`, `height_of_ambiguity`, `height_per_radian`, `displacement_per_radian`, `wavelength`,
  `topography`, `phase_std`, `flatness`, and the coherence's statistics and `floor`.
  `coherent_sum(a, b, window=15, *, equalize=True, coherence_window=7, dark=0.01)`: `image` (complex), `gain`
  (`signal`, `noise`, `snr`, `second`, `expected`, `incoherent`, `coherence`), `amplitude_factor`, `fringe`.
- `baseline(a, b, point=None, aperture='square')`: from two CPHD files or products, `inside`, `overlap_range`,
  `overlap_azimuth`, `overlap`, `azimuth_available`, `d_graze`, `d_azimuth`, `critical_graze`,
  `critical_azimuth`, `height_of_ambiguity`, `days`. `common_support(a, b, point=None)`: the selection of each
  collection that covers what both have, `pulses` and `bands` (`form_cphd`'s `aperture=` and `band=`), with
  `phi`, `rho`, `azimuth_deg`, `azimuth_available_deg`, `azimuth_needed_deg`, `overlap_range`, `d_graze`,
  `critical_graze`, `height_of_ambiguity`, `resolution`.
  `geometry(source, channel=0)`: a collection's antenna positions, band and start without its samples.
- `write_geotiff(path, out, bands=None)`: a stack product on its grid as a GeoTIFF in its UTM zone.
- `process_stack`: as `fastsar.process_stack`.

## `fastsar.planes`, `fastsar.resample`, `fastsar.weighting`

- `planes.to_slant(out, spacing, bounds=None, *, margin=0, valid=None, taps=8, knots=32)`: a ground-plane image on
  axes along the line of sight resampled to the slant plane; `planes.to_ground(out, origin, e1, e2, spacing, shape,
  *, kernel=None, taps=8, pad=None, knots=32)` the reverse. `planes.deramp(out, sign=-1)` brings an image to
  baseband in place. `planes.slant_frame`, `ground_to_slant`, `slant_to_ground`: the slant plane and the closed-form
  maps. `planes.sampling`, `planes.support`: the spectrum of the image from the collection's geometry.
- `resample.lines(src, pos, m=None, *, kernel='sinc', taps=8, pad=None, out=None)` and `resample.warp(src, pos0,
  pos1, shape, ...)`: band-limited interpolation along one axis and two-pass warps, with positions given exactly or
  as a `resample.KnotMap`.
- `weighting.sva(img, oversample=2, *, axes=(1, 0), joint=True, inplace=False)`: spatially variant apodization.
  `weighting.keystone(S, ant, fmin, df, scp, e1, symmetric=True)`: the phase history trimmed to a rectangle of
  slant-plane spatial frequencies.

## `fastsar.rgiqe`, `fastsar.doctor`

- `rgiqe.Radiometry(cphd_path, meta, *, channel=0, pulses=None, nesz_db=None, sicd=None, sigma0_db=None,
  scene_sigma0_db=-10.0, eval_pulses=64)` with `gain_db`, `nesz_db`, `snr`, `rniirs`; `rgiqe.area_mask` and `rgiqe.area_bounds`;
  `rgiqe.rniirs`, `information_density`, `snr_to_rniirs`, `rniirs_to_snr`
  ([product-options.md](product-options.md#processed-area-by-image-quality)).
- `python -m fastsar.doctor [--install] [--json]`, or `doctor.check()`: what this machine can run and the commands
  that install what is missing.

## `fastsar.sidd`, `fastsar.remap`

- `sidd.write_sidd(path, out, *, remap='nrl', remap_args=None, valid=None, shadows_down=True, version=3,
  product_name=None, classification='UNCLASSIFIED', sicd=None, collections=None, product_class=None,
  product_type=None, extensions=None)` (`collections`: the SICD metadata of each input of a product made from
  several collections, one `ExploitationFeatures` `Collection` each), `sidd.to_display(out, remap='nrl', valid=None,
  shadows_down=True, **remap_args)`, `sidd.shadows_down(out)`, `sidd.orient(array, transform)`.
- `remap.remap(x, name='nrl', **kw)` and the named remaps `density`, `brighter`, `darker`, `highcontrast`, `pedf`,
  `nrl`, `log`, `linear`.

## `fastsar.autofocus`

- `image_autofocus(img, mode, **kw)`: `(img2, info)` for a baseband image whose axis 0 is azimuth; mode
  `'quadratic'`, `'pga'`, `'spatially_variant'`, `'spatially_variant_pga'` or `None`
  ([product-options.md](product-options.md#autofocus-on-the-image)). `quadratic(img, *, iterations=6, lines=0.3,
  tol=0.02, guard=True, ...)` and `spatially_variant(img, *, block=512, overlap=0.5, method='quadratic', order=2,
  guard=True, ...)` are the modes themselves.
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
