# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The date of a version is the day it
was published on PyPI.

## [0.3.0] - 2026-11-14

### Added

- Terrain-informed focusing (`fastsar.terrain`, `process_cphd(..., dem=, geoid=)`). A scatterer above or below the
  image plane is displaced (layover) and, on a curved track, defocused by a phase across the aperture that is
  proportional to its height and to the track's curvature out of the slant plane. `terrain.focus` computes that
  phase from the antenna positions and a DEM, with nothing estimated from the image, and removes it from the
  baseband ground image before the data-driven autofocus, on the device that holds the image (NumPy, CuPy, JAX):
  in strips along azimuth, each pixel corrected for the height it shows (interpolated between the corrections
  for a few heights that span each column's heights within the strip), and with the range walk, the part of the
  phase that scales with frequency over the band, removed through each row's spectrum. Against exact
  backprojection onto the DEM's surface (the rms of the complex difference over that of the exact chip) a Capella
  spotlight of Mount Etna, with 1,690 m of relief, gives -49 to -52 dB on chips 500 m above the plane, 300 to
  600 m below it within one chip, and 1,050 m below it (2.8 to 6 rad of phase), where the plane's image is within
  -51 to -52 dB of exact backprojection on the plane; a correction per column with the height averaged over
  strips and the phase at the center frequency alone gave -25 to -33 dB there. Over 72 chips drawn by phase and
  slope the median is -50 dB and the worst -45 dB (that correction: -35 and -25 dB). On a Capella spotlight of
  Kilauea the chips reach -53 to -55 dB at 1.1 to 2 rad (before -35 to -50 dB), and -46 dB in the collapsed
  caldera with a 2019 lidar surface (before -29 dB). The image stays on its plane,
  with its geolocation and its SICD projection; the product records the step in `out['terrain']`, a note,
  `timing['terrain']` and an `ImageFormation.Processing` element of the SICD. `dem` is a GeoTIFF (or a list of
  tiles), a function of latitude and longitude, or a number; a file needs `geoid`, the geoid's height above the
  ellipsoid at the scene (`terrain.as_dem`; tiles on one grid, as adjacent Copernicus tiles are, are joined into
  one raster so that heights interpolate across their seams, here and in `products.read_dem`, which takes a list
  of tiles too), and a DEM whose median over the image is off the scene height by more than its relief explains
  raises `terrain.TerrainWarning`. `terrain.heights` gives the surface each pixel shows through layover,
  `terrain.phase_error` and `terrain.coefficients` the exact model, and `terrain.surface_points` the points at
  which `fastsar.backproject` forms the image on the surface itself
  ([docs/product-options.md](docs/product-options.md#terrain-informed-focusing)). Spotlight collections only;
  results without a `dem` are unchanged. The correction is made on the ground image as formed, before any
  resampling, so it serves the default ground-plane SICD, the slant-plane SICD and the SIDD alike, and it leaves
  the scale factors of the `Radiometric` block as they are.
- `process_cphd` on moving-beam collections (stripmap, sliding spotlight) takes the weightings, autofocus and the
  quality areas that 0.2.0 refused there ([docs/product-options.md](docs/product-options.md#moving-beam-collections)):
  - `weighting='uniform'` (a rectangle over each pixel's processed azimuth band, no frequency window),
    `'taylor'` (the Taylor window along frequency and over that band) and `'sva'` (the uniform image at twice
    Nyquist of both bands, cut to the same rectangle at every pixel and apodized on the ground grid). Apodization
    is refused, with the reason, where the squint shears the range band by more than 10% of its width, the ground
    range band changes by more than 5% across the swath, or the azimuth band differs by more than 10% over the
    scene. The defaults are a spotlight's, `'uniform'` for a SICD and `'sva'` for a SIDD; a SIDD whose weighting
    was not given is Taylor weighted where apodization is refused, with a note that says why, and
    `weighting='sva'` raises a `ValueError` there.
  - `autofocus='quadratic'` and `'spatially_variant'`: map drift on blocks along the track no longer than half
    the footprint of the synthetic aperture, each under the guard of the spotlight modes, tied together by a
    robust local fit along the track and across the swath, and removed with a cross-faded correction
    (`moving.autofocus`). The guard's contrast test measures its scatter over cells along and across the track,
    as many as the corrected rows hold. The phase gradient modes are refused on a moving beam.
  - `area='tasked_quality'`, an RNIIRS and `'black'`: the tasked extent along the track, and in ground range as
    far as the criterion holds for the moving beam's own gain and receive windows. `'black'` ends at
    `black_sigma0_db` as for a spotlight, and the areas follow the pointing the data chose for the antenna
    pattern (below): where no pattern is trusted they are refused with a spotlight's reason.
- `fastsar.moving`: the baseband step of a moving-beam image, exact at every pixel on both axes for any squint or
  slide of the beam (`baseband`), the spectrum of the image from the collection (`spectrum`), `band_limit` and
  `autofocus`, on NumPy, CuPy and JAX arrays. On JAX's device everything is float32, complex64 or int32, as a TPU
  needs: the baseband phase is the sum of two vectors the host solves in double and a remainder of a few radians
  the device computes, to 2e-5 rad at 600 km, and every stage runs in blocks sized to the device's memory
  (`FASTSAR_JAX_BLOCK_MB`), with an image the device cannot hold kept on the host and passed through it.
- `form_cphd(..., azimuth_window=)`: the per-pixel azimuth window of a moving beam, `'hann'` (the default, as
  before), `'taylor'` or `'uniform'`; `patches.form_mosaic(..., amask=)` cuts a window at the edge of the band.
  The step this leaves at the edge is placed per final subaperture of the former, and every backend places it at
  the same pulses: under such a window the jax and tpu backends, which pad a patch's pulses and range gate to the
  sizes their compiled programs are shared at, lay the patch out as the cpu and cuda backends form it and apply
  the weight in their form, in one final stage instead of three (`ImageFormer(padded=)`,
  `ffbp2.make_plan(pulse=)`).
- Radiometric calibration in the SICD. `process_cphd` writes the `Radiometric` block of a spotlight SICD when the
  CPHD file supports it, and only then: `NoiseLevel` (ABSOLUTE) and `RCSSFPoly`, `SigmaZeroSFPoly`,
  `BetaZeroSFPoly` and `GammaZeroSFPoly`, by SICD's definitions. With `PTRef` and `PNRef` in the file both the
  signal and the noise level are calibrated; with `PTRef` alone (Umbra) the noise level is the noise power
  measured from the phase history, and the product's notes say so; without `PTRef` (Capella) there is no block,
  and the notes say why. An `'sva'` product and a moving-beam collection get none. The factors follow the antenna
  gain over the image as 2-D polynomials, since the image is not corrected for the antenna pattern, and the notes
  state their order and residual. `out['radiometric']` holds what was written;
  `rgiqe.calibration`, `rgiqe.reference_levels` and `rgiqe.support_area` are the pieces
  ([docs/product-options.md](docs/product-options.md#radiometric-calibration), with the simulator's accuracy
  and the comparison with a vendor's SICD). The radiometric model's time is the `radiometry` entry of `timing`
  (for the quality areas it was part of `form`).
- The antenna pattern's pointing is decided by the data (`Radiometry.pointing`, `Radiometry.footprint`). The
  beam's footprint is measured from the range-Doppler power of a few hundred pulses and compared with the modeled
  gain. The file's antenna frames are used where they pass (the fitted beam center within 0.4 of a half-power
  radius of the modeled one). Where they fail, the same pattern is tried with each pulse's boresight on that
  pulse's scene reference point, where a tracking beam is steered, and used if it passes. Where neither passes
  the model has no trusted pattern: the quality areas other than `'tasked'` raise a `ValueError` and the scale
  factors of the `Radiometric` block are constants. A scene that shows no footprint keeps the file's pointing,
  with a note that it is unverified. A moving beam is checked the same way, once, on the pulses of the middle of
  the strip: its echoes are referred to each pulse's own scene reference point, so the model takes the antenna's
  frame at the pulses it measured and the ground around their reference point (on a Capella stripmap the file's
  frames pass with the fitted center on the modeled one, at a correlation of 0.85). A beam whose center turns
  between the measured blocks of pulses (a sliding spotlight) shows no one footprint and is not checked: the
  file's pointing is kept, unverified. Of seven Umbra collections
  five fail as their files point them (0.65 to 1.00 radii; the frames put the boresight 1.4 to 1.6 km ahead of
  the scene reference point) and pass on the reference point (0.12 to 0.31), and two lie halfway and pass
  neither (0.44 to 0.55); on one of the five the black area
  moves from 7.90 x 9.45 km to 7.75 x 9.05 km, centered on the reference point, and the tasked-quality area from
  7.65 x 6.75 km to 5.95 x 6.70 km. The tolerance was set against the brightness of the formed images, which
  follows the gain on the reference point at +0.70 to +0.80 dB per dB over ground of low texture for the five,
  and the gain as the file points it at +0.31 and +0.45 for the two
  ([docs/product-options.md](docs/product-options.md#the-antenna-patterns-pointing)).
- The beam's width is measured from the same footprint (`Radiometry.footprint['width']`): the measured width over
  that of the modeled two-way beam, along the antenna's X and Y axes, where the clutter stands 6 dB over the
  noise in the map. It is a record and does not change the model. A pattern applied once where the antenna
  applies it on transmit and on receive would read 1.41; the seven Umbra collections read 1.07 across the track
  (standard deviation 0.07) and 1.05 along it (0.05).
- The clutter factor of the `Radiometric` block uses the samples' own density in spatial frequency
  (`rgiqe.clutter_gain`): pulses spaced unevenly in angle moved a simulated sigma zero by up to 0.1 dB under a
  Taylor window.
- `sim.write_cphd(..., pt_ref=, pn_ref=, beam=)` writes the reference levels and an antenna block, and
  `sim.beam_gain` gives the matching gain of simulated scatterers.
- Several collections of one scene ([docs/stack.md](docs/stack.md)). `process_stack(cphds, output, product=)`
  forms each collection on one Earth-fixed ground grid, registers them and writes the incoherent average as a
  SIDD (`'mean'`), or the coherence or the change of a pair as a GeoTIFF. The collections are read and formed one
  at a time, and the stack stages run where the images are (all cores, CuPy, JAX). A stack product carries no
  radiometric calibration, and its `dem` is the surface of an interferogram: terrain-informed focusing works on
  one collection's own axes and is not applied to a stack.
- `form_cphd(..., grid=, shift=)` and `process_cphd(..., grid=)`: a collection formed on an Earth-fixed ground
  grid (`stack.ground_grid`, or an earlier product for its pixels) in place of its own, and on that grid
  displaced by a translation. Products record the grid. `process_cphd` on a grid keeps its defaults where they
  apply: the SICD is unweighted, and the SIDD, whose apodization a grid rules out, is Taylor weighted with a note
  (`weighting='sva'` by name, the quality areas, the slant plane and a `dem` are refused). A SICD on a grid
  carries no `Radiometric` block. A moving beam on a grid along its track is processed as on its own grid; on
  another grid the product is the image as formed there, with a note.
- `fastsar.stack`: `ground_grid` and `as_grid`; `register` (one translation per image from the detected images'
  cross-correlation, with its significance and the residual shifts of tiles) and `shift_image` (the translation
  applied by a phase ramp on the baseband spectrum); `mean` (with the equivalent number of looks measured on
  homogeneous tiles and optional brightness equalization); `coherence` (on the common band of the two spectra,
  with a linear fringe removed and the estimator's floor measured on the pair) and `change`; `baseline` (whether
  a pair is inside the critical baseline, from the per-pulse parameters alone).
- Coherent pairs ([docs/stack.md](docs/stack.md#coherent-pairs)). `stack.common_support(a, b)` gives, from two
  collections' geometry, the pulses and the frequencies of each that cover the part of the ground spectrum both
  have, and `form_cphd(band=)` takes the frequencies (one pair, or one per pulse) beside `aperture=`: the samples
  outside get no weight and the window spans the selection, along the aperture as a function of the look
  direction, so two collections cut to one support carry one weighting. `process_stack` forms the two
  collections of a pair product that way. With apertures displaced by 30% the coherence of stable surfaces was
  0.56 untrimmed, 0.63 with the images cut to their common band and 0.81 formed on the common support.
- `stack.interferogram` (the phase and coherence of a pair, one linear fringe and an optional surface's
  topographic phase removed, with the height of ambiguity), `stack.coherent_sum` (with its gain as measured) and
  `process_stack(product='interferogram' | 'coherent_sum')`. `stack.change` also returns the coherence the two
  brightnesses allow and the fraction of the pixels that can show change which do.
  `stack.register(coherent=True)` refines a translation on the complex images. `stack.baseline` reports the
  overlap in range at equal look direction, how much of a square aperture the two whole apertures share, and
  the height of ambiguity.
- `stack.coherence(fringe='tiles')` gives a tile its own fringe only where the peak of the tile's interferogram
  spectrum stands clear of noise (`fringes_found`); other tiles take the image's.
- `write_sidd(..., collections=, product_class=, product_type=, extensions=)`: a SIDD made from several
  collections holds one `ExploitationFeatures` `Collection` per input.
- `ImageFormer(..., range_gate=)`, also on `form_image` and `form_cphd`: the phase history gated in range to the
  image's swath before formation. On the cuda backend the gate sizes its blocks to the device memory that is free
  (`fastsar._xp.cuda_budget`), and where other work has filled the device it runs on the host on all cores, with
  a note in the product. `form_cphd(aperture=(first, last + 1))`: the pulses to use.

### Changed

- README and docs carry the 0.3.0 release records, measured at the tag on the six devices of the 0.2.0 records.
  The images of the three Umbra collections and the five Capella collections are bitwise those of 0.2.0, and the
  median change in time is -1.3 to +0.6% per device; the largest single changes are stated in
  [docs/performance.md](docs/performance.md).
- The defaults of `process_cphd` (a change of behavior from 0.2.0). The default SICD is now on the ground plane
  and unweighted: `plane=None` is `'ground'` for every product, and `weighting=None` is `'uniform'` for a SICD
  (`'sva'` for a SIDD, as before). 0.2.0 wrote a slant-plane SICD with Taylor weighting, and
  `process_cphd(cphd, output, plane='slant', weighting='taylor')` reproduces it. The ground plane is the plane
  FastSAR forms on, so the default product is the image as formed (`Grid.Type` PLANE, `Grid.ImagePlane` GROUND,
  rows along ground range away from the radar, the view from above); most producers deliver slant-plane SICDs,
  and [docs/product-options.md](docs/product-options.md#the-ground-plane-sicd-the-default) lists what a consumer
  that assumes one should check. No weighting leaves the choice of an apodization to the user of the complex
  image. Since apodization runs on the slant plane, a SICD with `weighting='sva'` and no `plane` is a slant-plane
  SICD as before, with a note in the product that says so; `plane='ground'` with `'sva'` raises a `ValueError`.
  The defaults of a SIDD and of `form_cphd` are unchanged. A moving-beam collection takes the same defaults (the
  next entry).
- The noise measurement of `rgiqe.Radiometry` is bounded and no longer reads the file a second time from
  `process_cphd`: about 8 million samples at most (64 pulses of an Umbra collection with 120,736 samples per
  pulse, 0.03 dB from the level of 512), taken from the phase history in memory (`Radiometry(..., history=)`). On
  that collection the whole radiometric model, with the check of the antenna block, takes 0.55 to 1.2 s on 16
  cores where that of 0.2.0 took 0.8 to 3.6 s, and it no longer slows by orders of magnitude on a host whose
  cores are oversubscribed.
- `area='black'` ends where a scene of sigma zero -10 dB has a signal-to-noise ratio of 1, ambiguities counted as
  noise: where the noise-equivalent sigma zero plus the ambiguity level reaches -10 dB. In 0.2.0 the reference was
  the RGIQE's scene (0 dB for a co-polarized channel), which put the edge about 10 dB further down the beam, past
  where most terrain shows anything. `process_cphd`, `rgiqe.area_mask` and `rgiqe.area_bounds` take
  `black_sigma0_db=-10.0`, and `black_sigma0_db=0.0` gives the area of 0.2.0 for a co-polarized channel. The
  RNIIRS areas (`'tasked_quality'`, a number) keep the RGIQE's reference, so that an RNIIRS computed here is the
  one sarpy's `PREDICTED_RNIIRS` states; `'black'` therefore no longer contains every RNIIRS area, only those whose
  threshold needs an SNR of at least 10 dB of the reference scene. `Radiometry.snr(..., sigma0_db=)` gives the
  ratio for a scene of any sigma zero. The areas of a moving beam, new in this version, use the same reference.
- `process_cphd` on a moving beam (a change of behavior from 0.2.0, which wrote `form_cphd`'s image: Taylor
  weighting along frequency, a Hann window over the azimuth band, no autofocus). The defaults are now those of a
  spotlight: the SICD is unweighted (`'uniform'`: a rectangle over each pixel's azimuth band in place of every
  pulse of a patch, and no frequency window), the SIDD is apodized (`'sva'`, or Taylor weighted with a note
  where apodization is refused), and autofocus is `'quadratic'`. `weighting='taylor'` weights the azimuth band
  with the Taylor window in place of the Hann window (a main lobe 18% narrower, a peak sidelobe 3.5 dB lower).
  The product is at baseband (0.2.0 gave the image as formed), and its SICD states a band centered at zero. The
  SICD of 0.2.0 is `products.write_sicd(path, form_cphd(cphd))`; the nearest product of `process_cphd` is
  `weighting='taylor', autofocus=False`. The default spacing along the track follows the azimuth band of
  the image, which is wider than the beam's Doppler band over the platform's speed wherever the footprint moves
  slower than the platform (1.10 times on a Capella stripmap, 5.8 times on a sliding spotlight; 0.2.0 sampled
  both for the narrower band), and the `bandwidth` entry of the product states that band. An autofocus that was
  applied is recorded in the SICD as `AzAutofocus` SV. A grid smaller than one patch is formed as a single patch
  of its own size. `form_cphd` itself gives the image of 0.2.0 for the same arguments.

### Fixed

- `process_cphd` stated, and sampled for, an azimuth band the image does not have on a sliding spotlight whose
  scene reference points (SRPs) move slower than its beam. On a Capella C10 sliding spotlight (2024-04-05) the
  default SICD of a 1 by 3 km cut stated 11.29 cycles/m and was sampled at 0.083 m. Its SRPs move 612 m/s along
  the track at the scene center, and the beam, as the file's antenna block points it, 1,061 m/s: the processed
  spread about the SRPs took each pixel through pulses up to 0.0149 in sin(look) from the boresight, beyond the
  0.0111 the PRF samples, which hold no echo of it. The image showed it: the -3, -6 and -10 dB widths of its
  azimuth spectrum were 3.35, 5.2 and 7.0 cycles/m, against 3.8, 5.3 and 6.9 predicted from the boresight's
  motion and the file's pattern and 6.0, 8.5 and 11.0 from the SRPs'. Where the SRPs' spread takes the tasked
  area beyond the band the PRF samples about the antenna's two-way boresight, the boresight is now the beam
  center (`moving.antenna_sine`; a note beginning `beam center:`), and the product states the band the beam
  gives, sampled at 0.96 of its reciprocal as before: 7.76 cycles/m at 0.120 m on that cut (8,324 rows for
  12,024). Its spectrum fills that band and ends at the edge (98% of the energy within 6.9 cycles/m, 0.06%
  beyond 7.76, against 4.4% before), and the formation takes 14.5 s on the GPU for 21.7 s. The vendor's SICD
  processes 2.5 cycles/m of it. The C09 sliding spotlight and stripmap and the C14 sliding spotlight of the
  documentation keep their SRPs and are the same to the bit; `form_cphd` keeps the SRPs. The brightest returns
  of the C10 scene are specular (each lit over 0.3 to 1.2 cycles/m of the band) and are 0.5 to 5.4 m long along
  the track with either beam center (docs/product-options.md, Real collections).
- `process_cphd(..., sicd=)` on a moving beam took the vendor SICD's azimuth band (`Grid.Col.ImpRespBW`) as the
  spread of a pulse about its SRP, which a footprint moving at a fraction of the platform's speed widens at a
  pixel by the inverse of that fraction: 11.29 cycles/m stated and formed for the vendor's 2.50 on the Capella
  C10 sliding spotlight (sampled at 0.083 m for the vendor's 0.399 m, through pulses the beam does not light),
  6.98 for 1.23 on the C09 sliding spotlight of 2024-08-29, and 1.37 for 0.77 on a C09 stripmap of 2024-09-23
  whose SRPs also lag its beam. The product now has the vendor's band: ImpRespBW wide at each pixel about the
  pixel's center of aperture (`Grid.TimeCOAPoly`, seen from `Position.ARPPoly`), which on both sliding
  spotlights moves with the antenna's footprint (1,075 and 1,067 m/s), not with the SRPs (612 and 1,290 m/s). A
  SICD whose center of aperture does not move along the track takes the beam center of the product without a
  SICD. On the 1 by 3 km cuts the products state 2.505, 1.228 and 0.767 cycles/m, and their azimuth spectra fill
  that band and end at its edge (-6 and -10 dB widths 2.49 and 2.52, 1.23 and 1.24, 0.76 and 0.77; 0.3 to 0.4%
  of the energy beyond it). The vendor's spacing is kept where it samples the band at the top of the frequency
  band (the stripmap's 1.04 m) and replaced by 0.96 of the reciprocal of that band where it does not (0.371 m
  for 0.399, 0.758 for 0.814). The C09 sliding spotlight's brightest points are 0.95 m long along the track
  (the vendor's image: 0.87 m), and 20 stand 25 dB over their surround. A note beginning `azimuth band:` gives
  the band and its center, and another says where the band reaches pulses the beam does not light.
- Forming a patch much longer on one side than the other (832 by 96 pixels in tiles of 32: a narrow cut sampled
  finely) failed with "1 levels cannot split 832 pixels into tiles of 32": the levels were chosen for the short
  side. A patch now takes at least the levels its long side needs (`patches._levels`); other patches are formed
  as before.
- The cuda backend refused an image narrow in range from a wide-band collection ("the cuda backend's first-level
  window ... exceeds the GPU's shared memory", for example a 2 km image of a collection with 76,032 samples per
  pulse). Such a history is now gated in range to the image's swath first, on the device where it has room and
  on the host otherwise, with a note in the product; the image equals that of the whole history to -63 dB
  (measured against the cpu backend).
- Writing a NITF file could fail with "BufferError: cannot close exported pointers exist" on a heavily loaded
  host, when a worker thread still held its last block of the file's memory map as the map was closed. The
  workers now reach the map through a holder that is emptied before the map is closed.
- `fastsar._xp.device(a, 'cuda')` returned before a page-locked host array (the image a cuda former returns) had
  been read, so that writing the host array afterwards changed the device copy. No path of 0.2.0 wrote it. The
  copy is now complete when the call returns.
- The CPHD reader refused a file whose added per-vector parameter states its Size in bytes where the standard
  counts 8-byte words (a Capella spotlight of 2023: RefTime, F8, Size 8 at the end of a 280-byte vector). Where the
  Size equals its format's bytes and fits only as bytes, it is read as the format's words, with a warning.
- The cpu backend's first level in pulse blocks (the path it takes when memory is short) raised a shape error when
  a block of output rows lay wholly past the last pulse (the pulse filter's halo), as on a long sliding spotlight
  under a memory limit. Such a block reads zeros.
- `stack.coherent_sum` read its noise on the darkest ground of the scene, which can be ground the two images share
  (a desert playa): it adds coherently, and the measured gain fell short of what the pair's coherence gives (1.18
  dB for 2.11 on a Capella pair of the Black Rock Desert). The noise is now read on dark ground that is
  incoherent, where the coherence over 31 x 31 pixels is at most twice that estimator's floor on the pair
  (`gain['noise_reference']`, `gain['floor31']`). On three seen pairs the noise gain is 2.97 to 2.99 dB (3.01 for
  pure noise), and on a desert pair the signal-to-noise gain moved from 1.89 to 2.29 dB, against 2.25 expected.
  The gain the sum is expected to reach (`gain['expected']`) now counts the two images' noise as measured on that
  ground: 10 log10(2 (1 + g) / (1 + r)), r the second image's noise as added over the first's (`gain['noise_ratio']`),
  which is 10 log10(1 + g) when they are equal (`gain['expected_equal_noise']`). On a Capella pair of the Black Rock
  Desert whose second image is 1.2 dB noisier, the measured gain is 1.29 dB and the expectation 1.46 dB (2.11 dB for
  equal noise).
- The cuda backend could run out of device memory after its range gate (new in this version) on a long
  wide-band collection: the gated history stayed on the device and the former copied it into its planes beside
  it (a Capella sliding spotlight of 150,162 pulses and a 1,024-pixel image on a 24 GB L4). The gate's transform
  plans are released, and a gated history whose planes the former would stream from the host is given to it from
  the host; the image is the same to the bit.
- The cuda backend at `precision='float16'` could run out of device memory at the very end of a formation that
  fit: the image was scaled back from the half range into a new array while the padded image and an uploaded
  phase history were still on the device (a Capella spotlight of 94,513 pulses by 38,400 samples on a 24 GB L4).
  The image is now scaled in place; the pixels are unchanged.
- Under a memory limit (a container, a systemd scope, a batch job) the memory FastSAR took as available was the
  limit less the group's usage with its page cache counted, so that a collection just read from disk counted
  twice and `form_cphd` could refuse, with "... GiB is available", an image that fit. The group's file cache,
  which the kernel gives back before it enforces the limit, is no longer counted (`memory.host_available`).
- The autofocus of a moving-beam product without weighting (the default SICD of 0.3.0) could miss a large error:
  on a Capella sliding spotlight it found an injected quadratic error of 5 rad and not one of 15 rad, which the
  Taylor-weighted product of the same collection found. `moving.autofocus` now estimates on a Taylor-weighted
  copy of the columns it reads (along the track and across range) and corrects the image as it was delivered,
  and its guard counts a block only when the iteration of its estimate has settled (`taper=`, `range_band=`,
  `info['settled']`, `info['capture']`).
- `moving.autofocus` declined a moving beam whose brightest return is not a point. On a Capella C14 sliding
  spotlight (2025-02-19) seven of the eight sets of columns of both blocks agreed on 2.2 rad, and the eighth,
  which holds a specular return 51 dB over the median pixel (86 to 92% of its band in one quarter, one look 8 to
  54 times as bright as the other), read -6.9 rad and carried the estimate of all columns to -6.5 rad. The
  guard's first test compared that estimate with the sets' median and declined every block, as the collection
  came and with 2, 5 and 15 rad injected. Where the estimate of all columns lies off the median and the sets
  far from it are no more than a quarter, the estimate is now made again from the columns of the sets that
  agree, and the sets again from it, under the same test (`info['pooled']`). The injected errors are found to
  within 0.08 rad and the points return to 1.001 to 1.004 times the delivered product's width; the second test,
  the guard's promise of no harm, is unchanged, and blocks that passed the first test are untouched (the C09
  sliding spotlight and stripmap of the documentation give the same products to the bit). Relief is not the
  cause: the GLO-30 surface under the cut varies by +-7 m, +-0.2 rad.
- `docs/products.md` said that two passes formed onto one grid are coregistered by construction; that holds for
  the geometry as known, and real repeat collections lay 9 to 22 m apart. `products.coherence` gave its bias for
  independent pixels; the note now states the measured floor of oversampled images.

### Known issues

- Polar format on two Umbra spotlight collections of 2023 (UMBRA-04 and UMBRA-05 over Myra, North Dakota) differs
  from exact backprojection by -24.9 and -24.4 dB, where most collections checked give -26 to -33 dB. 0.2.0 gives the
  same images.
- A moving-beam collection can carry its own quadratic phase error (about 3.1 rad on a Capella sliding spotlight
  of 2024 near Tampa). The default product removes it with autofocus; a product made with `autofocus=False` keeps
  it, and its points are then wider than the stated band gives (1.7 reciprocal bands where the antenna pattern
  alone gives 1.04).
- Range widths follow the range spectrum the collection holds: on a Capella stripmap of 2024 whose spectrum falls
  by 2.3 dB over the outer 15 per cent of the band at each edge, they are 1.06 times those of a flat band, as
  that spectrum predicts.
- On a moving beam, quadratic autofocus finds errors of 2 to 15 rad on the collections checked, within 0.1 rad.
  Where an image's contrast rests on a few bright returns, an error that also moves them along the track (its
  linear part) leaves the corrected image's contrast a few per cent under the original's (0.953 at 15 rad on a
  Capella sliding spotlight of 2025), with the points' widths restored.
- Some scenes have no compact points: the brightest returns of a Capella sliding spotlight of 2024 over Yerevan
  are lit over 5 to 13 per cent of each pixel's aperture and are 0.5 to 8.5 m long along the track in any image
  formed from the whole aperture. Exact backprojection gives the same lengths.

## [0.2.0] - 2026-11-10

### Added

- `read_cphd(..., ref_freq=)` for CPHD 0.3 files with `RefFreqIndex` set: the reference frequency the file omits is
  added to `Fx0`, and `ref_freq=0.0` asserts that the producer wrote absolute frequencies despite the flag. Such files
  are still refused without it, and the refusal now reports the file's `Fx0` range. From `form_cphd`, pass
  `reader_options=dict(ref_freq=0.0)`.
- `process_cphd(cphd, output, product=, area=, plane=, autofocus=, weighting=, remap=)`: from a CPHD file to a SICD
  or a SIDD on disk, with the processed area, the image plane, the autofocus, the sidelobe weighting and the
  display remap as options ([docs/product-options.md](docs/product-options.md)). The defaults are the tasked area
  and a slant-plane SICD with Taylor weighting and quadratic autofocus; a SIDD defaults to the ground plane,
  spatially variant apodization and the NRL remap. `form_cphd` and its defaults are unchanged.
- Slant-plane products (`fastsar.planes`). A spotlight image is formed on the ground plane on axes along the line
  of sight (`form_cphd(axes='los')`), brought to baseband exactly (`planes.deramp`) and resampled to the slant
  plane of a SICD with `Grid.Type` PLANE (`planes.to_slant`), by two one-dimensional passes of band-limited
  interpolation (`fastsar.resample`). `planes.to_ground` goes the other way. `products.sicd_meta` and `write_sicd`
  take the slant-plane and baseband products.
- Processed area by image quality (`fastsar.rgiqe`): the radar generalized image quality equation on the
  collection's own antenna pattern and noise level gives the region at least as good as the tasked area, the
  region above a given RNIIRS, or everything with a signal-to-noise ratio of at least 1. Pixels of the rectangular
  product outside the area are 0.
- Autofocus on the image (`autofocus.image_autofocus`): a global quadratic phase error by map drift, phase
  gradient autofocus, and a spatially variant correction estimated on overlapping blocks. A correction is applied
  only when the contrast of the whole image rises and, for the two global modes, the parts of the image agree on
  the estimate; otherwise the image is left as formed and `autofocus_info['reason']` says why.
- Spatially variant apodization (`weighting.sva`) on the slant plane at twice Nyquist, with the trim of the phase
  history that makes the spectrum a rectangle (`weighting.keystone`).
- SIDD output (`sidd.write_sidd`, SIDD 3.0 or 2.0) and the display layout (`sidd.shadows_down`,
  `sidd.to_display`): range increases down the rows and the view is the one from above, for either look side.
- The remaps of the NGA MATLAB SAR Toolbox (`fastsar.remap`: `nrl`, `density`, `brighter`, `darker`,
  `highcontrast`, `pedf`, `log`, `linear`), checked against the toolbox run in Octave.
- The stages after formation (baseband, autofocus, resampling, apodization, remap) take the image as a NumPy,
  CuPy or JAX array and run where it lives, so a GPU or TPU run keeps the image on the device until it is written.
- On the cuda backend the stages after formation size their blocks from the free device memory
  (`FASTSAR_CUDA_BLOCK_MB` overrides it), and `process_cphd` keeps on the host the images of a product that the
  device cannot hold together, passing them through the GPU in blocks: the same pixels, with a note in the
  product (`memory.cuda_plan`; `docs/performance.md`, Memory).
- `python -m fastsar.doctor` (also the `fastsar-doctor` command): reports which backends this machine can run and
  prints the command that installs each missing piece, from the C++ compiler of the CPU kernels to the CuPy build
  for the GPU driver's CUDA version; `--install` runs the pip commands among them. It also reports the CPU kernels
  in use (a native build, or the prebuilt level), and says when a compiler would let the CPU backend use vector
  extensions the prebuilt level leaves out.
- Platform wheels with prebuilt CPU kernels for Linux x86-64, Linux 64-bit ARM and macOS on Apple silicon: the CPU
  backend runs on a machine with no C++ compiler. On x86-64 the kernels are carried at the levels x86-64-v2,
  x86-64-v3 (AVX2) and x86-64-v4 (AVX-512), and the best one the processor runs is loaded. The prebuilt objects
  need no OpenMP runtime: their loops run on threads kept between calls (`par.hpp`), and `OMP_NUM_THREADS` sets
  their number. Where a compiler is present the kernels are still compiled on first use for the exact processor,
  and a compiler that cannot build one (no OpenMP runtime, GCC before 12 on ARM) falls back to the prebuilt
  kernel. `FASTSAR_KERNELS=native` or `prebuilt` forces one of the two, with no fallback to the other
  ([docs/performance.md](docs/performance.md#cpu-kernels-native-and-prebuilt)). Other platforms install the source
  distribution and compile as before. `.github/workflows/wheels.yml` builds the wheels with cibuildwheel and tests
  each one in a container without a compiler and with one.
- The stages after formation on a TPU. `planes.to_slant` and `planes.to_ground` of a JAX image on a TPU read the
  samples a block of 16 or 32 output samples needs as one row of a table of the line's windows, where a TPU does
  an indexed load per tap at only about 1e8 a second (`fastsar/resample_jax.py`; `FASTSAR_RESAMPLE_JAX=blocks|gather`,
  `blocks` by default on a TPU, `gather` elsewhere); `remap.nrl` and `remap.percentile` find their order
  statistics from counts of the pixels below thresholds (`FASTSAR_REMAP_JAX=count|histogram`); a complex image
  leaves the device as its two real planes; every stage bounds its temporary arrays on the device; and
  `process_cphd` keeps on the host the images of a product that the device cannot hold (an image written there is
  there twice while its blocks are joined) and passes them through it in blocks, with a note in `out['notes']`
  (`FASTSAR_JAX_BLOCK_MB` overrides the device memory counted on, by default nine tenths of what is not in use).
  A complex image is handed to the per-block programs as its two real arrays: a TPU converts a complex argument
  at every call, which made the resampling's cost per pixel grow with the image. On a TPU v6e a 1,155 Mpx slant
  image at twice Nyquist (9.2 GB) is made in 9.8 s and apodized in 4.0 s with 22 GB of the 31 GB device in use at
  the peak. On a TPU v5e a 23 Mpx image goes to the
  slant plane in 0.57 s, a 49 Mpx slant image at twice Nyquist is made in 0.89 s and warped back to the ground
  with the cubic kernel in 0.24 s, within -90 dB of the CPU kernel's result, and a 196 Mpx slant image is
  apodized, detected and projected on the 16 GB device with 6.4 GB in use at the peak.
- `process_cphd(..., spacing=, extent=)`: the product's pixel spacing, and the tasked area's sides for a collection
  that states no image area.
- `form_cphd(..., axes='los' | 'auto', aperture='auto' | 'square' | 'full')`, and the grid's geometry in its
  output (`plane`, `axes`, `center`, `height`, `graze`, `pulses`).

### Changed

- README and docs carry the 0.2.0 release records, measured at the tag on the six devices of the 0.1.7 records.
  The images of the three Umbra collections are bitwise those of 0.1.7, and the times are within 2.5% of them but
  for three single figures ([docs/performance.md](docs/performance.md)).
- Installation. CuPy with the CUDA runtime libraries is a dependency on Linux x86-64, so that `pip install fastsar`
  alone gives the GPU backend; the `cuda` extra remains as a name for other platforms. An environment that already
  holds another CuPy build should install with `--no-deps`.
- FastSAR no longer uses sarpy at run time: CPHD files (1.0.1, 1.1.0, 0.3) are parsed, SICD and SIDD metadata built
  and projected, and the NITF files written and read by FastSAR's own code (`fastsar._cphdxml`, `_sicdxml`,
  `_siddxml`, `_nitf`), with only NumPy, SciPy and JAX installed. sarpy and sarkit remain in the `test` group as
  independent readers and consistency checkers of the files. The `io` extra is kept, empty.
  - `products.sicd_meta` and `write_sicd` return `fastsar._sicdxml.SICD`, and `sidd.write_sidd` returns
    `fastsar._siddxml.SIDD`, in place of sarpy's `SICDType` and `SIDDType`: attribute access by the XML's element
    names (`sm.Grid.Row.SS`, `sm.GeoData.SCP.ECF.get_array()`, `sm.SCPCOA.GrazeAng`), `None` for an absent
    element, `to_xml_bytes()`. Dates are their XML text (not `numpy.datetime64`), `sm.is_valid()`,
    `sm.derive()` and the other methods of the sarpy classes are gone, and the SIDD object is read-only. A sarpy
    `SICDType` is still accepted wherever a SICD is an argument (it is converted through its XML).
  - `write_sidd(classification=)` takes `'UNCLASSIFIED'` or a dict of the ISM markings where it took a sarpy
    `ProductClassificationType`.
  - The SCPCOA angles use the geodetic vertical at the scene center point, as the standard defines them; they
    differ from sarpy's by less than 1e-4 degrees.
  - A CPHD with a compressed signal block is refused (it was read through sarpy); `meta['tx_time']` and
    `meta['rcv_time']` are in the machine's byte order.
  - A SICD is written at 2.6 GB/s into the page cache where sarpy's writer ran at 0.6 GB/s (12000 x 12000 pixels,
    48 cores).
- `io.read_sicd(path)`: the pixels (a memory map) and metadata of a SICD file.
- `products.sicd_meta`: the fields of each grid direction follow SICD's rules among themselves, so that every SICD
  FastSAR writes passes sarkit's consistency check, on real collections as on simulated ones. DeltaK1 and DeltaK2
  are the band's center less and plus half of ImpRespBW, or the whole sampled band (-1 / (2 SS) to 1 / (2 SS)) for
  a band that wraps around the sampling rate; ImpRespBW is the band measured from the collection where the product
  carries it. Rows run along the grid axis nearer the line of sight (the first axis of a grid on track axes squinted
  past 45 degrees), so shadows fall down the rows. Where the first pulse precedes the collection start the file
  states, the SICD's start is moved to that pulse (its times cannot be negative).
- The readers of CPHD, NITF and SICD files check a file before they use it, and every file that cannot be read
  raises a `ValueError` that starts with the file's path and says what is wrong. Header offsets and sizes are checked
  against the file's length, the blocks' order and each other (the CPHD blocks against the arrays the XML describes,
  the section terminator after the XML block, a NITF's image segments against the SICD's NumRows and NumCols). XML
  that declares a document type or entities is refused. A SICD's root element is found however far into its segment
  it starts.
- `io.read_sicd(path, native=True)` converts AMP8I_PHS8I pixels to complex through the SICD's amplitude table.
- `read_cphd` reads the samples of a CPHD 1.x file from the file's own bytes on all cores, and conjugates and checks them on all cores. The samples agree with 0.1.8's to float32 rounding (2e-7 of the largest), the conversion, the check for
  empty and non-finite samples and the phase ramps along frequency being compiled passes over the rows. Reading
  an 8.7 GB Umbra collection takes 0.74 s where it took 5.9 s, and a 9.2 GB Capella collection 1.14 s where it
  took 13.0 s (48 cores, file in the page cache).
- `products.sicd_meta` arranges the SICD array as the view from above for either look side: rows along range away
  from the radar and row x column upward. For a collection whose grid axis e2 points away from the radar (a
  left-looking pass on track axes) the columns are now reversed with respect to 0.1.8, which wrote the mirror
  image there. `TStartProc` and `TEndProc` are those of the pulses used.

### Fixed

- Memory planning under a control group's limit (a container, a systemd scope, a batch job): the memory available
  was read from the host alone, so a run limited to less than the host's free memory planned for more than it
  could use and was killed. The limit, less the group's usage, now bounds it.
- The CUDA backend on an image narrow in range from a wide-band collection (a range decimation of 55 with a
  403-tap filter on an Umbra collection formed on axes along the line of sight): the first-level kernel's window
  did not fit the GPU's shared memory and the former raised a `ValueError`. The kernel now keeps one copy of the
  window there and reloads the raw samples.
- Polar format on the CPU from an installed wheel or source distribution. `pfa_cpu.cpp`, the C++ kernel of the
  final resampling added in 0.1.7, was not listed as package data, so 0.1.7 and 0.1.8 installed from PyPI raise
  `FileNotFoundError` on `form_image(..., algorithm='pfa')` with the CPU backend (a checkout or an editable install
  has the file, which is why the tests passed). The package data now takes every kernel source, and a test checks
  the list against the files the modules open. Factorized backprojection, the default, was not affected.

### Known issues

- The 2026 Capella spotlight collection named under 0.1.8 still gives the images it gave there (factorized
  backprojection -42.6 dB from exact backprojection on an 8192 by 8192 grid, polar format -17.4 dB). On one more
  collection of 2024 the float16 former overflows the half range, as in 0.1.8, and `precision='float32'` forms it.
- Polar format on one Umbra spotlight collection (2023-10-13, UMBRA-04) differs from exact backprojection by
  -24.8 dB, where the other collections checked give -26 to -33 dB. 0.1.8 gives the same image.
- Quadratic autofocus applies nothing where the parts of an image disagree on the error, which is the case on
  scenes with strong relief off the focus plane (a caldera; estimates from -4 to +3 rad across one scene). The
  image is then the one formed. `autofocus='spatially_variant'` is the mode for such scenes.
- After spatially variant apodization the main lobe measured on 30 compact points of six real collections is
  0.97 of the unweighted width along range and 1.05 along azimuth, with the sidelobe energy 5.3 dB lower.
- A vendor's SICD of the same collection can differ from the FastSAR product by a range offset and an azimuth
  scale that are properties of the vendor's processing ([docs/product-options.md](docs/product-options.md#against-a-vendors-sicd-of-the-same-collection)):
  several Capella spotlight SICDs show an azimuth scale of 4.5e-4 to 4.8e-4.

## [0.1.8] - 2026-10-27

### Added

- `read_cphd` and `form_cphd` read CPHD 0.3, the pre-standard NGA format, in the frequency domain: the frequency
  grid from `Fx0` and `Fx_SS`, the phase sign from `PhaseSGN`, the channels from `Data.ArraySize` (selected by index
  or polarization), the collection details from `CollectionInfo` and the image area from `Global.ImageArea`'s
  corners. A 0.3 file whose frequencies are offsets from an unstated reference frequency (`RefFreqIndex` set) is
  refused, and 0.3's `TropoSRP`, whose units the format does not define, is reported but not applied. `meta`
  carries the file's `cphd_version`. On a simulated collection written as both 1.0.1 and 0.3, the two read
  identically and form images within -123 dB of each other. FastSAR reads each 0.3 channel's vector parameters and
  samples at the offsets the 0.3 layout gives: sarpy's 0.3 reader places channel c's vector block c vectors into the
  file instead of after the earlier channels' vectors, which gives every channel but the first the first channel's
  positions and frequencies, shifted by c pulses, and scales its samples with them. A single channel's
  polarization is also taken from a `Polarization` or `Polarisation` parameter of `CollectionInfo` (as the SARCASTIC
  simulator writes it), and image-area corners that coincide (SARCASTIC writes all four at 0, 0, 0) count as no
  image area. All 168 files of a SARCASTIC v2.0 T-72 data dome (CPHD 0.3, doi 10.5522/04/c.5892392) read and form.

### Changed

- The README, `docs/comparison.md` and `docs/performance.md` give the 0.1.7 release records. 0.1.8 changes only
  the CPHD readers: on the 14 Capella and Umbra collections of the release check it forms images bitwise identical
  to 0.1.7's, at 0.98 to 1.01 times 0.1.7's time, so those records stand for 0.1.8.

### Known issues

- On one 2026 Capella spotlight collection from the open-data archive
  (`CAPELLA_C13_SP_CPHD_HH_20260105024946_20260105024955`), factorized backprojection on an 8192 by 8192 grid
  differs from exact backprojection by -42.6 dB, where the other collections checked give -60 to -62 dB, and the
  polar format differs by -17.4 dB. 0.1.7 gives the same images. The collection has 3047 pulses without samples
  inside its aperture, more than any collection checked before; whether these cause the error is not yet
  established. The fix is planned for 0.1.9.

## [0.1.7] - 2026-10-19

### Changed

- Polar format: the final resampling (the planar-wavefront displacement removed) is a CUDA kernel on GPUs and a
  C++ kernel on the CPU that computes each pixel's apparent position and carrier itself (`FASTSAR_PFA_GATHER=0`
  keeps the JAX gather); the two chirp-z transforms are replaced by direct Kaiser-windowed-sinc resampling of the
  windowed profiles, CUDA and C++ kernels reading the kernel from a table (`FASTSAR_PFA_CZT=1` keeps the
  transforms, which the TPU still uses); on a GPU the image is scaled on the device and returned through pinned
  memory and the history is uploaded through pinned memory. The Panama image in development: 3.0 s at 0.1.6 and
  1.45 s now on an L4, 11.2 and 8.9 s on the c4d-highmem-16, the same -32.1 dB; the direct resampling agrees with
  the transforms to -61 dB.
- Polar format: a range or azimuth band wider than the grid's raster (a pixel spacing coarser than the resolution
  the aperture or the bandwidth supports) is cropped to the raster with a warning; before, the padding to the raster
  failed with a negative width. The CPU polar format uploads nothing; on a GPU the history goes up in pinned row
  blocks, so no second host copy of it is made.
- Polar format warns when the grid's range axis `e2` is more than 2 degrees from the mid-aperture line of sight in
  the image plane, the orientation its polar raster assumes (a grid rotated 15 degrees on a squinted Umbra spotlight
  gave -5 dB against backprojection where the aligned grid gave -27 dB; found by the release gate, present in 0.1.6).
- `ExactFormer` on cuda with `interp='linear'` keeps the cropped range profiles in float16 on grids of 2^25 pixels
  or more (`profiles='auto'`; `'float32'`, `'float16'` and `'float16-pair'` select explicitly), scaled per chunk on
  the device: the Panama image in 10.7 s against 11.7 s with float32 profiles on an L4 in development, at the same
  -56.9 dB. The saving is in the pixel gather, so on twelve held-out collections it was 4 to 6 per cent on 8192 by
  8192 grids and a cost of up to 3 per cent on 4096 by 4096 grids, where the per-chunk scaling is the larger term;
  below the threshold the former reads float32 profiles as before. Cubic interpolation keeps float32 profiles
  (float16 would cost 0.5 dB at -67 dB).

## [0.1.6.post1] - 2026-10-18

### Changed

- README, `docs/performance.md` and `docs/comparison.md` carry the 0.1.6 release records (the PyPI page shows the
  README of the upload, so the text is republished); no code change.

## [0.1.6] - 2026-10-18

### Changed

- cuda `ImageFormer` and `form_cphd`: the first-level kernel's raw window is kept in registers on most devices and in
  a second shared-memory copy on devices with 200 KB or more of shared memory per multiprocessor (the H100), whichever
  the paper's record measured faster; 0.1.5 kept it in registers everywhere, which cost the H100 14 per cent of the
  fast former's time against 0.1.4 and the A100 8 per cent (the L4 gained 3 to 6 per cent). With the choice the H100
  is 6 per cent slower than 0.1.4 on the Panama record (0.600 against 0.568 s), the A100 8 per cent (1.17 against
  1.08 s) and the L4 3 to 6 per cent faster: the price of an image that does not depend on the memory free at the
  call, which the 0.1.5 release fixed. The rotation's arithmetic is written in explicit fused multiply-adds, so the
  variants give the same bits (left to the compiler they differed in the last bit on the H100 and A100); the images
  differ from 0.1.5's at that rounding level.

## [0.1.5] - 2026-10-18

### Changed

- cuda `ExactFormer`: the thread blocks take their tiles along one grid axis or the other, so that the blocks resident
  together read the same range bins from the L2 cache instead of the image's whole range window, and the image comes
  back through pinned memory, as `ImageFormer`'s does. The order is timed: the second and third chunks of pulses run
  in the two orders between events, the faster is kept from the first chunk at which the events have completed
  (asked, never waited for), and every 32 chunks another pair checks the choice; the geometric guess was not reliable
  (the axis closer to constant range at mid-aperture made a 212,858-pulse Capella spotlight 3 percent slower than
  0.1.4 on an L4, and two short calls 2 to 3 percent slower), and a call of fewer than five chunks keeps 0.1.4's
  order (compiled into the kernel, two builds, since a run-time flag cost it 3 per cent on an L4). On an RTX PRO 6000 Blackwell Max-Q the 12,207 by 8,808 pixel, 15,186-pulse Panama image took 2.59 s with
  linear and 3.82 s with cubic interpolation (4.54 and 5.84 s in 0.1.4); the images are bitwise the same.

### Fixed

- cuda `ImageFormer` and `form_cphd`: the image depended on the GPU memory free at the call. The first-level kernel
  loaded a window of the history into shared memory once per launch and rotated it in place from one child to the
  next, so a child's rounding carried every rotation before it in the same launch, and the children a launch holds
  are the group (sized from the free memory, halved after an out-of-memory) at the first level and the level's
  children or their halves at the later ones. Two calls on a 24 GB L4 could give images -107 dB of the peak apart
  (-84 dB in RMS) for a 9 GB Capella spotlight, and group sizes of one and eight on a 96 GB card -118 dB apart for an
  Umbra spotlight; the same at 0.1.4. Every child is now rotated from the raw window (this thread's samples kept in
  registers, or a second copy in shared memory for a window of more than 48 samples per thread), and the image is
  the same whatever the group, the batch or the splitting; the kernel's variants are compiled from their own source,
  so a first call does not rebuild the other kernels for each. The images differ from 0.1.4's at that rounding level
  and from nothing else; the CPU and TPU formers are unchanged.
- cuda: the blocks in which a host phase history is moved to the device were 4096 pulses whatever the row length,
  so a collection with 134,400 samples per pulse (an Umbra spotlight) made 4.4 GB blocks and ran out of GPU memory
  in the window multiply. Blocks are now at most 512 MB; the planes are the same.
- cuda `form_cphd` on a mosaic (a sliding spotlight or stripmap): a patch's range profiles gathered on a prefetch
  thread while the former held its peak could fail with CuPy's `OutOfMemoryError` (a 29 GB Capella sliding
  spotlight on an L4 with four prefetch workers, in three of four calls at 0.1.4 and this release's candidate; a
  14 GB one in one of four calls at 0.1.4 and three at the candidate). The gather holds one whole-gate array past
  each step instead of six (the ramp and taper are applied in place; the values are the same), and after an
  out-of-memory it returns the pool's free blocks to the device and tries again for up to 20 s, from the fourth
  attempt giving up the range-profile window on the device as well, since the former cannot release memory while it
  waits for this patch.
- cuda `ImageFormer` with `precision='float16'`: a history that streams from host memory (larger than the GPU's free
  memory) was scaled by the peak of its first 4096 pulses without the fallback the uploaded path has, so a later,
  stronger block overflowed the half range and the image came out non-finite (a 21.8 GB Capella spotlight on an L4).
  A non-finite float16 image from a finite history is now formed again with the whole history's peak, and a
  `ValueError` that names float32 is raised if that overflows too.
- After a cuda `ExactFormer` image, CuPy's FFT plan cache kept the plan of its range compression, and with it a work
  area in CuPy's memory pool (7 GB for an 8192 by 8192 image of a wide-band Umbra collection on an L4), so a
  following float16 `ImageFormer` could fail with CuPy's `OutOfMemoryError`. Every retry after an out-of-memory
  error on the device (the fast and exact formers, a mosaic's gathers, JAX's turn on the GPU) now returns CuPy's
  cached blocks and its FFT plans to the device before trying again (`memory.cuda_release`); between calls the plans
  stay cached as in 0.1.4 (a candidate that made them per call ran a wide-band exact image 3 per cent slower on an
  L4). The images are unchanged.
- TPU: the host preparation of a large phase history uploaded each float32 plane in one transfer, which on a v6e
  took 26 s for a 5 GB plane in every call of some processes and 1 s in others (the 2024 Capella spotlight, 10 GB:
  images of 12 s took up to 149 s). Planes of 1 GB or more now go to the device in blocks of 4096 pulses, written
  into planes allocated there (smaller planes, such as the patches of a stripmap mosaic, keep the single transfer,
  which is faster for them). On a v6e the ten warm calls of that spotlight took 11.0 to 11.5 s (11.9 to 12.4 s in 0.1.4 in the
  same process when no call stalled), the 2025 spotlight 9.6 s (9.9 s) and the 2021 stripmap 8.4 s (8.4 s), with
  the same region errors.
- TPU: a first-level kernel whose pulse blocks by range window exceeded the v6e's 128 MB of vector memory failed to
  compile, so that collections with range lines of about 10,000 samples formed no image (a 0.8 GB Capella stripmap
  and two Umbra spotlights). `ImageFormer` now retries such a plan with smaller pulse blocks, halving them until the
  kernels compile, and keeps the setting for the plan; plans that compiled before run unchanged.
- TPU: a collection whose decimated range line fits one output block of the level kernel but whose row is very long
  (134,000 samples, an Umbra spotlight; 36,600 for another) made that block of the decimation matrix up to 137 MB in
  vector memory, beyond the 128 MB of a v6e, and no pulse block was small enough. Rows longer than 32,768 samples are
  now rotated and multiplied in segments whose products accumulate in the output block, and a row whose decimated
  length allows it is cut into narrower output blocks; rows that compiled before run the same program. With this and
  the two entries above, the v6e forms 1024 by 1024 pixel images of ten held-out Capella and Umbra collections (0.8
  to 22 GB) on which 0.1.4 compiled no grid up to 8192 by 8192, with warm calls of 0.9 to 11.8 s.
- TPU: a device buffer allocation that fails surfaced through a `ValueError` carrying XLA's text, which the upload's
  conversion to `MemoryError` did not catch; every exception that reports the device out of memory is now converted.
- TPU: the padding of the uploaded planes for the first-level kernel held both copies at once, so a 22 GB history
  needed 34 GB of temporaries on a 32 GB v6e. The blocks of a large history now go straight into the padded planes.
  A history that does not fit raises `MemoryError` with the memory it needs, as documented, not XLA's error.

## [0.1.4] - 2026-10-16

### Fixed

- CUDA mosaics were 1 to 40% slower in 0.1.3 than in its first release candidate, from two causes. `form_mosaic` freed
  CuPy's pinned memory pool at its end, so every later mosaic page-locked its host buffers again, among them the
  block of the range profiles' size that `cupy.asarray` copies them through on the way to the device (2.3 GB on the
  84-patch 2021 Capella stripmap). Large uploads now go through two 64 MiB page-locked blocks, a patch's gathered
  profiles through page-locked staging buffers kept for the mosaic, and pinned requests are rounded up to size
  classes (at most 1/8 larger) so that patches of slightly different sizes share blocks; the pool is still emptied
  after every mosaic. The reader's row passes (troposphere and re-reference phases, the window), which 0.1.3 runs on
  all cores, allocated large temporaries in every worker thread; the allocator kept them, and the later mosaic took
  more page faults. Their scratch arrays are now made on the calling thread, one per block in flight. In paired runs
  on one L4 instance (g2-standard-32) the three Capella mosaics took 9.4, 50.8 and 189.0 s (12.8, 53.6 and 197.7 s
  in 0.1.3) at 4 to 8 GB less peak memory, and every image is
  bitwise the same as in 0.1.3.

### Known issues

- On a GPU whose memory is nearly filled by an earlier former (an 8192 by 8192 `ExactFormer` image of a wide-band
  Umbra collection on an L4), a following float16 `ImageFormer` can fail with CuPy's `OutOfMemoryError` instead of
  FastSAR's `MemoryError` or a slower path. Release 0.1.3 behaves the same way.
- On a TPU v6e the host preparation of a 10 GB phase history (the 2024 Capella spotlight: scaling to float32 planes
  and upload, `ffbp2.prepare`) takes 4 s in most calls but 23 to 96 s in some, at random within a process, so a
  12 s image took up to 149 s. Earlier releases behave the same way.

## [0.1.3] - 2026-10-15

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
- Speed on the Umbra Panama image: float32 from 3.86 to 2.9 s on an L4, from 1.87 to 1.07 s on an A100 and
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
  estimated from the product's calibration packets (`sentinel1.internal_delay`, about 0.43 µs, within 4 ns of
  ESA's annotated value on a 2014 datatake). Compared in the agencies' own radar coordinates, the images agree
  with JPL's RSLC of a NISAR pass to 0.2 m and with ESA's SLC of a Sentinel-1 datatake to 0.5 m in slant range
  (1.4 m and 1.3 to 1.5 m along track), from 100 to 180 m before the delays were removed. The Sentinel-1 packet
  time stamp is taken as latched 1.439 µs before the packet's own transmit pulse (`sentinel1.T_LATCH`, ESA's Level 1
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
  `oss/bench_cpu_synth.py` in sar-accel-study (5,794 pulses) forms in 3.7 s factorized and 3.5 s exact at 1024 by 1024, from 6.3 and
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

[0.1.4]: https://github.com/saulpingerman/FastSAR/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/saulpingerman/FastSAR/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/saulpingerman/FastSAR/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/saulpingerman/FastSAR/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/saulpingerman/FastSAR/releases/tag/v0.1.0
