# Real data: CPHD, SICD and vendor collections

Reading CPHD needs sarpy (the `io` extra).

## Reading CPHD

`io.read_cphd` returns the arguments of `form_image` and, with `meta=True`, metadata
([api.md](api.md#fastsario)). Its local frame has x along track, y along ground range away from the radar and z along
the ellipsoid normal, with its origin at the (mid-aperture) scene reference point. The reader handles:

- per-pulse frequency grids, resampled by a 16-tap Kaiser sinc (-80 dB on a test signal); on Panama the largest
  offset is 0.0045 samples and the image changes by -65 to -94 dB
- a moving scene reference point, re-referenced to mid-aperture when the range change is small, else left for
  `backproject` with `ref=meta['ref']`
- channels by index, identifier or polarization
- invalid positions (trimmed at the ends, interpolated inside), empty or flagged pulses (reported in `notes`)
- a vendor SICD's grid: the image is the SICD array (transposed when range runs along rows); non-planar SICD grids
  need `io.sicd_points` and `backproject` ([processing-chain.md](processing-chain.md#2-form))
- the troposphere delay at the scene reference point, removed by default when the file gives a nonzero delay
  (`troposphere=False` keeps it; Umbra's SICD images keep it, Capella's remove it)
- a warning when the valid delay window (TOA1 to TOA2) exceeds the unambiguous range c/(2 df)

Time-of-arrival (TOA) CPHD raises a `ValueError`.

## form_cphd against vendor images

`form_cphd` was checked against the vendors' SICD images on 600 m grids around the scene center: correlation over
256 by 256 vendor pixels projected to the grid's height. These values were measured before `read_cphd`'s z axis
changed from the geocentric radial to the ellipsoid normal (up to 0.19 degrees apart; see the CHANGELOG). They
are also without the troposphere correction, which `read_cphd` applies by default since 0.1.1.

| Collection (SICD given) | Amplitude | Intensity, 5 by 5 average |
|---|---|---|
| Capella stripmap, 2021-11-12 (no) | 0.921 | 0.979 |
| Capella stripmap, 2021-11-12 (yes) | 0.891 | 0.973 |
| Umbra spotlight, Panama Canal (no) | 0.621 | 0.897 |
| Capella spotlight, mountains (no) | 0.440 | 0.868 |
| Capella spotlight, mountains (yes) | 0.605 | 0.904 |
| Capella dynamic stripmap, 2022 (yes) | 0.224 | 0.752 |

Amplitude correlation is limited by speckle where the two processors' windows or azimuth bands differ; the averaged intensity is the better indicator of common structure. The 2022 dynamic stripmap (Capella's name for sliding
spotlight) is the collection [below](#capella-dynamic-stripmap-scene-center-only).

## Umbra spotlight

Three Umbra open-data spotlight collections were formed on the vendor's grid:

| Scene | Image (az by rg) | Pulses by samples | Bandwidth | Range |
|---|---|---|---|---|
| Panama Canal, 2023-07-18 | 12,207 by 8,808 | 15,186 by 14,399 | 337 MHz | 738 km |
| Melbourne, 2023-02-08 | 10,077 by 10,273 | 15,948 by 14,399 | 405 MHz | 760 km |
| Iowa farmland, 2023-10-20 | 8,521 by 8,465 | 13,003 by 11,663 | 356 MHz | 653 km |

![The Panama Canal, Melbourne and Iowa collections, with four regions marked on Panama](images/scenes.png)

*Float64 exact backprojection, downsampled (45 dB, azimuth horizontal, 1 km bar). Boxes: the regions of
[precision.md](precision.md).*

Near the scene center FastSAR's pixels match the vendor's to a quarter pixel. The vendor's polar format does not correct the planar-wavefront displacement; its Panama image is displaced by up to 13.2 pixels in
azimuth and 9.0 in range, against 13.2 and 9.1 predicted from the geometry. After resampling the vendor image
through that model, the residual is at most 0.25 pixel rms on all three collections.

![Panama port, lock and ship regions: reference, vendor SICD, resampled vendor SICD and their coherence](images/sicd_panama.png)

*Port, lock and ship regions (512 by 512 pixels): the reference, the vendor image as delivered and resampled through
the displacement model, and their 5 by 5 coherence (0 to 1).*

The coherence in these panels is limited by differences in aperture weighting and motion compensation between the
two processors.

### Geolocation

A 2048 by 2048 Panama crop geocoded onto the vendor's GEC GeoTIFF grid, or formed there by exact backprojection,
lands 6.98 m from the GEC with the surface at the scene reference point's height (-0.34 m). The offset changes by
1.33 m per meter of assumed height and vanishes 5.3 m higher, where exact backprojection also focuses best
(correlation with the GEC 0.75, against 0.13 at -0.34 m). The vendor's SICD projected through its own model lands on
the GEC with no offset. These figures also predate the z-axis change noted above.

## Capella

Two Capella Open Data collections were compared with the vendor's SICD:

- Stripmap (2021-11-12, 39,898 pulses of 6,003 samples, scene reference point moving 27.5 km):
  [`examples/form_capella_stripmap.py`](../examples/form_capella_stripmap.py) forms a 512 by 512 crop of the
  vendor's range / zero-Doppler grid by the patch mosaic (165 patches of about 10,300 pulses; 12 s on the
  c4d-highmem-16 with release 0.1.0). The amplitude images correlate at 0.993.
- Spotlight (2024-10-04, 74,203 pulses of 17,282 samples): exact backprojection onto the vendor's pixels correlates
  with the vendor's polar-format image at 0.78 at the scene center and 0.54 off center, where the vendor's image
  carries the polar-format distortion.

A 2025 Capella spotlight (54,267 pulses of 17,282 samples) serves the timings of
[performance.md](performance.md#memory).

### Capella phase sign

Capella's open-data CPHDs declare SGN = +1, but both collections follow SGN = -1 (with +1, correlation with the
vendor's image is -0.01 to 0.05). `read_cphd` takes -1 for Capella and says so in `notes`; `phase_sign` overrides.

### Capella dynamic stripmap: scene center only

The 2022 Capella dynamic stripmap collection forms correctly only near the scene center. Its valid delay window is
2,962 m long, 4.1 to 7.1 km beyond the scene reference point, but its 114.3 kHz frequency spacing gives an unambiguous
range of 1,311 m. FastSAR does not model this convention, and `read_cphd` warns about it. Near the center `form_cphd`
and exact backprojection match the vendor's image after a 2-pixel registration (intensity correlation 0.75 after 5 by
5 averaging); near the swath edge they fail. The mosaic agrees with a float64 backprojection of the same samples to
-62 dB; this tests internal consistency only.

## ICEYE dwell spotlight

ICEYE dwell files declare the non-standard radar mode EXPERIMENTAL (`meta['mode']` None). `form_cphd` takes
ICEYE's SICD metadata `.xml` (`sicd='scene_SICD.xml'`). A 91,426-pulse dwell (28 GB) is read in place or streamed
([performance.md](performance.md#large-collections)); the plan of a 71,790 by 10,000 pixel image takes 0.1 s and
2.7 GB. No comparison with ICEYE's image is published.
