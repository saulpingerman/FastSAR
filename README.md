# FastSAR

SAR image formation for Cloud TPUs, Nvidia GPUs and x86 CPUs from one Python call.

What it covers, with the section that documents and validates each part:

- spotlight phase histories (CPHD, frequency domain), by factorized backprojection with device kernels or polar
  format; the final tile size follows the collection's range (Tile size and range)
- exact backprojection onto any points: DEM surfaces, map grids, bistatic geometry, moving reference points
  (Exact backprojection)
- CPHD reading with per-pulse frequency grids, moving reference points, channel selection and alignment to a
  vendor SICD grid (Reading CPHD)
- wide-angle and circular apertures (Wide-angle and circular apertures)
- phase gradient autofocus (Autofocus)
- stripmap focusing by range-Doppler and omega-k (Stripmap)
- ScanSAR and TOPS burst modes (Burst modes)
- long apertures and arbitrary tracks (airborne motion) by patches of factorized backprojection (Long apertures and arbitrary tracks)
- interferograms and coherence for interferometry and change detection (Interferometry and change detection)
- multilooking, terrain-corrected geocoding, GeoTIFF and SICD output (Products and geolocation)
- checks on real collections: Umbra spotlight, Capella stripmap and spotlight against the vendors' images (Real collections)
- comparison with RITSAR, the AFRL backprojection, ISCE3 and torchbp on matched hardware (Comparison with open-source implementations)

```python
import fastsar
col = fastsar.io.read_cphd('scene_CPHD.cphd', sicd='scene_SICD.nitf')   # phase history and the vendor's grid
img = fastsar.form_image(**col)                                          # complex64 [azimuth, range]
```

For many images of the same geometry, build the plan and compile the kernels once:

```python
former = fastsar.ImageFormer(col['ant'], col['fmin'], col['df'], col['S'].shape[1],
                             col['nx'], col['ny'], col['spx'], col['spy'], col['e1'], col['e2'])
img = former(col['S'])            # each further call pays only the image formation
```

The main algorithm is factorized backprojection (three levels of tiles, Kaiser decimation filters, a 2T by 2T
product per final tile), with a kernel for each kind of device:

| backend | device | kernels |
|---|---|---|
| `tpu` | Cloud TPU (v5e, v6e) | Pallas: fused rotation and decimation, final stage on the matrix units |
| `cuda` | Nvidia GPU | CUDA through CuPy: shared-memory filters, tensor-core final stage for float16 |
| `cpu` | x86-64 with AVX-512 or AVX2 | C++ with OpenMP, compiled with g++ on first use |
| `jax` | anything JAX runs on | the plain JAX program, for checking |

All four use the same plan, filters and float64 geometry, and the CUDA and C++ float32 images agree with each
other to about -87 dB. `backend='auto'` picks the TPU if JAX sees one, else a GPU if CuPy sees one, else the CPU.
Polar format, with the resampling that removes its planar-wavefront displacement, is there as `algorithm='pfa'`.

## Comparison with open-source implementations

On the Umbra Panama collection (12,207 by 8,808 pixels from 15,186 pulses), against a float64 exact backprojection,
with every CPU implementation on the same c4d-highmem-16 instance (AMD EPYC 9B45, 16 vCPUs) and every GPU
implementation on the same Nvidia L4. Errors span three 512 by 512 pixel regions; times are for the full image,
estimated (est.) from one region for the slow implementations, which overstates them (the same estimate for
FastSAR's exact backprojection gives 2.4 h against 46 min measured):

- RITSAR as published, 16x oversampling: -25.3 to -27.3 dB, 44 h (est.)
- AFRL backprojection (`bpBasic`, MATLAB toolbox of Gorham and Moore, NGA release, in Octave), default FFT length:
  -38.1 to -43.2 dB, 57 h (est.); FFT length 2^18: -52.9 to -55.4 dB, 69 h (est.)
- ISCE3 `backproject`, 16 threads: -31.5 to -54.9 dB after a common linear phase ramp, 4.8 h (est.)
- FastSAR exact backprojection (`fastsar.backproject`): -54.9 to -58.7 dB, 46 min
- FastSAR factorized backprojection: -55.8 to -61.2 dB, 12.4 s
- on the L4: FastSAR exact backprojection -54.7 to -58.3 dB, 67 s; ISCE3 CUDA backprojection -31.5 to -52.2 dB, 27 min (est.); torchbp exact backprojection forms no image
  (its float32 pixel-to-antenna distance is 6 cm coarse at the 744 km range of the collection) in 15.2 s, and its
  factorized backprojection does not fit in memory; FastSAR factorized 3.6 s (float32) and 2.9 s (float16) at
  -54.6 to -61.2 dB

Polar format on the same CPU instance: the NGA `pfa_mem`, in Octave, forms the full image in 44 s with a log-amplitude
correlation of 0.80 to 0.89 and a mean 5 by 5 coherence of 0.76 to 0.88 against the reference, after registration of
its uncorrected image; FastSAR's corrected polar format takes 11.3 s at 0.996 to 0.999 and 0.999. The scripts and
records are in the study repository (https://github.com/saulpingerman/sar-accel-study, directories `oss` and
`results/oss`), and Appendix G of its paper gives the details.

## Speed

Time to form one Umbra spotlight image of the Panama Canal (12,207 by 8,808 pixels from 15,186 pulses of 14,399
samples), at float32-class accuracy (about -59.5 dB against a float64 exact backprojection), on-demand prices
of October 2026:

Times are the image formation itself (an `ImageFormer` call); a one-off `form_image` call adds planning and
compilation, a few seconds on an image of this size.

| device | seconds per image | dollars per 1000 images |
|---|---|---|
| TPU v6e-1 (three-pass products) | 2.4 | 1.89 |
| TPU v5e-1 (three-pass products) | 4.6 | 1.62 |
| Nvidia L4 (float32) | 3.6 | 0.76 |
| Nvidia L4 (float16 storage, float32 accumulation) | 2.9 | 0.61 |
| 16-vCPU AMD EPYC 9B45 (c4d-highmem-16) | 12.4 | 3.25 |

On the TPUs `precision='single-pass'` (the device default) runs in 1.7 s (v6e) and 2.8 s (v5e) at -45.8 dB, which
leaves the amplitude image unchanged but adds phase error beside bright returns.

## Precision options

- `float32` (default): float32 data and products. On a TPU this means three bfloat16 passes per product.
- `float16`, CUDA only: float16 phase history and intermediates with float32 accumulation; 0.9 dB above float32.
  Needs `T=32`.
- `single-pass`, `three-pass`: the TPU's one- and three-pass products.

## Tile size and range

The final stage treats each T by T tile with a plane-wave model plus an aperture-mean curvature term. What that
leaves out grows as the square of the tile size and falls with range. With `T='auto'` (the default) FastSAR
predicts the error from the collection geometry and takes the largest tile (32 or 16 pixels) that meets
`target_db` (default -40 dB), warning when neither does. The prediction tracks measurement to within about 2 dB
on simulated scenes of 0.5 m pixels:

- 1 km: T=16, -36 dB (T=32 would give -22 dB; the warning suggests coarser pixels or exact backprojection)
- 4 km: T=16, -48 dB (T=32: -35 dB)
- 16 km and beyond: T=32, -47 dB at 16 km, falling to about -62 dB at orbital range

Spaceborne collections therefore keep T=32. CUDA float16 always uses T=32.

## Install

```
pip install -e .                 # numpy, scipy, jax
pip install -e .[cuda]           # adds cupy-cuda12x (the tensor-core kernel also needs nvcc and g++)
pip install -e .[io]             # adds sarpy for CPHD and SICD
```

For a TPU install `jax[tpu]` as Google documents. The CPU backend needs g++; set `CXX` to use another compiler
and `FFBP_CPU_FLAGS` to change the flags (default `-O3 -march=native -mprefer-vector-width=512 -funroll-loops`).
For the CPU, run one thread per core: `OMP_NUM_THREADS=<cores> OMP_PLACES=cores OMP_PROC_BIND=close`.

## Tests

```
python tests/test_api.py           # every backend this machine has against the JAX program, and polar format
python tests/test_ffbp_cpu.py      # C++ kernels against the dense JAX image, two and three levels
python tests/test_ffbp_cuda.py     # CUDA kernels against the dense JAX image
python tests/test_pallas_fused.py  # TPU kernels in interpret mode (runs on a CPU)
python tests/test_autofocus.py     # phase gradient autofocus against injected phase errors (jax; or pass cpu)
python tests/test_stripmap.py      # stripmap omega-k and RDA against float64 backprojection
python tests/test_patches.py       # patch-wise FFBP for stripmap and non-linear tracks against backprojection
python tests/test_burst.py         # ScanSAR and TOPS burst focusing against float64 backprojection
python tests/test_products.py      # layover projection, geocoding and multilooking on off-plane targets
python tests/test_bp.py            # exact backprojection: backends, bistatic, orbital range, moving reference
python tests/test_io.py            # CPHD helpers: frequency resampling, re-referencing, geodetic conversions
python tests/test_wide_angle.py    # 10 to 360 degree apertures against exact backprojection
python tests/test_insar.py         # change detection and interferometric height (needs finufft)
```

The tests use a small simulated scene and take seconds to a few minutes.

## Coordinates

`ant` holds the antenna phase centers in a frame whose origin is the scene reference point. The output grid is
given by pixel counts `nx, ny`, spacings `spx, spy` (m) and unit vectors `e1` (azimuth) and `e2` (range) of the
image plane; pixel (i, j) sits at `(i - nx/2) spx e1 + (j - ny/2) spy e2`. `read_cphd` builds all of
these from the files, using the SICD's grid when one is given.

## Exact backprojection

`fastsar.backproject(S, ant, fmin, df, points, rcv=None, ref=None)` forms the image at any points [..., 3]: the
plane of `plane_points(nx, ny, spx, spy, e1, e2, height=None)` (the grid of `form_image`, optionally lifted onto a
DEM), a map grid, or scattered points. With `rcv` the geometry is bistatic (`ant` is then the transmitter), and
`ref` gives per-pulse reference ranges for a phase history compensated to a moving point. Its cost is pulses times
points, so for large planar images `form_image` is the fast path. Ranges are computed in float64 at the centers of
blocks of 256 nearby points and in float32 within a block, so the float32 kernels keep their accuracy at orbital
range. Against a float64 reference (`tests/test_bp.py`, upsample 16) the CPU kernel, which works in float64,
agrees to -77 to -82 dB, and the JAX and CUDA kernels to -72 to -76 dB, monostatic and bistatic, at 5 and 600 km;
a phase history compensated to a moving point and imaged with its reference ranges matches the fixed-point image
to -60 dB. Each pulse is range compressed by a zero-padded FFT (`upsample` times its length) and read with linear
interpolation; the image error against a 128x reference falls by 12 dB per doubling of `upsample`, from -39 dB at
2 to -63 dB at 8 (the default) and -88 dB at 32.

## Reading CPHD

`fastsar.io.read_cphd(path, sicd=None, channel=0, meta=False)` reads a frequency-domain CPHD with sarpy and returns
the arguments of `form_image`; with `meta=True` it also returns the transmitter and receiver positions, the
per-pulse reference ranges and the frame (`io.local_to_ecf`, `io.ecf_to_geodetic` and their inverses convert
points). It handles:

- per-pulse frequency grids (FXFixed false), resampled onto a common grid with a 16-tap Kaiser sinc (-80 dB on a
  test signal); on the Panama collection the largest offset is 0.0045 samples and the image changes by -65 to
  -94 dB
- a moving scene reference point, re-referenced to the mid-aperture point when the range change is small and
  otherwise left for `backproject` with `ref=meta['ref']`
- channels by index or identifier (polarizations)
- pulses with invalid positions (trimmed at the ends, interpolated inside) and empty or flagged pulses (reported)
- with a vendor SICD, the vendor's grid: the origin moves onto its pixel grid so that the image is the SICD array
  (transposed when range runs along rows); on Panama our pixels coincide with the vendor's to a quarter pixel near
  the scene center
- optionally (`troposphere=True`) the per-pulse troposphere delay at the scene reference point

## Real collections

Besides the Umbra spotlight collections of the paper, FastSAR has been run on two Capella collections from the
Capella Open Data program, compared with the vendor's own SICD of each:

- stripmap (2021, 39,899 pulses of 6,003 samples, scene reference point moving 27.5 km with the beam):
  `examples/form_capella_stripmap.py` forms a 512 by 512 pixel crop of the vendor's range / zero-Doppler grid by the
  patch mosaic on the CPU (165 patches of about 10,300 pulses, 398 s on four threads) and samples it on the vendor's
  pixels; the amplitude images correlate at 0.960
- spotlight (2024, 74,203 pulses of 17,282 samples): exact backprojection onto the vendor's pixels (`io.sicd_points`)
  correlates with the vendor's polar-format image at 0.78 at the center of the scene and 0.54 off center, where the
  vendor's image carries the polar-format distortion

Every Capella CPHD in the program declares SGN = +1, but the phase of both collections follows SGN = -1: with the
declared sign the images do not focus or do not match the vendor's (correlation -0.01 to 0.05). `read_cphd`
therefore takes SGN = -1 for Capella collectors and says so in its notes; `phase_sign` overrides it.

A Capella dynamic stripmap (sliding spotlight) collection of 2022 is not supported. Its CPHD declares a valid delay
window (TOA1 to TOA2) 2,962 m long and 4.1 to 7.1 km beyond the scene reference point, while its frequency spacing
(114.3 kHz) gives an unambiguous range of 1,311 m: the samples cannot hold that window as a plain frequency-domain
phase history, so the file follows a convention FastSAR does not model. Exact backprojection matches the vendor's
image near the scene center only after a 2-pixel registration (amplitude correlation 0.40 to 0.51 on low-contrast
terrain) and fails near the edge of the swath.

## Wide-angle and circular apertures

Factorized backprojection makes no small-angle assumption, so the same call images wide-angle and circular
collections, which polar format cannot. `tests/test_wide_angle.py` forms 256 by 256 images at the resolution of
apertures of 10, 45, 120 and 360 degrees (X band, 1.5 GHz of bandwidth, pixels of 72 to 6 mm) and compares them
with exact backprojection: -53.0, -61.7, -63.8 and -65.2 dB. An axis of the phase history that has become shorter
than its decimation kernel at a late level (a few frequency samples remain at these resolutions) is left
undecimated at that level.

## Interferometry and change detection

Images of two passes formed onto the same grid or the same points are coregistered by construction, and a
scatterer that lies on the grid surface contributes zero interferometric phase, so no flat-earth or topographic
phase needs removing. `products.interferogram` gives the multilooked interferogram and `products.coherence` the
sample coherence over moving windows, the statistic of coherent change detection. `tests/test_insar.py` (needs
finufft for the clutter simulation) checks three cases on simulated X-band clutter at 5 km:

- change detection, same geometry, coherence 0.98 outside two disturbed areas and a vehicle track: median sample
  coherence 0.981 on unchanged ground and 0.275 inside the changes (7 by 7 windows)
- height, passes 3.5 m apart vertically (height of ambiguity 22.3 m), a 6 m block on flat ground, both formed onto
  the ground plane: the phase converts to 5.98 m on the block (at its layover position, 3.5 m toward the radar) and
  0.00 m on the flat ground, with coherence 0.999 there
- the same pair formed by exact backprojection onto the terrain itself: median phase 0.008 rad on the block top and
  0.003 rad on the flat ground

## Products and geolocation

`fastsar.products` turns a formed image into products: `multilook` and `to_db` for detected images, `project` and
`geocode` to place 3-D points (a map grid, optionally on a DEM) in an image and sample it there, `write_geotiff`
(needs rasterio) and `write_sicd` (needs sarpy). A point off the image plane appears where the plane has its range
and Doppler cone angle at the aperture center; `project` solves that pair of conditions in closed form, so terrain
correction is exact under that model.

`tests/test_products.py` forms an image on the ground plane of targets 0 to 30 m above it: each target appears
within 0.5 cm of its projection, with layover up to 21 m. On the Umbra Panama collection, a 2048 by 2048 crop
formed by factorized backprojection and geocoded onto the pixel grid of the vendor's GEC GeoTIFF, and exact
backprojection directly onto the same ground points, both land 6.98 m from the GEC when the surface is taken at the
height of the scene reference point (-0.34 m above the ellipsoid); the offset changes by 1.33 m per meter of
assumed height and vanishes at 5.3 m above that height, where exact backprojection onto the ground points also
focuses best (correlation with the GEC 0.75, against 0.13 at -0.34 m). The vendor's SICD projected through its
own model lands on the GEC with no offset.

`write_sicd` writes an image formed on a vendor SICD's grid (`read_cphd(..., sicd=...)`) with that SICD's
metadata, ImageFormAlgo OTHER and Grid.Type PLANE: the pixels are backprojection's on the template's image plane.
A round trip of the Panama SICD keeps every pixel and passes sarpy's validity check; its projection of the center
pixel agrees with the vendor's exactly and departs from it by up to about 6 m at the corners, where the vendor's
polar-format image is displaced by the distortion its own projection model accounts for.

## Stripmap

`fastsar.stripmap` focuses stripmap data from a straight, constant-velocity track, starting from raw echoes or from
range-compressed ones:

```python
import numpy as np
from fastsar import stripmap as sm, quality
p = sm.make_params(squint_deg=5.0)                          # airborne X-band geometry; see make_params
raw = sm.simulate(p, [[0.0, p.r0], [20.0, p.r0 + 80.0]])    # point targets at (x, zero-Doppler range)
img, r, x = sm.focus_stripmap(raw, p, algorithm='omegak', rwin='taylor')    # or 'rda', 'bp'
ij = np.unravel_index(np.abs(img).argmax(), img.shape)
m = quality.point_target(img, ij, d_az=x[1] - x[0], d_rg=r[1] - r[0])     # resolution, PSLR, ISLR, peak
```

The simulator computes the echoes of each target exactly in float64 in the time domain (up-chirp, two-way sinc^2 or
Gaussian azimuth pattern, optional squint). Omega-k uses the exact two-dimensional reference phase and the exact
Stolt mapping, interpolated with a windowed sinc; the range-Doppler algorithm corrects the exact range migration
r/cos(theta) by sinc interpolation and has optional secondary range compression. Both run in JAX on any device, in
float32 by default or float64, with phases and interpolation positions computed in float64 on the host. Time-domain
backprojection in numpy float64 is the reference. All three return the image on the zero-Doppler grid with the same
complex scale and phase, so they can be compared sample by sample.

`tests/test_stripmap.py` checks them on a 9.6 GHz scene (100 MHz chirp of 2 us sampled at 125 MHz, PRF 650 Hz,
200 m/s, 5 km range, 1.5 m antenna, 1024 pulses of 448 samples) with five targets across a 200 m swath, a Taylor
range window and the sinc^2 pattern as the only azimuth weighting. The error against backprojection on 64 by 64
pixel patches around the targets is:

- broadside: omega-k -64.9 to -65.5 dB, RDA with secondary range compression -66.5 to -67.7 dB, without it -65.4 to
  -66.3 dB
- 5 degrees of squint (Doppler centroid 1116 Hz, 1.7 times the PRF): omega-k -60.2 to -62.5 dB, RDA with secondary
  range compression -61.1 to -67.4 dB, without it -27.4 to -27.7 dB

At broadside every algorithm measures 0.585 m in azimuth and 1.79 m in range against 0.585 m and 1.775 m expected
from the weighting, with peak sidelobe ratios of -39.6 dB (azimuth) and -32.7 dB (range; the chirp's spectral ripple
raises the Taylor sidelobes by about 2 dB) and peak positions within 1.5 mm of the truth. The squinted image is
sheared, so its range cut measures 1.74 m. The float32 and float64 omega-k images differ by -125 dB, and the residual
against backprojection comes from the edge of the processed Doppler band, which backprojection cuts on pulses and
the other two on Doppler bins. On four CPU threads omega-k takes 0.3 s and RDA 0.1 s for this scene.

## Long apertures and arbitrary tracks

`fastsar.patches` forms stripmap images, and images from any antenna track, with the spotlight factorized
backprojection. The output grid is cut into patches, and each patch is formed by an `ImageFormer` from the pulses
whose beam covers it, with the phase history referenced to the patch center:

```python
from fastsar import stripmap as sm, patches
img, r, x = patches.form_stripmap(raw, p, rwin='taylor')       # straight track, the grid of focus_stripmap
fx = patches.echoes_to_fx(raw, p, rwin='taylor')               # echoes -> frequency-domain phase history
img = patches.form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2,
                          beam=patches.stripmap_beam(p, ant), backend='cpu')   # antenna positions ant [P, 3]
```

`echoes_to_fx` converts time-domain echoes in the conventions of `fastsar.stripmap` into the CPHD form with
per-pulse reference ranges: the range FFT of the range-compressed pulse gives the phase history at fc + f_tau,
referenced to the range of the first fast-time sample. `form_mosaic` also accepts such a phase history directly. For
each patch, every pulse is re-referenced to the patch center (a linear phase in frequency), its range profile is kept
over a gate that covers the patch plus a margin, and the gated profile is transformed back onto a frequency grid
whose unambiguous range equals the gate. A raised-cosine taper over the guard band between the signal band and the
sampling rate (the frequency axis is padded where there is none) shortens the reconstruction kernel; the returns
dropped by the gate reach the patch only through the kernel's tail, which is below -60 dB at the default margin,
36 m for a 12.5 MHz guard. A pulse
serves a patch when its normalized Doppler satisfies |u| <= umax at some point of the patch. An azimuth window,
which `stripmap.backproject` evaluates per pixel, is split by a singular value decomposition of the pulse-by-pixel
weight into separable terms, each formed by one factorized backprojection; terms are kept down to 50 dB below the
first, three or four for 40 m patches here. The C++ and CUDA final stages take tiles of 16 or 32 pixels only, so a
patch whose predicted error at T = 16 misses the target (-40 dB) is formed on a grid twice as fine in range and
decimated; the JAX program also accepts T = 8.

`tests/test_patches.py` uses the scene of `tests/test_stripmap.py` and 128 by 128 pixel patches (12 patches of
730 to 900 pulses for a 280 by 448 pixel image, about 1 s on four CPU threads). Against float64 backprojection
(`stripmap.backproject`) on 64 by 64 pixel neighborhoods of the five targets, after the best complex gain:

- broadside: -48.0 to -60.4 dB (CPU), -54.8 to -60.2 dB (JAX, which uses T = 8 where needed); the same from
  range-compressed input
- 5 degrees of squint: -48.0 to -62.6 dB
- Taylor azimuth window: -48.5 to -58.8 dB with three or four terms per patch
- the same patches formed by exact backprojection instead of FFBP: -65.3 to -66.7 dB without an azimuth window and
  -54.7 to -67.6 dB with the Taylor window, the share of the patching itself

Resolution agrees with backprojection to 0.7 % in azimuth and 0.1 % in range, peak sidelobe ratios to 0.4 dB (1.3 dB
for the -62 dB azimuth sidelobes of the Taylor window) and peak positions to 2 mm. A phase history cut to the signal
band (no guard, so the frequency axis is padded) and referenced per pulse to a point that moves with the platform
gives the same image as the echo input to -141 dB.

The second part flies the same radar at 3 km altitude over flat ground on a track with smooth cross-track and
vertical motion, two sinusoids each with periods from 0.47 s to 1.7 s, 1.68 m and 1.57 m peak to peak, which changes
the range to the swath center by up to 0.97 m. Omega-k on the nominal straight track does not focus these data: the
peaks reach 6 to 9 % of the exact image and the error is about 0 dB, while on echoes from the straight track the
same comparison gives -64.9 to -65.5 dB. The patch mosaic with the true positions, on a ground-plane grid of 0.31 by
1.5 m, matches exact backprojection with the true positions to -49.9 to -64.9 dB (-65.6 to -66.1 dB with exact
patches), with the same resolution, sidelobes and positions as above.

Limitations. Each pulse enters every patch its beam covers, about six along track for this 208 m footprint and
40 m patches, so longer patches lower the cost per pixel. Without an azimuth window the per-pixel mask |u| <= umax of
`stripmap.backproject` is reproduced only where the data vanish beyond umax (umax at the extent of the simulated
pattern, the default); with a smaller umax, pixels near a patch edge also receive pulses that their own mask would
exclude. `echoes_to_fx` assumes one fast-time grid for all pulses. `stripmap_beam` models an antenna held along a
fixed direction (or one direction per pulse); other patterns can be passed as a function of pulse and point. The
method has been validated on simulated point targets only.

## Burst modes (ScanSAR and TOPS)

`fastsar.burst` simulates and focuses burst-mode data on the same straight, constant-velocity track:

```python
from fastsar import burst as bm
bs = bm.make_bursts('tops', nburst=2)                   # or 'scansar', subswaths=[(5e3, 200.0), (5.25e3, 200.0)]
raw = bm.simulate(bs[0], [[0.0, bs[0].r0]])              # one burst's echoes
img, r, x = bm.focus_burst(raw, bs[0], algorithm='omegak', rwin='taylor')    # or 'rda'
```

A burst is a `BurstParams`, the stripmap parameters of its pulses plus a steering rate `kpsi`: the beam points at
psi(eta) = squint + kpsi (eta - eta_mid), so the Doppler centroid is 2 v sin(psi)/lambda and varies at
k_t = 2 v kpsi/lambda within the burst. ScanSAR has `kpsi = 0`. In TOPS (`kpsi > 0`, backward to forward) the
footprint moves at v + r kpsi, each target is seen for the fraction alpha = v/(v + r kpsi) of the stripmap dwell,
and the azimuth resolution is the stripmap value divided by alpha. Subswaths differ in range window and in an
elevation gain applied to each target's zero-Doppler range. The simulator keeps the exact time-domain echoes of
the stripmap simulator with the beam of each pulse.

Each burst is padded in azimuth to cover the zero-Doppler times of all targets it illuminates. A ScanSAR burst is
then focused with stripmap omega-k or RDA, which is the full-aperture approach. A TOPS burst covers a Doppler band
of k_t T_burst + 4 v/La, several times the PRF, so the stripmap focusers cannot take it directly. It is deramped
with the exact integral of its Doppler centroid, scaled by (fc + f_tau)/fc in the range-frequency domain, which
brings every instant to the beam's band. The azimuth window is applied there as a function of the position in the
beam, the spectrum is zero-padded to L times the PRF (L = 3 in the example below), the data are reramped on the
fine grid and focused, and every L-th row is kept. Outside the burst the deramp holds the steering at its end
values, which keeps the reramped signal inside the fine band. The burst images lie on the zero-Doppler grid with
rows on the pulse grid, have the complex scale of backprojection of the same burst's pulses, and are combined
by `mosaic`, which takes each row from the burst that illuminates it with the most pattern energy.

`tests/test_burst.py` uses the radar of the stripmap test (9.6 GHz, 100 MHz, PRF 650 Hz, 200 m/s, 1.5 m antenna,
sinc^2 pattern, Taylor range window) and compares each burst image with float64 backprojection of the same pulses
on 64 by 64 pixel patches around targets at two ranges:

- TOPS, one burst of 512 pulses (0.79 s) steered at 0.12 rad/s from -2.70 to +2.70 degrees; the Doppler centroid
  runs from -604 to +604 Hz and the band including the beam spans 1741 Hz, 2.68 times the PRF. For targets
  illuminated at the beginning, middle and end of the burst, omega-k is within -60.1 to -63.1 dB of
  backprojection and RDA with secondary range compression within -61.8 to -64.4 dB. The azimuth resolution is
  2.316 to 2.356 m against 2.318 to 2.357 m expected (0.585 m divided by alpha = 0.25 at 5 km), the azimuth PSLR
  -39.3 to -39.5 dB and the peak positions within 1.6 mm. A target seen by only half the beam, at the first pulse,
  agrees to -56 dB. The float32 and float64 omega-k images differ by -126 dB. On four CPU threads omega-k takes
  3.7 s and RDA 1.2 s for the 2940 by 448 pixel burst image.
- ScanSAR, two subswaths at 5.0 and 5.25 km with alternating bursts of 128 pulses (0.197 s; 0.394 s between the
  bursts of a subswath; stripmap dwell 1.04 s). For targets crossing the beam center 0.25 s before the burst, at
  its first, middle and last pulse and 0.25 s after it, omega-k is within -59.0 to -68.1 dB of backprojection and
  RDA within -59.4 to -72.4 dB. The azimuth resolution, 1.766 to 2.185 m, agrees to within 0.1% with the value
  computed from each target's weighted Doppler support (1.735 m for uniform illumination of the whole burst at
  4.94 km). Targets seen near the edges of the beam have the coarser resolution and a lower PSLR (-19 dB against
  -14 dB for the sinc response of central targets), and all peak positions are within 1.8 mm. The mosaic of two
  bursts of one subswath takes each target from the burst with the larger illumination energy.

Two limits follow from the comparison. First, backprojection cuts the processed band on pulses and the frequency
domain on Doppler bins, which matters wherever data sit close to the band edge. The unweighted ScanSAR response
has sinc sidelobes, and 80 m from a target the two differ by -45 dB relative to the peak, where the sidelobes
themselves are at -41 dB; the ScanSAR targets are therefore simulated and compared one at a time. A ScanSAR target
seen only between u = -0.96 and -0.58 of the beam agrees to -48.6 dB. Second, the TOPS chain processes L times
the pulses of the burst. SPECAN and extended chirp scaling, which avoid this cost, are not implemented, and the
images keep the Doppler centroid variation of TOPS along azimuth.

## Autofocus

`fastsar.autofocus` estimates an unknown phase error per pulse by phase gradient autofocus (Wahl, Eichel,
Ghiglia and Jakowatz, 1994) and removes it from the phase history:

```python
img, phi = fastsar.autofocus.autofocus(S, ant, fmin, df, nx, ny, spx, spy, e1, e2, backend='cpu')
S_corrected = S * np.exp(-1j * phi)[:, None]
```

The image is formed by factorized backprojection, and PGA runs along its azimuth axis (axis 0): the brightest
pixel of each range line is centered, the lines with the highest peak-to-mean intensity are kept and windowed, and
the phase difference between adjacent azimuth frequency bins is estimated jointly over the kept lines. The window
narrows as the image focuses. The estimate on bins is mapped to pulses through each pulse's azimuth spatial
frequency at the center frequency, the phase history is corrected and the image re-formed, twice by default.

A backprojection image keeps the spherical wavefront, so the azimuth spectrum of a scatterer moves with its
position (by 16 of 128 bins for a scatterer 20 m from the center at 5 km range in the test below). Before PGA the
image is therefore multiplied by `exp(-1j * deramp_phase(...))`, the difference between the spherical and planar
wavefronts of the center pulse, which returns every scatterer to the polar-format convention in which one pulse
occupies one bin. `pga` can be called on its own on an image in that convention.

`tests/test_autofocus.py` injects phase errors into a simulated 128 by 128 image of 0.5 m pixels (0.6 m
resolution, 5 km range, 190 pulses) holding 40 point targets over 3,000 clutter scatterers. Image errors are
relative to the error-free image after the best complex gain, and residuals are rms over pulses after removing
the constant and linear terms:

- quadratic, 5.3 rad peak: image error -2.7 dB before, -30.1 dB after; residual 0.036 rad
- polynomial to fifth order, 4.1 rad peak: -6.8 dB before, -29.2 dB after; residual 0.037 rad
- low-pass random walk, 3.0 rad peak: -0.2 dB before, -30.3 dB after; residual 0.036 rad

On the error-free image the procedure changes the image by -30.1 dB (estimated phase 0.038 rad rms), and on the
same targets without clutter by -34.3 dB. This floor is the estimator's error on the scene, set by clutter and
by scatterers that share a range line, rather than a residual of the injected errors; two other random scenes
gave residuals of 0.035 and 0.05 rad. The method assumes an aperture that fits inside the image's azimuth band,
that is, azimuth pixels finer than the resolution, and an image long enough in azimuth for its bin spacing,
1/(nx spx), to follow the variation of the phase error across the aperture. It has been validated with
backprojection images only.

## Paper

The measurements behind the table, the kernels' design and their bounds against each device's units are in the
companion study, Singerman and Braun, *Cost and Precision of Spotlight SAR Image Formation on Google TPUs, an
Nvidia GPU and a CPU* (2026), code at https://github.com/saulpingerman/sar-accel-study.

## License

MIT.
