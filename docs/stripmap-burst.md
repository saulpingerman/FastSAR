# Stripmap, long apertures and burst modes

Three modules cover data that is not a single spotlight aperture:

- `fastsar.stripmap`: stripmap focusing by omega-k and the range-Doppler algorithm on a straight,
  constant-velocity track, with a simulator and a time-domain backprojection reference.
- `fastsar.patches`: stripmap images, and images from any antenna track, as mosaics of factorized-backprojection
  patches.
- `fastsar.burst`: ScanSAR and TOPS burst modes.

The numbers on this page are test results on simulated point targets. The patch mosaic has also been run on a
real Capella stripmap collection ([real-data.md](real-data.md)).

## Stripmap

```python
import numpy as np
from fastsar import stripmap as sm, quality
p = sm.make_params(squint_deg=5.0)                          # airborne X-band geometry; see make_params
raw = sm.simulate(p, [[0.0, p.r0], [20.0, p.r0 + 80.0]])    # point targets at (x, zero-Doppler range)
img, r, x = sm.focus_stripmap(raw, p, algorithm='omegak', rwin='taylor')    # or 'rda', 'bp'
ij = np.unravel_index(np.abs(img).argmax(), img.shape)
m = quality.point_target(img, ij, d_az=x[1] - x[0], d_rg=r[1] - r[0])     # resolution, PSLR, ISLR, peak
```

The simulator computes the echoes of each target exactly in float64 in the time domain (up-chirp, two-way sinc^2
or Gaussian azimuth pattern, optional squint). Omega-k uses the exact two-dimensional reference phase and the
exact Stolt mapping, interpolated with a windowed sinc. The range-Doppler algorithm corrects the exact range
migration r/cos(theta) by sinc interpolation and has optional secondary range compression. Both run in JAX, in
float32 by default or float64, with phases and interpolation positions computed in float64 on the host.
Time-domain backprojection in numpy float64 is the reference. All three return the image on the zero-Doppler
grid with the same complex scale and phase, so they can be compared sample by sample.

Test results (`tests/test_stripmap.py`: 9.6 GHz, 100 MHz chirp of 2 us sampled at 125 MHz, PRF 650 Hz, 200 m/s,
5 km range, 1.5 m antenna, 1024 pulses of 448 samples, five targets across a 200 m swath). Error against
backprojection on 64 by 64 pixel patches around the targets:

- broadside: omega-k -64.9 to -65.5 dB, RDA with secondary range compression -66.5 to -67.7 dB, without it
  -65.4 to -66.3 dB
- 5 degrees of squint (Doppler centroid 1116 Hz, 1.7 times the PRF): omega-k -60.2 to -62.5 dB, RDA with
  secondary range compression -61.1 to -67.4 dB, without it -27.4 to -27.7 dB

At broadside every algorithm measures 0.585 m in azimuth and 1.79 m in range against 0.585 m and 1.775 m expected
from the weighting, with peak sidelobe ratios of -39.6 dB (azimuth) and -32.7 dB (range) and peak positions within
1.5 mm of the truth. The float32 and float64 omega-k images differ by -125 dB. On four CPU threads omega-k takes
0.3 s and RDA 0.1 s for this scene.

## Long apertures and arbitrary tracks

`fastsar.patches` cuts the output grid into patches and forms each with an `ImageFormer` from the pulses whose
beam covers it, with the phase history referenced to the patch center:

```python
from fastsar import stripmap as sm, patches
img, r, x = patches.form_stripmap(raw, p, rwin='taylor')       # straight track, the grid of focus_stripmap
fx = patches.echoes_to_fx(raw, p, rwin='taylor')               # echoes -> frequency-domain phase history
img = patches.form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2,
                          beam=patches.stripmap_beam(p, ant), backend='cpu')   # antenna positions ant [P, 3]
```

`echoes_to_fx` converts time-domain echoes into the CPHD form with per-pulse reference ranges; `form_mosaic` also
accepts such a phase history directly. For each patch, every pulse is re-referenced to the patch center, its range
profile is kept over a gate that covers the patch plus a margin, and the gated profile is transformed back onto a
frequency grid whose unambiguous range equals the gate. A raised-cosine taper over the guard band shortens the
reconstruction kernel, whose tail is below -60 dB at the default margin (36 m for a 12.5 MHz guard). A pulse
serves a patch when its normalized Doppler satisfies |u| <= umax at some point of the patch. An azimuth window is
split by a singular value decomposition of the pulse-by-pixel weight into separable terms, each formed by one
factorized backprojection. The C++ and CUDA final stages take tiles of 16 or 32 pixels only, so a patch whose
predicted error at T = 16 misses the target is formed on a grid twice as fine in range and decimated; the JAX
program also accepts T = 8. Pass `exact=True` to form each patch by exact backprojection instead, which separates
the error of the patching from that of factorized backprojection.

Test results (`tests/test_patches.py`, the scene of the stripmap test, 128 by 128 pixel patches, 12 patches of
730 to 900 pulses for a 280 by 448 pixel image, about 1 s on four CPU threads). Error against float64
backprojection on 64 by 64 pixel neighborhoods of the five targets:

- broadside: -48.0 to -60.4 dB (CPU), -54.8 to -60.2 dB (JAX, which uses T = 8 where needed)
- 5 degrees of squint: -48.0 to -62.6 dB
- Taylor azimuth window: -48.5 to -58.8 dB with three or four terms per patch
- the same patches by exact backprojection: -65.3 to -66.7 dB without an azimuth window, -54.7 to -67.6 dB with
  the Taylor window

The second part of the test flies the same radar at 3 km altitude on a track with smooth cross-track and
vertical motion (1.68 m and 1.57 m peak to peak), which changes the range to the swath center by up to 0.97 m.
Omega-k on the nominal straight track does not focus these data (error about 0 dB). The patch mosaic with the true
positions, on a ground-plane grid of 0.31 by 1.5 m, matches exact backprojection with the true positions to
-49.9 to -64.9 dB.

Limitations: each pulse enters every patch its beam covers, so longer patches lower the cost per pixel.
`echoes_to_fx` assumes one fast-time grid for all pulses. `stripmap_beam` models an antenna held along a fixed
direction (or one direction per pulse); other patterns can be passed as a function of pulse and point.

## Burst modes (ScanSAR and TOPS)

```python
from fastsar import burst as bm
bs = bm.make_bursts('tops', nburst=2)                   # or 'scansar', subswaths=[(5e3, 200.0), (5.25e3, 200.0)]
raw = bm.simulate(bs[0], [[0.0, bs[0].r0]])              # one burst's echoes
img, r, x = bm.focus_burst(raw, bs[0], algorithm='omegak', rwin='taylor')    # or 'rda'
```

A burst is a `BurstParams`, the stripmap parameters of its pulses plus a steering rate `kpsi`. ScanSAR has
`kpsi = 0`. A ScanSAR burst is padded in azimuth and focused with stripmap omega-k or RDA. A TOPS burst covers a
Doppler band several times the PRF, so it is deramped with the exact integral of its Doppler centroid, windowed,
zero-padded to L times the PRF, reramped on the fine grid, focused, and decimated back. The burst images lie on
the zero-Doppler grid with the complex scale of backprojection of the same pulses, and `mosaic` combines them,
taking each row from the burst that illuminates it with the most pattern energy.

Test results (`tests/test_burst.py`, the radar of the stripmap test):

- TOPS, one burst of 512 pulses steered from -2.70 to +2.70 degrees (band 2.68 times the PRF): omega-k within
  -60.1 to -63.1 dB of backprojection and RDA with secondary range compression within -61.8 to -64.4 dB. Azimuth
  resolution 2.316 to 2.356 m against 2.318 to 2.357 m expected. On four CPU threads omega-k takes 3.7 s and RDA
  1.2 s for the 2940 by 448 pixel burst image.
- ScanSAR, two subswaths at 5.0 and 5.25 km with alternating bursts of 128 pulses: omega-k within -59.0 to
  -68.1 dB of backprojection and RDA within -59.4 to -72.4 dB. Azimuth resolution 1.766 to 2.185 m, within 0.1%
  of the value computed from each target's weighted Doppler support.

Limitations: backprojection cuts the processed band on pulses and the frequency-domain algorithms on Doppler bins,
which matters where data sit close to the band edge, so the ScanSAR targets are simulated and compared one at a
time. The TOPS chain processes L times the pulses of the burst. SPECAN and extended chirp scaling, which avoid
this cost, are not implemented, and the images keep the Doppler centroid variation of TOPS along azimuth.
