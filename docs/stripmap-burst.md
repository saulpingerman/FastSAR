# Stripmap, long apertures and burst modes

`fastsar.stripmap` focuses a straight track by omega-k or the range-Doppler algorithm (RDA). `fastsar.patches`
forms mosaics of factorized backprojection patches for any track; `form_cphd` uses it for moving beams.
`fastsar.burst` handles ScanSAR and TOPS. Numbers are simulated test results unless Capella is named.

## Stripmap

```python
import numpy as np
from fastsar import stripmap as sm, quality
p = sm.make_params(squint_deg=5.0)
raw = sm.simulate(p, [[0.0, p.r0], [20.0, p.r0 + 80.0]])    # targets at (x, zero-Doppler range)
img, r, x = sm.focus_stripmap(raw, p, algorithm='omegak', rwin='taylor')    # or 'rda', 'bp'
ij = np.unravel_index(np.abs(img).argmax(), img.shape)
m = quality.point_target(img, ij, d_az=x[1] - x[0], d_rg=r[1] - r[0])     # resolution, PSLR, ISLR, peak
```

The simulator is exact in float64. Omega-k uses the exact reference phase and Stolt mapping; RDA corrects the
exact migration by sinc interpolation, with optional secondary range compression (SRC). Both run in JAX, float32 by
default. Float64 backprojection (`'bp'`) is the reference; all three share the zero-Doppler grid and complex scale.

Test results (`tests/test_stripmap.py`: 9.6 GHz, 100 MHz, PRF 650 Hz, 200 m/s, 5 km, five targets), error against
backprojection on 64 by 64 pixel patches:

- broadside: omega-k -64.9 to -65.5 dB; RDA -66.5 to -67.7 dB with SRC, -65.4 to -66.3 dB without
- 5 degree squint (Doppler centroid 1116 Hz, 1.7 times the PRF): omega-k -60.2 to -62.5 dB; RDA -61.1 to -67.4 dB
  with SRC, -27.4 to -27.7 dB without

At broadside all three measure 0.585 m in azimuth and 1.79 m in range (0.585 and 1.775 m expected), PSLR -39.6 dB
(azimuth) and -32.7 dB (range), peaks within 1.5 mm. Float32 and float64
omega-k differ by -125 dB. On four CPU threads omega-k takes 0.3 s and RDA 0.1 s.

## Long apertures and arbitrary tracks

```python
from fastsar import patches
img, r, x = patches.form_stripmap(raw, p, rwin='taylor')       # straight track, the grid of focus_stripmap
fx = patches.echoes_to_fx(raw, p, rwin='taylor')               # echoes -> frequency-domain phase history
img = patches.form_mosaic(fx, ant, origin, nx, ny, spx, spy, e1, e2,
                          beam=patches.stripmap_beam(p, ant))  # antenna positions ant [P, 3], any track
```

An `ImageFormer` forms each patch from the pulses whose beam covers it (|u| <= `umax`). Each pulse's range profile,
computed once, is re-referenced to the patch center, gated to the patch plus a margin, and transformed back to
frequency. A raised-cosine taper keeps the kernel's tail below -60 dB at the default margin (36 m for a 12.5 MHz
guard). The azimuth window enters the final stage per subaperture and tile, with its first-order variation across the
tile (the JAX program instead evaluates its final stage three times). C++ and CUDA take tiles of 16 or 32 only, so a
patch that misses the target at T=16 is formed on a finer grid (2 or 4 times, range first) and decimated; JAX also
accepts T=8. `exact=True` uses exact backprojection, with the window split into separable SVD terms.

Test results (`tests/test_patches.py`, 12 patches of 128 by 128 pixels, about 1 s on four CPU threads), error
against float64 backprojection near the targets:

- broadside: -48.0 to -60.4 dB (CPU), -54.7 to -63.4 dB (JAX, T=8 where needed); 5 degree squint: -47.9 to -62.6 dB
- Taylor azimuth window: -49.4 to -61.2 dB (CPU), -55.9 to -62.5 dB (JAX)
- exact backprojection per patch: -65.3 to -66.7 dB, or -54.7 to -67.6 dB with the Taylor window
- a 3 km altitude track with 1.68 m cross-track and 1.57 m vertical motion: omega-k on the nominal track fails
  (about 0 dB); the mosaic matches exact backprojection to -49.9 to -65.0 dB

On a 4-patch 2021 Capella stripmap sub-mosaic, the final-stage window (instead of SVD terms) improved the seam
error from -59.9 to -65.7 dB. It also cut the time per patch on the c4d-highmem-16 from 4.3 s to about 1.8 s. Shared
range profiles made patch histories 7 times faster ([switches](performance.md#environment-variables)).

`echoes_to_fx` assumes one fast-time grid. `stripmap_beam` models an antenna
held along a fixed direction; `beam` can be any function of pulse and point.

## Burst modes (ScanSAR and TOPS)

```python
from fastsar import burst as bm
bs = bm.make_bursts('tops', nburst=2)                   # or 'scansar', subswaths=[(5e3, 200.0), (5.25e3, 200.0)]
raw = bm.simulate(bs[0], [[0.0, bs[0].r0]])
img, r, x = bm.focus_burst(raw, bs[0], algorithm='omegak', rwin='taylor')    # or 'rda'
```

A ScanSAR burst (`kpsi = 0`) is padded in azimuth and focused as stripmap. A TOPS burst spans several PRFs of
Doppler: it is deramped with the exact integral of its Doppler centroid, zero-padded to L times the PRF, reramped,
focused and decimated. Burst images share backprojection's zero-Doppler grid and complex scale; `mosaic` takes each
row from the burst that illuminates it most.

Test results (`tests/test_burst.py`):

- TOPS, 512 pulses steered over +/-2.70 degrees (2.68 PRFs): omega-k -60.1 to -63.1 dB, RDA with SRC -61.8 to
  -64.4 dB; azimuth resolution 2.316 to 2.356 m (2.318 to 2.357 m expected) for a 2940 by
  448 image
- ScanSAR, subswaths at 5.0 and 5.25 km, bursts of 128 pulses: omega-k -59.0 to -68.1 dB, RDA -59.4 to -72.4 dB;
  azimuth resolution 1.766 to 2.185 m, within 0.1% of each target's weighted Doppler support

ScanSAR targets are compared one at a time, since backprojection cuts the band on pulses and the other algorithms
on Doppler bins. TOPS processes L times the burst's pulses (SPECAN and extended chirp scaling are not implemented)
and keeps its Doppler centroid variation.
