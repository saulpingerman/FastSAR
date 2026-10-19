# Comparison with open-source implementations

Six open-source backprojection codes were run on the Umbra Panama collection, on the c4d-highmem-16 or the L4 (the open-source GPU codes on a g2-standard-8 host,
MatX on the g2-standard-4 of FastSAR's records; [performance.md](performance.md#timing-protocol-and-devices)), from
the same Taylor-windowed phase history and antenna positions.

Errors are against the float64 reference (linear interpolation, 16 times oversampled) over the lock, port and ship
regions of [precision.md](precision.md), worst to best. FastSAR's rows (truth) are against a float64
backprojection at 64 times, since the reference's own error, -67.3 to -68.8 dB, is above that of FastSAR's exact
backprojection; ISCE3 was scored on three 256 by 256 patches of its own grid after removal of a common linear phase
ramp. Times are for the full image, host to host; FastSAR's factorized rows use a warm former. Estimates (est.) scale
one region's time by the pixel count and, if the codes scale as FastSAR does, overstate the full-image time; applied to FastSAR's exact backprojection, the same estimate gives 27
min against 366 s measured.

| CPU (c4d-highmem-16) | Error (dB) | Time |
|---|---|---|
| RITSAR, as published, 16x oversampling | -25.3 to -27.3 | 44 h (est.) |
| AFRL `bpBasic` (Octave), FFT length 2^17 (default) | -38.1 to -43.2 | 57 h (est.) |
| AFRL `bpBasic` (Octave), FFT length 2^18 | -52.9 to -55.4 | 69 h (est.) |
| ISCE3 `backproject`, 16 threads | -31.5 to -54.9 | 4.8 h (est.) |
| GRDL, factorized | no image | 77.5 s |
| FastSAR, exact backprojection, cubic, 8x | -77.5 to -81.7 (truth) | 372 s |
| FastSAR, factorized, float32 | -57.2 to -62.6 (truth) | 11.2 s |

| GPU (Nvidia L4) | Error (dB) | Time |
|---|---|---|
| ISCE3 `backproject`, CUDA | -31.5 to -52.2 | 27 min (est.) |
| torchbp, exact | no image | 16.0 s |
| torchbp, factorized | out of memory | |
| MatX `sar_bp`, `taylor_fast`, as shipped, 8x | -37.7 to -43.2 | 12.5 s |
| MatX `sar_bp`, `taylor_fast`, float64 pixel coordinates, 8x | -54.9 to -58.6 | 11.6 s |
| FastSAR, exact backprojection, linear, 8x | -54.9 to -58.7 | 10.9 s |
| FastSAR, exact backprojection, cubic, 8x | -77.5 to -81.7 (truth) | 16.8 s |
| FastSAR, factorized, float32 | -57.1 to -62.6 (truth) | 3.0 s |
| FastSAR, factorized, float16 throughout | -55.6 to -61.6 (truth) | 2.5 s |

- RITSAR: a `linspace` range axis too coarse by one part in N_fft - 1, and a frequency axis that assumes an even
  sample count (the collection has 14,399).
- `bpBasic` (AFRL MATLAB toolbox, NGA release, in GNU Octave): the error falls 12 to 15 dB per doubling of the FFT
  length, as linear interpolation predicts. RITSAR and `bpBasic` use about one of eight cores.
- ISCE3: amplitude correlation 0.999 or more; without the phase ramp the center patch agrees to -54.9 dB and the
  others keep a residual near -32 dB.
- torchbp computes distance in float32, whose spacing at 744 km is 6 cm, about two wavelengths.
- GRDL's factorized backprojection, documented as a stripmap algorithm, formed no focused image with default
  parameters.
- MatX (NVIDIA's C++ library, `examples/sarbp`, commit 3f3cc13, BSD-3): exact backprojection whose `taylor_fast`
  kernel expands the range to second order within a tile of pixels, with linear interpolation of profiles oversampled
  8 times. As shipped, the example builds the pixel coordinates with a float32 `linspace`; the rounding, about 0.12 mm
  at 2.5 km from the scene center, is a phase error at X band that caps the whole-image error at -33.9 dB. The
  example's own validation compares against its float64 mode, which shares the coordinates, so it does not see this.
  With a float64 `linspace` (one line, `cloud/matx/dvox.patch` in the study repository) the errors are those of
  linear interpolation at 8 times, within 0.1 dB of FastSAR's linear row, which uses the same scheme.

With the algorithm held fixed, FastSAR's exact backprojection (cubic) is 47 times faster than ISCE3's estimated time
on the CPU and 96 times on the L4, with lower errors in every region; with linear interpolation it takes 5% longer
than MatX's corrected form at matched accuracy (12.1 against 11.6 s), about the day-to-day difference between L4
instances. The factorized algorithm cuts the time by a further factor of 5.6 on the L4 and 33 on the CPU.

On the same CPU, the AFRL toolbox's `pfa_mem` (Octave) forms a polar-format image in 44.5 s and GRDL's polar format in 24.5 s.
Neither corrects the planar-wavefront displacement. Their log-amplitude correlations with the reference are 0.80 to
0.89 and 0.78 to 0.93, against 0.996 to 0.999 for FastSAR's corrected polar format (9.0 s on the CPU, 1.4 s on the L4).

Scripts and records: [sar-accel-study](https://github.com/saulpingerman/sar-accel-study), directories `oss` and
`results/comparison`.
