# Comparison with open-source implementations

Five open-source backprojection codes were run on the Umbra Panama collection, on the c4d-highmem-16 or the L4
([performance.md](performance.md#timing-protocol-and-devices)), from the same Taylor-windowed phase history and
antenna positions. Two produced no usable image.

Errors are against float64 exact backprojection over the lock, port and ship regions of [precision.md](precision.md),
worst to best; ISCE3 was scored on three 256 by 256 patches of its own grid after removal of a common linear phase
ramp. Times are for the full image, host to host; FastSAR's factorized rows use a warm former. Estimates (est.) scale
one region's time by the pixel count and overstate it: the same estimate for FastSAR's exact backprojection gives 2.4
h against 46 min measured.

| CPU (c4d-highmem-16) | Error (dB) | Time |
|---|---|---|
| RITSAR, as published, 16x oversampling | -25.3 to -27.3 | 44 h (est.) |
| AFRL `bpBasic` (Octave), FFT length 2^17 (default) | -38.1 to -43.2 | 57 h (est.) |
| AFRL `bpBasic` (Octave), FFT length 2^18 | -52.9 to -55.4 | 69 h (est.) |
| ISCE3 `backproject`, 16 threads | -31.5 to -54.9 | 4.8 h (est.) |
| GRDL, factorized | no image | 77.5 s |
| FastSAR, exact backprojection, 8x | -54.9 to -58.7 | 46 min |
| FastSAR, factorized, float32 | -55.8 to -61.2 | 12.4 s |

| GPU (Nvidia L4) | Error (dB) | Time |
|---|---|---|
| ISCE3 `backproject`, CUDA | -31.5 to -52.2 | 27 min (est.) |
| torchbp, exact | no image | 16.0 s |
| torchbp, factorized | out of memory | |
| FastSAR, exact backprojection, 8x | -54.7 to -58.3 | 67 s |
| FastSAR, factorized, float32 | -55.7 to -61.2 | 4.3 s |
| FastSAR, factorized, float16 | -54.6 to -60.4 | 3.7 s |

- RITSAR: a `linspace` range axis too coarse by one part in N_fft - 1, and a frequency axis that assumes an even
  sample count (the collection has 14,399).
- `bpBasic` (AFRL MATLAB toolbox, NGA release, in GNU Octave): the error falls 12 to 15 dB per doubling of the FFT
  length, as linear interpolation predicts. RITSAR and `bpBasic` use about one of eight cores.
- ISCE3: amplitude correlation 0.9995 or more; without the phase ramp the center patch agrees to -54.9 dB and the
  others keep a residual near -32 dB.
- torchbp computes distance in float32, whose spacing at 744 km is 6 cm, about two wavelengths.
- GRDL's factorized backprojection, documented as a stripmap algorithm, formed no focused image with default
  parameters.

With the algorithm held fixed, FastSAR's exact backprojection is 6 times faster than ISCE3's estimated time on the
CPU and 24 times faster on the L4. The factorized algorithm cuts the CPU time by a further factor of 220.

On the same CPU, NGA's `pfa_mem` (Octave) forms a polar-format image in 44 s and GRDL's polar format in 24.5 s.
Neither corrects the planar-wavefront displacement. Their log-amplitude correlations with the reference are 0.80 to
0.89 and 0.78 to 0.93, against 0.996 to 0.999 for FastSAR's corrected polar format (11.3 s).

Scripts and records: [sar-accel-study](https://github.com/saulpingerman/sar-accel-study), directories `oss` and
`results/oss`.
