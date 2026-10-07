# Comparison with open-source implementations

FastSAR was compared with five open-source SAR image formation codes on the Umbra Panama collection (12,207 by
8,808 pixels from 15,186 pulses). Every CPU implementation ran on the same c4d-highmem-16 instance (AMD EPYC 9B45,
8 cores, 16 vCPUs) and every GPU implementation on the same Nvidia L4. Each received the same Taylor-windowed phase
history and antenna positions in its own input format.

Errors are relative to a float64 exact backprojection, over the lock, port and ship regions of
[precision.md](precision.md), worst to best. ISCE3 was scored instead on three 256 by 256 pixel patches of its own
grid, after removal of a common linear phase ramp. Times are for the full image. Those marked est. scale the time
on one region by the number of pixels and overstate the true time: the same estimate for FastSAR's exact
backprojection gives 2.4 h against 46 min measured.

**CPU (c4d-highmem-16)**

| Implementation | Error (dB) | Time per image |
|---|---|---|
| RITSAR, as published, 16x oversampling | -25.3 to -27.3 | 44 h (est.) |
| AFRL `bpBasic` (Octave), default FFT length 2^17 | -38.1 to -43.2 | 57 h (est.) |
| AFRL `bpBasic` (Octave), FFT length 2^18 | -52.9 to -55.4 | 69 h (est.) |
| ISCE3 `backproject`, 16 threads | -31.5 to -54.9 | 4.8 h (est.) |
| GRDL, factorized | no image | 77.5 s |
| FastSAR, exact backprojection, 8x | -54.9 to -58.7 | 46 min |
| FastSAR, factorized, float32 | -55.8 to -61.2 | 12.4 s |

**GPU (Nvidia L4)**

| Implementation | Error (dB) | Time per image |
|---|---|---|
| ISCE3 `backproject`, CUDA | -31.5 to -52.2 | 27 min (est.) |
| torchbp, exact | no image | 15.2 s |
| torchbp, factorized | does not fit in memory | |
| FastSAR, exact backprojection, 8x | -54.7 to -58.3 | 67 s |
| FastSAR, factorized, float32 | -55.7 to -61.2 | 3.6 s |
| FastSAR, factorized, float16 | -54.6 to -60.4 | 2.9 s |

![Cost per 1000 Panama images against error for FastSAR configurations and the open-source implementations](images/teaser.png)

Notes on each implementation:

- **RITSAR**: the error comes from two indexing defects, a range axis built with `linspace` whose spacing is too
  large by one part in N_fft - 1, and a frequency axis centered on the assumption of an even sample count, which
  the 14,399 samples of the collection do not satisfy.
- **`bpBasic`** (backprojection of the AFRL MATLAB SAR image formation toolbox, NGA release, run in GNU Octave): the error falls by 12 to
  15 dB per doubling of its FFT length, as linear interpolation of the range profiles predicts. RITSAR and
  `bpBasic` do not parallelize their backprojection and kept about one of the eight cores busy.
- **ISCE3**: its amplitude correlates with the reference at 0.9995 or more. After removal of the common phase
  ramp the center patch agrees to -54.9 dB and the off-center patches keep a residual near -32 dB.
- **torchbp**: it evaluates the pixel-to-antenna distance in float32, whose spacing at the 744 km range of the
  collection is 6 cm, about two wavelengths, so its exact backprojection forms no image. Its factorized
  backprojection needed more than the L4's memory.
- **GRDL**: its factorized backprojection, which its documentation describes as a stripmap algorithm, ran with
  default parameters and formed no focused image of this spotlight collection.

With the algorithm held fixed, FastSAR's exact backprojection is 6 times faster than ISCE3's estimated time on the
CPU and 24 times faster on the L4. The factorized algorithm then reduces the CPU time by a further factor of 220.

## Polar format

On the same CPU instance, the NGA `pfa_mem` (in Octave) forms the full image in 44 s and GRDL's polar format in
24.5 s. Neither corrects the planar-wavefront displacement. Their log-amplitude correlations with the reference
are 0.80 to 0.89 and 0.78 to 0.93, against 0.996 to 0.999 for FastSAR's corrected polar format, which takes
11.3 s.

The scripts and records of every comparison are in the
[sar-accel-study](https://github.com/saulpingerman/sar-accel-study) repository, directories `oss` and
`results/oss`.
