# Performance

## Time and cost per image

One Umbra spotlight image of the Panama Canal (12,207 by 8,808 pixels from 15,186 pulses of 14,399 samples),
factorized backprojection at float32-class accuracy (about -59.5 dB against a float64 exact backprojection, see
[precision.md](precision.md)), on-demand us-central1 prices of October 2026:

| Device | Seconds per image | Dollars per 1000 images |
|---|---|---|
| TPU v6e-1 (three-pass products) | 2.4 | 1.89 |
| TPU v5e-1 (three-pass products) | 4.6 | 1.62 |
| Nvidia L4 (float32) | 3.6 | 0.76 |
| Nvidia L4 (float16 storage, float32 accumulation) | 2.9 | 0.61 |
| 16-vCPU AMD EPYC 9B45 (c4d-highmem-16) | 12.4 | 3.25 |

Times are the image formation on a warm device (an `ImageFormer` call). A one-off `form_image` call adds planning
and compilation, a few seconds on an image of this size. Cost is the hourly price divided by the throughput of a
loop that forms images back to back. The v6e is the fastest device, but its hourly price is 3.8 times that of
the L4, so the L4 is the cheapest at float32-class accuracy. The order is the same on the Melbourne and Iowa
collections.

![Cost per 1000 Panama images against error relative to the float64 image for every FastSAR configuration and the open-source implementations, with the Pareto front](images/teaser.png)

On the TPUs, `precision='single-pass'` runs in 1.7 s (v6e) and 2.8 s (v5e) at -45.8 dB, $1.39 and $1.01 per 1000
images. Float16 storage on the L4 and single-pass products on the TPUs lower each device's cost by 19 to 38%.

Polar format forms the Panama image in 2.1 s on the L4 ($0.42 per 1000) and 11.3 s on the CPU, but at -32.1 dB.
On the TPUs its final resampling, a scattered gather, is slow: 97.4 s on the v5e and 105.3 s on the v6e.

The L4 processor uses 0.07 kWh per 1000 float32 images (mean `nvidia-smi` draw during the timed loop; host
excluded).

## Large collections and mosaics

A long spotlight or a stripmap collection can hold tens of gigabytes of phase history (28 GB for a 91,426-pulse
ICEYE dwell, 10 GB for a Capella spotlight). The formers avoid full-size copies:

- CPU: the first level reads a C-contiguous complex64 history in place, and the first-level group holds up to 8
  tiles within a quarter of the physical memory (`FASTSAR_CPU_GROUP_GB` sets the budget).
- CUDA: a host history larger than 30% of the free device memory is streamed through the first level in blocks of
  4,096 pulses (`FASTSAR_CUDA_STREAM=1` forces it); the later levels work per first-level tile.
- JAX and TPU: `ImageFormer` keeps up to 16 compiled programs, keyed by the plan's signature, with the filters and
  tile geometry that depend on the plan alone. `patches.form_mosaic` pads each patch to a multiple of 256 pulses so
  that patches share programs: in `tests/test_patches.py`, 24 JAX formers compile 8 programs. Before the plan's
  arrays were cached, the 2021 Capella stripmap mosaic on a TPU v6e spent 19 s of 52 s rebuilding them for its 84
  patches.
- Mosaics: the range profiles of the whole history are computed once and each patch is gated from them; on the
  `cuda` backend they stay in GPU memory when they fit, and the range gate runs on the GPU. The next patch's host
  work runs on a second thread while the current patch forms: 0.91 s per patch against 1.08 s without, on a 2 by 4
  patch sub-mosaic of the 2021 Capella stripmap on 16 CPU cores.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `OMP_NUM_THREADS`, `OMP_PLACES`, `OMP_PROC_BIND` | OpenMP's | run one thread per physical core: `<cores>`, `cores`, `close` |
| `CXX` | `g++` | compiler for the CPU kernels, built for the host CPU on first use and cached in `~/.cache/fastsar` |
| `FFBP_CPU_FLAGS` | `-O3 -march=native -mprefer-vector-width=512 -funroll-loops` | compiler flags for the factorized kernels |
| `FASTSAR_CPU_GROUP_GB` | a quarter of physical memory | memory budget (GB) of the CPU first-level group |
| `FASTSAR_CUDA_STREAM` | stream above 30% of free GPU memory | `1` streams every host history |
| `FASTSAR_SHARED_PROFILES` | `1` | `0` re-transforms the full history for every mosaic patch |
| `FASTSAR_WEIGHT_TERMS` | `0` | `1` applies a mosaic's azimuth window as separable SVD terms, one factorized backprojection each |
| `FASTSAR_WEIGHT_GRAD` | `1` | `0` drops the first-order variation of the aperture weight across a final tile |
| `FASTSAR_MOSAIC_PREFETCH` | `1` | `0` prepares mosaic patches one at a time |
| `FASTSAR_TIMING` | off | `1` times each mosaic step; `patches.report_timing()` returns the totals as text |

`FASTSAR_FIRP_GLOBAL` and `FFBP_FORCE_TPU_KERNELS` exist for the tests only.

## Where the time goes

The device kernels were written after a stage-by-stage profile of the plain JAX program (`backend='jax'`). Run
as a single JAX program, the factorized algorithm takes 4.6, 9.3, 13.7 and 217.4 s for the float32-class Panama
image on the v6e, v5e, L4 and CPU. The kernels cut the device time by factors of 1.9 to 2.0 on the TPUs, 3.8 on
the L4 and 17.6 on the CPU. Every stage of the kernels runs within 1.0 to 2.5 times a lower bound set by the
hardware unit that limits it, measured by microbenchmarks on the same device.

![Time per first-level tile of each stage of the factorized algorithm, by implementation form, on the TPU v6e and the Nvidia L4, on a logarithmic scale](images/profile.png)

*Time per first-level tile of each stage (phase ramps, rotation, decimation in range and in pulses) by form, on
the TPU v6e (single-pass products) and the L4 (TF32 products), with float32 data, in the JAX program.*

The cloud run scripts and the measurement records are in the
[sar-accel-study](https://github.com/saulpingerman/sar-accel-study) repository.
