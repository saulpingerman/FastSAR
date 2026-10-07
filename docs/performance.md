# Performance

## Time and cost per image

The Umbra Panama image ([real-data.md](real-data.md#umbra-spotlight)) at float32-class accuracy (about -59.5 dB,
[precision.md](precision.md)), on-demand us-central1 prices of October 2026:

| Device | Seconds per image | Dollars per 1000 images |
|---|---|---|
| TPU v6e-1 (three-pass) | 2.4 | 1.89 |
| TPU v5e-1 (three-pass) | 4.6 | 1.62 |
| Nvidia L4 (float32) | 3.6 | 0.76 |
| Nvidia L4 (float16 storage, float32 accumulation) | 2.9 | 0.61 |
| 16-vCPU AMD EPYC 9B45 (c4d-highmem-16) | 12.4 | 3.25 |

Times are for a warm `ImageFormer`; `form_image` adds a few seconds of planning and compilation. Cost is the
hourly price over back-to-back throughput. The v6e is fastest, but the L4 costs 3.8 times less per hour and is the
cheapest. The order holds on Melbourne and Iowa.

![Cost per 1000 Panama images against error for every FastSAR configuration and the open-source implementations](images/teaser.png)

Single-pass TPU products take 1.7 s (v6e) and 2.8 s (v5e) at -45.8 dB, $1.39 and $1.01 per 1000 images. Float16
on the L4 and single-pass on the TPUs lower each device's cost by 19 to 38%. Polar format takes 2.1 s on the L4
($0.42 per 1000) and 11.3 s on the CPU, at -32.1 dB; on the TPUs its final resampling, a scattered gather, takes
97.4 s (v5e) and 105.3 s (v6e). The L4 draws 0.07 kWh per 1000 float32 images (mean `nvidia-smi` power, host
excluded).

## Large collections

Phase histories of tens of gigabytes (28 GB for an ICEYE dwell, 10 GB for a Capella spotlight) are not copied:

- CPU: the first level reads a C-contiguous complex64 history in place; its group holds up to 8 tiles within a
  quarter of physical memory.
- CUDA: a host history larger than 30% of free device memory streams through the first level in blocks of 4,096
  pulses.
- JAX and TPU: up to 16 compiled programs and their plan arrays are cached by plan signature; mosaic patches are
  padded to multiples of 256 pulses to share them (24 formers, 8 programs in `tests/test_patches.py`). Without the
  cache, the 2021 Capella stripmap mosaic on a v6e spent 19 s of 52 s rebuilding plan arrays.
- Mosaics: range profiles are computed once and stay in GPU memory on `cuda` when they fit, with the range gate on
  the GPU. The next patch is prepared on a second thread: 0.91 s per patch against 1.08 s without, on a 2 by 4
  patch Capella sub-mosaic on 16 CPU cores.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `CXX` | `g++` | CPU kernel compiler; builds are cached in `~/.cache/fastsar` |
| `FFBP_CPU_FLAGS` | `-O3 -march=native -mprefer-vector-width=512 -funroll-loops` | compiler flags |
| `FASTSAR_CPU_GROUP_GB` | a quarter of memory | CPU first-level group budget (GB) |
| `FASTSAR_CUDA_STREAM` | above 30% of free GPU memory | `1` always streams |
| `FASTSAR_SHARED_PROFILES` | `1` | `0` re-transforms the history per mosaic patch |
| `FASTSAR_WEIGHT_TERMS` | `0` | `1` applies the mosaic window as SVD terms |
| `FASTSAR_WEIGHT_GRAD` | `1` | `0` drops the weight's variation across a final tile |
| `FASTSAR_MOSAIC_PREFETCH` | `1` | `0` prepares patches one at a time |
| `FASTSAR_TIMING` | off | `1` times mosaic steps; `patches.report_timing()` returns them |

`FASTSAR_FIRP_GLOBAL` and `FFBP_FORCE_TPU_KERNELS` exist for the tests only.

## Where the time goes

As a single JAX program (`backend='jax'`), the float32-class Panama image takes 4.6, 9.3, 13.7 and 217.4 s on the
v6e, v5e, L4 and CPU. The kernels cut that by 1.9 to 2.0 times on the TPUs, 3.8 on the L4 and 17.6 on the CPU.
Every kernel stage runs within 1.0 to 2.5 times a lower bound set by its limiting hardware unit, measured by
microbenchmarks on the same device.

![Time per first-level tile of each stage by implementation form on the TPU v6e and the L4](images/profile.png)

*Time per first-level tile of each stage (phase ramps, rotation, decimation in range and in pulses) by form, on the
TPU v6e (single-pass) and the L4 (TF32), float32 data, in the JAX program.*

Run scripts and records: [sar-accel-study](https://github.com/saulpingerman/sar-accel-study).
