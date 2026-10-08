# Performance

## Timing protocol and devices

Times run from the phase history in host memory to the image in host memory, transfers included, with a warm
`ImageFormer` (planned and compiled beforehand). The other pages call this "host to host, warm former".
`form_image` adds a few seconds of planning and compilation. The devices are a TPU v6e and a TPU v5e (one chip
each), an Nvidia L4, and a c4d-highmem-16 CPU instance (AMD EPYC 9B45, 16 vCPUs on 8 cores).

## Time and cost per image

The Umbra Panama image ([real-data.md](real-data.md#umbra-spotlight)) at float32-class accuracy (about -59.5 dB,
[precision.md](precision.md)), on-demand us-central1 prices of October 2026:

| Device | Seconds per image | Dollars per 1000 images |
|---|---|---|
| TPU v6e (three-pass) | 3.1 | 1.89 |
| TPU v5e (three-pass) | 5.6 | 1.62 |
| L4 (float32) | 4.3 | 0.75 |
| L4 (float16 storage, float32 accumulation) | 3.7 | 0.62 |
| c4d-highmem-16 | 12.4 | 3.25 |

Cost is the hourly price divided by the throughput of back-to-back images, with uploads overlapping formation. The
v6e is fastest. The L4, at about a quarter of the v6e's hourly price, is cheapest per image. Both rankings hold on
Melbourne and Iowa.

![Cost per 1000 Panama images against error for every FastSAR configuration and the open-source implementations](images/teaser.png)

Single-pass TPU products take 2.4 s (v6e) and 3.8 s (v5e) at -45.8 dB, $1.39 and $1.01 per 1000 images. Float16
on the L4 and single-pass on the TPUs lower each device's cost by 18 to 38%. Polar format takes 2.4 s on the L4
($0.42 per 1000) and 11.3 s on the c4d-highmem-16, at -32.1 dB. On the TPUs its final resampling, a scattered
gather, adds about 94 s (v5e) and 104 s (v6e), for 98 and 106 s per image. The L4 draws 0.07 kWh per 1000 float32
images (mean `nvidia-smi` power, host excluded).

## Large collections

Phase histories of tens of gigabytes (28 GB for an ICEYE dwell, 10 GB for a Capella spotlight) are handled as
follows. The CPU and CUDA backends do not copy them; the TPU backend copies the history once per image to pad it.

- CPU: the first level reads a C-contiguous complex64 history in place. A group holds up to 8 first-level children
  within a quarter of the available host memory. When fewer than 4 fit, the first level runs in blocks of 2,048
  pulses instead: 14.7 s against 12.4 s on Panama (19% slower). On a Capella spotlight, groups of 2 and 1 took
  58% and 140% longer than groups of 8.
- CUDA: a host history whose device planes exceed half the free device memory streams through the first level in
  blocks of 4,096 pulses.
- TPU: the history is padded once per image for the first-level kernel. A level's pulse decimation is a dense
  matrix product up to 2^24 matrix entries; above that, the TPU applies the filter in banded blocks. A dense matrix
  for the 74,203-pulse spotlight would take 2.8 GB, and its cost grows as the square of the pulse count.

## Compiled programs

The JAX and TPU backends cache up to 32 compiled programs and their plan arrays by plan signature. Before forming,
a mosaic finds each patch's pulse span and picks a few shared pulse counts, so patches are padded up to a count
that already has a compiled program. `FASTSAR_COMPILE_PATCHES` sets how many patch formations one compilation is
worth: 11 on a TPU, where a program compiles in about 8 s and a dynamic stripmap patch forms in 0.7 s, and 3 on
JAX. Range gates are planned the same way, and gates and pulse counts outside the plan are widened by up to
`FASTSAR_PULSE_SLACK` (0.04). The 78-patch 2022 Capella dynamic stripmap mosaic compiles one program instead of 38.
Each patch's history is checked, scaled and uploaded on a worker thread (`ImageFormer.stage`). Without the program
cache, the 2021 Capella stripmap mosaic on a v6e spent 19 s of 52 s rebuilding plan arrays.

In a mosaic the range profiles are computed once and stay in GPU memory on `cuda` when they fit, with the range gate
on the GPU. While one patch forms, worker threads prepare the next ones. The CPU uses one thread, since its
formation already uses every core; a GPU or TPU uses up to four, which would otherwise wait about 0.5 s per patch
for host preparation. On a 2 by 4 patch Capella sub-mosaic on the c4d-highmem-16, one prefetch thread cuts the
time per patch from 1.08 s to 0.91 s.

## Memory

When memory is short, FastSAR falls back to a slower path instead of failing and issues a `fastsar.MemoryWarning`.
`former.memory()` reports the requirement in advance, in about 15 µs.

Full speed keeps the phase history on the device and forms the first level in groups of 8 children (4 on a TPU).
On a v6e, groups of 4 and 8 formed the 2025 Capella spotlight equally fast, and groups of 2 took 1% longer. On the
L4 the same spotlight took 17.9 s in groups of 8. It took 2% longer in groups of 4 and 44% longer with the history
streamed from host memory. The 2025 spotlight fits the L4's 24 GB at full speed; the 74,203-pulse 2024 spotlight
does not. The number of first-level children carried together through the later levels (up to half the free
memory) does not affect speed. Batches of 1, 2 and 4 formed that spotlight within 1% of each other, so FastSAR does
not count a smaller batch as a fallback.

The fallbacks are smaller groups or pulse blocks on the CPU, smaller groups on GPU and TPU, and streaming of the
history through the first level on CUDA. A CUDA mosaic keeps a GPU window of the range profiles and cuts the gates in
host memory when a patch's profiles exceed it. Each fallback that FastSAR chooses issues a `MemoryWarning` that names
it; the warnings issued before forming also give the bytes needed for full speed and the bytes free. A size set by
`FASTSAR_CPU_GROUP_GB`, `FASTSAR_CUDA_GROUP`, `FASTSAR_TPU_GROUP` or an explicit `ng` issues none. A TPU that cannot
hold the history and one first-level child raises `MemoryError` with the memory those two need.

```python
import warnings
former = fastsar.ImageFormer(...)
former.memory()     # bytes: {'backend': 'cuda', 'needed': ..., 'available': ..., 'full_speed': True, 'parts': {...}}
warnings.simplefilter('ignore', fastsar.MemoryWarning)   # silence the fallback warnings
```

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `CXX` | `g++` | CPU kernel compiler; builds are cached in `~/.cache/fastsar` |
| `FFBP_CPU_FLAGS` | `-O3 -march=native -mprefer-vector-width=512 -funroll-loops` (exact backprojection: `-O3 -march=native`) | compiler flags |
| `FASTSAR_CPU_GROUP_GB` | a quarter of available memory | CPU first-level group budget (GB) |
| `FASTSAR_CUDA_GROUP` | from free memory: 8 at full speed | CUDA first-level children per group |
| `FASTSAR_TPU_GROUP` | from the device's memory: 4 at full speed | TPU first-level children per group |
| `FASTSAR_CUDA_STREAM` | above 50% of free GPU memory | `1` always streams |
| `FASTSAR_SHARED_PROFILES` | `1` | `0` re-transforms the history per mosaic patch |
| `FASTSAR_WEIGHT_TERMS` | `0` | `1` applies the mosaic window as SVD terms |
| `FASTSAR_WEIGHT_GRAD` | `1` | `0` drops the weight's variation across a final tile |
| `FASTSAR_MOSAIC_PREFETCH` | `1` (CPU), `4` (GPU, TPU; at most a quarter of the CPU's threads) | worker threads preparing the next patches; `0` prepares them one at a time |
| `FASTSAR_COMPILE_PATCHES` | `11` (TPU), `3` (JAX) | patch formations one compilation is worth, for the pulse counts a mosaic compiles |
| `FASTSAR_PULSE_SLACK` | `0.04` | relative widening of mosaic range gates and pulse counts so patches share compiled programs |
| `FASTSAR_TIMING` | off | `1` times mosaic steps; `patches.report_timing()` returns them |

Numeric settings out of range raise `ValueError`. The `FFBP_` prefix is a legacy name. `FASTSAR_CPU_BLOCKED` (`1`
forces the CPU's pulse blocks, `0` forbids them), `FASTSAR_PROFILE_WINDOW_ROWS`, `FASTSAR_FIRP_GLOBAL`,
`FFBP_FORCE_TPU_KERNELS` and `FASTSAR_TEST_MAX_GB` (the default of `tests/run_all.py --max-rss-gb`) exist for the
tests only.

## Where the time goes

As a single JAX program (`backend='jax'`), the float32-class Panama image takes 5.3, 10.3, 15.0 and 217.4 s on the
v6e, v5e, L4 and c4d-highmem-16. The kernels cut that by 1.7 to 1.8 times on the TPUs, 3.4 on the L4 and 17.6 on
the CPU. Every kernel stage runs within 1.0 to 2.5 times a lower bound set by its limiting hardware unit, measured
by microbenchmarks on the same device.

![Time per first-level tile of each stage by implementation form on the TPU v6e and the L4](images/profile.png)

*Time per first-level tile of each stage (phase ramps, rotation, decimation in range and in pulses) by form, on the
v6e (single-pass) and the L4 (float32 data, TF32 tensor-core matrix products), in the JAX program.*

Run scripts and records: [sar-accel-study](https://github.com/saulpingerman/sar-accel-study).
