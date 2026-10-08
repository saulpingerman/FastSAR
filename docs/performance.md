# Performance

## Time and cost per image

The Umbra Panama image ([real-data.md](real-data.md#umbra-spotlight)) at float32-class accuracy (about -59.5 dB,
[precision.md](precision.md)), on-demand us-central1 prices of October 2026:

| Device | Seconds per image | Dollars per 1000 images |
|---|---|---|
| TPU v6e-1 (three-pass) | 3.1 | 1.89 |
| TPU v5e-1 (three-pass) | 5.6 | 1.62 |
| Nvidia L4 (float32) | 4.3 | 0.75 |
| Nvidia L4 (float16 storage, float32 accumulation) | 3.7 | 0.62 |
| 16-vCPU AMD EPYC 9B45 (c4d-highmem-16) | 12.4 | 3.25 |

Times are for a warm `ImageFormer`, from the phase history in host memory to the image in host memory (transfers
included); `form_image` adds a few seconds of planning and compilation. Cost is the hourly price over back-to-back
throughput, where uploads overlap formation. The v6e is fastest, but the L4 costs 3.8 times less per hour and is the
cheapest. The order holds on Melbourne and Iowa.

![Cost per 1000 Panama images against error for every FastSAR configuration and the open-source implementations](images/teaser.png)

Single-pass TPU products take 2.4 s (v6e) and 3.8 s (v5e) at -45.8 dB, $1.39 and $1.01 per 1000 images. Float16
on the L4 and single-pass on the TPUs lower each device's cost by 18 to 38%. Polar format takes 2.4 s on the L4
($0.42 per 1000) and 11.3 s on the CPU, at -32.1 dB; on the TPUs its final resampling, a scattered gather, takes
about 94 s (v5e) and 106 s (v6e). The L4 draws 0.07 kWh per 1000 float32 images (mean `nvidia-smi` power, host
excluded).

## Large collections

Phase histories of tens of gigabytes (28 GB for an ICEYE dwell, 10 GB for a Capella spotlight) are not copied:

- CPU: the first level reads a C-contiguous complex64 history in place; its group holds up to 8 tiles within a
  quarter of physical memory.
- CUDA: a host history larger than 30% of free device memory streams through the first level in blocks of 4,096
  pulses.
- TPU: the history is padded once per image for the first-level kernel, and a level's pulse decimation is a
  blocked banded product of its filter kernel once the dense matrix would exceed 2^24 entries (2.8 GB at 74,203
  pulses, with work growing as the square of the pulse count).
- JAX and TPU: up to 32 compiled programs and their plan arrays are cached by plan signature. A mosaic first finds
  every patch's pulse span and chooses the pulse counts of its programs for the whole mosaic, trading a program's
  compile time against the padding of the patches that share it (`FASTSAR_COMPILE_PATCHES`, the compile time in
  patch formations, default 11 on a TPU); range gates are widened onto shared lengths (`FASTSAR_PULSE_SLACK`, 4%).
  A 78-patch Capella sliding spotlight compiles 2 programs instead of 38, where a v6e spends about 8 s per program.
  Each patch's history is checked, scaled and uploaded on a worker thread (`ImageFormer.stage`). Without the
  program cache, the 2021 Capella stripmap mosaic on a v6e spent 19 s of 52 s rebuilding plan arrays.
- Mosaics: range profiles are computed once and stay in GPU memory on `cuda` when they fit, with the range gate on
  the GPU. The next patches are prepared on worker threads while one forms (one on the CPU, whose formation uses
  every core: 0.91 s per patch against 1.08 s without, on a 2 by 4 patch Capella sub-mosaic on 16 cores; up to
  four for a GPU or TPU, which otherwise wait on the host's 0.2 to 0.5 s of preparation per patch).

## Memory

Full speed keeps the phase history on the device and forms the first level in groups of 8 children (4 on a TPU,
where groups of 4 and 8 formed the 2025 Capella spotlight equally fast and groups of 2 took 1% longer). On the L4
the same spotlight took 18.3 s in groups of 8, 2% longer in groups of 4, 6% in groups of 2 and 44% longer with the
history streamed from host memory; it runs at full speed in the L4's 24 GB, the 74,203-pulse 2024 spotlight does not.
With less memory FastSAR falls back instead of failing: smaller groups on CPU, GPU and TPU (the history is read more
often), streaming of the history from host memory through the first level on CUDA, and shared range profiles kept
on the host in a CUDA mosaic. Each fallback issues a `fastsar.MemoryWarning` naming it, with the memory full speed
needs and the memory free. A TPU that cannot hold the history raises `MemoryError` with the same estimate.
`former.memory()` reports the requirement before running (15 us; the free-memory query costs 13 us per image).

```python
former = fastsar.ImageFormer(...)
former.memory()     # {'backend': 'cuda', 'needed': ..., 'available': ..., 'full_speed': True, 'parts': {...}} (bytes)
warnings.simplefilter('ignore', fastsar.MemoryWarning)   # silence the fallback warnings
```

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
| `FASTSAR_MOSAIC_PREFETCH` | 1 on the CPU, up to 4 on a GPU or TPU | worker threads preparing the next patches; `0` prepares them one at a time |
| `FASTSAR_COMPILE_PATCHES` | 11 on a TPU, 3 on JAX | a program's compile time in patch formations, for the pulse counts a mosaic compiles |
| `FASTSAR_PULSE_SLACK` | `0.04` | widening of mosaic range gates (and of pulse counts not planned) to share programs |
| `FASTSAR_TIMING` | off | `1` times mosaic steps; `patches.report_timing()` returns them |

`FASTSAR_FIRP_GLOBAL` and `FFBP_FORCE_TPU_KERNELS` exist for the tests only.

## Where the time goes

As a single JAX program (`backend='jax'`), the float32-class Panama image takes 5.3, 10.3, 15.0 and 217.4 s on the
v6e, v5e, L4 and CPU. The kernels cut that by 1.7 to 1.8 times on the TPUs, 3.4 on the L4 and 17.6 on the CPU.
Every kernel stage runs within 1.0 to 2.5 times a lower bound set by its limiting hardware unit, measured by
microbenchmarks on the same device.

![Time per first-level tile of each stage by implementation form on the TPU v6e and the L4](images/profile.png)

*Time per first-level tile of each stage (phase ramps, rotation, decimation in range and in pulses) by form, on the
TPU v6e (single-pass) and the L4 (TF32), float32 data, in the JAX program.*

Run scripts and records: [sar-accel-study](https://github.com/saulpingerman/sar-accel-study).
