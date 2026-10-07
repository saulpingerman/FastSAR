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

## Running on the CPU

Run one OpenMP thread per physical core:

```bash
OMP_NUM_THREADS=<cores> OMP_PLACES=cores OMP_PROC_BIND=close python my_script.py
```

The kernels are compiled for the host CPU (`-march=native`) on first use. `CXX` selects another compiler and
`FFBP_CPU_FLAGS` replaces the flags.

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
