# Precision and image quality

## Precision options

`form_image` and `ImageFormer` take `precision=`:

- `'float32'` (default): float32 data and products. On a TPU this means three bfloat16 passes per product.
- `'float16'`, CUDA only: float16 phase history and intermediates with float32 accumulation, and the tensor-core
  final stage. Needs `T=32`.
- `'single-pass'`, `'three-pass'`: the TPU's one- and three-pass bfloat16 products. Single-pass is the TPU's own
  default for matrix products; FastSAR uses three-pass unless asked.

The geometry is evaluated in float64 on the host; only offsets within tiles are evaluated in reduced precision.

## Measured on the Umbra Panama collection

All errors are relative to a float64 exact backprojection of the same collection (range profiles oversampled 16
times), after one fitted gain, as the energy of the difference over the whole 12,207 by 8,808 pixel image.

| Configuration | Error (dB) |
|---|---|
| CPU, float32 | -59.6 |
| Nvidia L4, float32 | -59.5 |
| Nvidia L4, float16 | -58.6 |
| TPU v5e, three-pass | -57.6 |
| TPU v6e, three-pass | -57.5 |
| TPU v5e and v6e, single-pass | -45.8 |
| Polar format with the displacement correction (all devices) | -32.1 |

- The float32-class images (float32 on the CPU and L4, three-pass on the TPUs) leave the amplitude visually
  unchanged, and their 5 by 5 coherence with the reference exceeds 0.999 even at its 0.1 percentile. The L4 and
  CPU images are within 0.1 dB of a float64 factorized image. The TPU kernels compute the last level's geometry in
  float32, the probable cause of their 2 dB larger error.
- Float16 storage on the L4 adds 0.9 dB and keeps the image in the same group.
- Single-pass TPU products add 12 dB. The amplitude image is unchanged, but the phase deviates by up to 86
  degrees beside bright returns, against a 99th percentile of 1.0 degree. That matters more for interferometric
  products than for amplitude images.
- Polar format shows striped coherence below 0.99. Its residual grows from -40 dB at the scene center to -30 dB in
  the outer quarter ([algorithms.md](algorithms.md#polar-format)).
- Exact backprojection at the default eightfold oversampling is 0.8 to 4.9 dB farther from the reference than
  the float32 factorized image on the three regions below; its error is set by the range interpolation.

On the Melbourne and Iowa collections the errors of the same factorized configurations are within 1.4 dB of the
Panama values.

![Four 512 by 512 pixel regions of the Panama Canal image (locks, port, ship, vegetation) formed by the float64 reference, the L4 exact backprojection, the L4 float16 factorized image, the TPU v6e with single-pass and three-pass products, and polar format, each with its coherence map against the reference](images/zoom_panama.png)

*Four regions of 512 by 512 pixels at full resolution (100 m bar): the Cocolí Locks, the Port of Balboa, a ship in
the channel and vegetated terrain 2.7 km from the scene center. Top of each pair: amplitude over 45 dB. Bottom: 5
by 5 coherence with the reference from 0.99 (red) to 1 (white), mean below each panel. Coherence loss shows only
beside bright returns in the single-pass TPU images and in stripes in the polar-format image.*

![128 by 128 pixel windows around the brightest features of the lock, port and ship regions, drawn without interpolation, with coherence maps](images/pixels_panama.png)

*The same comparison at pixel scale: 128 by 128 pixel windows (53 m in azimuth by 46 m in slant range, pixels of
0.41 by 0.36 m) around the brightest feature of the lock, port and ship regions, drawn without interpolation.*
