"""Emulated 8-bit and 4-bit float operands for matrix products.

Tensor cores on recent GPUs take FP8 (E4M3) and FP4 (E2M1) operands and
accumulate in wider precision. Rounding the operands to those formats and then
multiplying in float32 reproduces that arithmetic exactly, so image quality
can be measured without the hardware.
"""
import jax.numpy as jnp

FORMATS = {
    # mantissa bits, minimum normal exponent, largest finite value
    'f8': (3, -6, 448.0),      # E4M3
    'f8e5': (2, -14, 57344.0),  # E5M2
    'f4': (1, 0, 6.0),         # E2M1
}


def _round_minifloat(x, mbits, emin, vmax):
    a = jnp.abs(x)
    e = jnp.floor(jnp.log2(jnp.maximum(a, 1e-30)))
    e = jnp.maximum(e, emin)                       # subnormals share the lowest exponent
    step = jnp.exp2(e - mbits)
    q = jnp.round(a / step) * step
    return jnp.sign(x) * jnp.minimum(q, vmax)


def quantize(x, fmt, block=16):
    """Round a float32 array to `fmt` and return it as float32.

    FP8 uses one scale per tensor that maps the peak to the format's maximum.
    FP4 has almost no dynamic range, so it gets one scale per `block` values
    along the last axis, as block-scaled FP4 formats do.
    """
    mbits, emin, vmax = FORMATS[fmt]
    if fmt == 'f4':
        shp = x.shape
        pad = (-shp[-1]) % block
        xp = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
        xb = xp.reshape(shp[:-1] + (-1, block))
        s = jnp.maximum(jnp.max(jnp.abs(xb), axis=-1, keepdims=True), 1e-30) / vmax
        out = (_round_minifloat(xb / s, mbits, emin, vmax) * s).reshape(xp.shape)
        return out[..., :shp[-1]]
    s = jnp.maximum(jnp.max(jnp.abs(x)), 1e-30) / vmax
    return _round_minifloat(x / s, mbits, emin, vmax) * s
