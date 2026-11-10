"""Remaps from complex or amplitude pixels to display values, matching the remap functions of the NGA MATLAB SAR
Toolbox (Visualization/remap/remap_funcs): density, brighter, darker, highcontrast, pedf, nrl, log and linear.

    img8 = fastsar.remap.nrl(image)                     # uint8, as nrlremap.m
    img8 = fastsar.remap.remap(image, 'density')        # by name
    img8 = fastsar.remap.nrl(image, valid=mask)         # statistics over the valid pixels only

Each function follows its .m file line for line: the arithmetic is single precision where MATLAB's is, and the
conversion to uint8 rounds halves away from zero and saturates, as MATLAB's uint8() does. Run in Octave on the same
pixels, the toolbox's functions give the same counts, except where a value sits on a rounding boundary (a few pixels
in 100,000 for the density family, up to one in 1,000 for log, never by more than one count): the toolbox sums
millions of single-precision values for its means, and such a sum depends on the order of its terms in the last
digits. The sums here are accumulated in double precision.

The toolbox computes its statistics (mean, minimum, maximum, 99th percentile) over every pixel. An image that is
zero outside its processed area would pull those statistics down, so each remap takes `valid`, a boolean mask of the
pixels that enter the statistics; the others come out 0. Without it the result is the toolbox's.

    density        log amplitude between 0.8 and 32 times the mean amplitude, on 30..255 (amplitudetodensity)
    brighter       the same on 60..255
    darker         the same on 0..255
    highcontrast   log amplitude between 0.8 and 3.2 times the mean amplitude, on 30..255
    pedf           density with the upper half (above 128) compressed by two (piecewise extended density format)
    nrl            linear up to the 99th percentile (output 0..220), logarithmic from there to the maximum (220..255)
    log            decibels over a 50 dB span centered on the mean power
    linear         the amplitude itself (floating point, not scaled)
"""
import numpy as np

__all__ = ['density', 'brighter', 'darker', 'highcontrast', 'pedf', 'nrl', 'log', 'linear', 'remap', 'REMAPS',
           'amplitude_to_density', 'percentile']


def _uint8(x):
    """MATLAB's uint8(): round to nearest with halves away from zero, saturate to 0..255, NaN to 0."""
    x = np.asarray(x)
    out = np.zeros(x.shape, np.uint8)
    ok = ~np.isnan(x)
    np.floor(np.clip(x, 0.0, 255.0, where=ok, out=np.zeros(x.shape, x.dtype)) + x.dtype.type(0.5), out=out, casting='unsafe')
    return out


def _amp(x):
    return np.abs(np.asarray(x)).astype(np.float32, copy=False)


def _sel(A, valid):
    return A.ravel() if valid is None else A[np.asarray(valid, bool)]


def amplitude_to_density(x, dmin=30.0, mmult=40.0, mean=None, valid=None):
    """amplitudetodensity.m: density values (float32, not clipped) of the amplitude of x. The amplitude 0.8 times
    the mean maps to dmin and mmult times that to 255, logarithmically. mean: the amplitude's mean, when it is
    known (default: over the finite pixels, or the finite valid ones)."""
    A = _amp(x)
    f = np.float32
    if mean is None:
        v = _sel(A, valid)
        v = v[np.isfinite(v)]
        mean = v.mean(dtype=np.float64) if v.size else 1.0
    cl = f(0.8) * f(mean)
    ch = f(mmult) * cl
    m = f(255.0 - dmin) / np.log10(ch / cl, dtype=f)
    b = f(dmin) - m * np.log10(cl, dtype=f)
    return m * np.log10(np.maximum(A, f(1e-5))) + b


def _density(dmin, mmult, doc):
    def fn(x, valid=None, mean=None):
        out = _uint8(amplitude_to_density(x, dmin, mmult, mean, valid))
        if valid is not None:
            out[~np.asarray(valid, bool)] = 0
        return out
    fn.__doc__ = doc
    return fn


density = _density(30.0, 40.0, 'densityremap.m: uint8 density, 30 at 0.8 times the mean amplitude, 255 at 32 times it.')
brighter = _density(60.0, 40.0, 'brighterremap.m: as density, on 60..255.')
darker = _density(0.0, 40.0, 'darkerremap.m: as density, on 0..255.')
highcontrast = _density(30.0, 4.0, 'highcontrastremap.m: uint8 density, 30 at 0.8 times the mean amplitude, 255 at 3.2 times it.')


def pedf(x, valid=None, mean=None):
    """pedfremap.m: piecewise extended density format, density with the values above 128 halved toward 128."""
    d = amplitude_to_density(x, 30.0, 40.0, mean, valid)
    hi = d > 128
    d[hi] = np.float32(0.5) * (d[hi] + np.float32(128))
    out = _uint8(d)
    if valid is not None:
        out[~np.asarray(valid, bool)] = 0
    return out


def linear(x, valid=None):
    """linearremap.m: the amplitude of a complex image, a real image unchanged. Floating point, not scaled."""
    x = np.asarray(x)
    out = np.abs(x.astype(np.complex128)) if np.iscomplexobj(x) else x.copy()
    if valid is not None:
        out[~np.asarray(valid, bool)] = 0
    return out


def log(x, valid=None, span_db=50.0):
    """logremap.m: decibels on a span of span_db centered on the mean power, 0..255."""
    f = np.float32
    A = _amp(x)
    v = _sel(A, valid)
    A = f(10) * A / f(v.mean(dtype=np.float64))
    v = _sel(A, valid)
    rcent = f(10) * np.log10(f(np.sum(v.astype(np.float64) ** 2) / v.size))
    A = A + max(f(1) - v.min(), f(0))
    A = f(20) * np.log10(A)
    v = _sel(A, valid)
    lo = max(v.min(), rcent - f(span_db / 2))
    hi = min(v.max(), rcent + f(span_db / 2))
    out = _uint8(f(255) * (A - lo) / (hi - lo))
    if valid is not None:
        out[~np.asarray(valid, bool)] = 0
    return out


def percentile(v, p, halfwidth=32):
    """The toolbox's percentile (pctl in nrlremap.m): a not-a-knot cubic spline through the sorted values, the i-th
    of n placed at (i - 0.5) / n, evaluated at p / 100. The spline is fitted to the 2 halfwidth order statistics
    around the query, which reproduces the spline through all of them (a spline's dependence on distant knots falls
    by a factor of about 3.7 per knot)."""
    from scipy.interpolate import CubicSpline
    v = np.ravel(np.asarray(v))
    n = v.size
    if n == 0:
        raise ValueError('percentile of no values')
    if n == 1:
        return v[0]
    t = p * 0.01 * n - 0.5                               # fractional index of the query among the sorted values
    lo = int(np.clip(np.floor(t) - halfwidth, 0, max(n - 2 * halfwidth, 0)))
    hi = min(n, lo + 2 * halfwidth)
    s = np.sort(np.partition(v, (lo, hi - 1))[lo:hi]).astype(np.float64) if n > 2 * halfwidth else np.sort(v).astype(np.float64)
    if hi - lo < 2 or np.all(s == s[0]):
        return v.dtype.type(s[0])
    return v.dtype.type(CubicSpline(np.arange(lo, hi), s, bc_type='not-a-knot', extrapolate=True)(t))


def nrl(x, a=1.0, c=220.0, valid=None):
    """nrlremap.m: linear from the minimum (0) to a times the 99th percentile (c), logarithmic from there to the
    maximum (255). a: scale of the 99th percentile; c: the output value where the logarithmic part starts."""
    f = np.float32
    A = _amp(x)
    v = _sel(A, valid)
    amin, amax = v.min(), v.max()
    knee = f(a) * percentile(v, 99) - amin
    with np.errstate(divide='ignore', invalid='ignore'):
        b = f(255 - c) / np.log10((amax - amin) / knee)
        lin = A <= f(a) * percentile(v, 99)
        y = np.where(lin, (A - amin) * f(c) / knee, b * np.log10((A - amin) / knee) + f(c))
    out = _uint8(y)
    if valid is not None:
        out[~np.asarray(valid, bool)] = 0
    return out


REMAPS = dict(density=density, brighter=brighter, darker=darker, highcontrast=highcontrast, pedf=pedf, nrl=nrl,
              log=log, linear=linear)


def remap(x, name='nrl', **kw):
    """The remap called `name` (a key of REMAPS, the toolbox's names without 'remap') applied to x."""
    try:
        fn = REMAPS[str(name).lower().removesuffix('remap')]
    except KeyError:
        raise ValueError(f'remap must be one of {sorted(REMAPS)}, got {name!r}') from None
    return fn(x, **kw)
