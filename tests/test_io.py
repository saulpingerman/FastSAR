"""CPHD helpers that need no file: per-pulse frequency resampling, re-referencing the motion-compensation point,
and the geodetic conversions."""
from types import SimpleNamespace

import numpy as np
import pytest

from fastsar import io, sim

C = 299792458.0


@pytest.fixture(scope='module')
def draws():
    """The random inputs of the three checks, drawn in the original order from one generator."""
    rng = np.random.default_rng(0)
    tau = rng.uniform(-0.3, 0.3, 6)
    a = rng.standard_normal(6) + 1j * rng.standard_normal(6)
    tg = np.stack([rng.uniform(-15, 15, 5), rng.uniform(-15, 15, 5), np.zeros(5)], 1)
    amps = rng.standard_normal(5) + 1j
    lat, lon, h = rng.uniform(-80, 80, 50), rng.uniform(-180, 180, 50), rng.uniform(-100, 9000, 50)
    return SimpleNamespace(tau=tau, a=a, tg=tg, amps=amps, lat=lat, lon=lon, h=h)


def test_sinc_regrid(draws):
    """A row of scatterer returns sampled on a slightly offset and stretched frequency grid, resampled onto the
    nominal one."""
    K = 2048
    f = lambda k: (draws.a * np.exp(-2j * np.pi * k[..., None] * draws.tau)).sum(-1)
    k = np.arange(K)[None, :].astype(float)
    u = 0.37 + k * (1 + 3e-6)
    out = io._sinc_regrid(f(k).astype(np.complex64), u)
    e = 10 * np.log10(np.sum(np.abs(out[0, 20:-20] - f(u)[0, 20:-20]) ** 2) / np.sum(np.abs(f(u)[0, 20:-20]) ** 2))
    print(f'frequency resampling (16 taps, interior): {e:.1f} dB')
    assert e < -70, e


def test_rereference(draws):
    """Data compensated to the origin, re-referenced to a point c, equals data simulated with c as reference."""
    col = sim.make_collect(res=0.5, scene=40.0, r0=8e3)
    c = np.array([3.0, -2.0, 0.5])
    S0 = np.zeros((col.Np, col.K), complex); S1 = np.zeros_like(S0)
    for x, amp in zip(draws.tg, draws.amps):
        r = np.linalg.norm(x - col.ant, axis=1)
        S0 += amp * np.exp(-4j * np.pi * col.freqs / C * (r - np.linalg.norm(col.ant, axis=1))[:, None])
        S1 += amp * np.exp(-4j * np.pi * col.freqs / C * (r - np.linalg.norm(col.ant - c, axis=1))[:, None])
    S2 = io.rereference(S0, col.fmin, col.df, np.linalg.norm(col.ant, axis=1) - np.linalg.norm(col.ant - c, axis=1))
    e = 10 * np.log10(np.sum(np.abs(S2 - S1) ** 2) / np.sum(np.abs(S1) ** 2))
    print(f're-referencing: {e:.1f} dB')
    assert e < -100, e


def test_geodetic_round_trip(draws):
    lat, lon, h = draws.lat, draws.lon, draws.h
    la, lo, hh = io.ecf_to_geodetic(io.geodetic_to_ecf(lat, lon, h))
    err = max(np.abs(la - lat).max() * 111e3, np.abs((lo - lon + 180) % 360 - 180).max() * 111e3, np.abs(hh - h).max())
    print(f'geodetic round trip: {err:.2e} m')
    assert err < 1e-4, err
