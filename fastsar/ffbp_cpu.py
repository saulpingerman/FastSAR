"""Factorized backprojection on the CPU with the C++/OpenMP kernels of ffbp_cpu.cpp: the same three stages and the
same host-side orchestration as the CUDA pipeline (ffbp_cuda.py), with NumPy for the float64 geometry.

    form = make_ffbp_cpu(plan, coll); img = form(S)        # S [P, K] complex64 (windowed) -> complex64 [nx, ny]

The shared library is compiled on first use with g++ (-O3 -march=native -fopenmp) into ~/.cache/sarbench.
"""
import ctypes, hashlib, os, subprocess, time

import numpy as np

from .ffbp import C
from .ffbp2 import HOST_LEVELS, make_plan, collection_arrays  # noqa: F401  (re-exported for callers)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffbp_cpu.cpp')
_lib = None
PROFILE = {}


def _mark(name, t0):
    if PROFILE.get('on'):
        PROFILE[name] = PROFILE.get(name, 0.0) + time.perf_counter() - t0


def lib():
    """Compile (once) and load the kernels."""
    global _lib
    if _lib is not None:
        return _lib
    src = open(_SRC).read()
    tag = hashlib.sha1(src.encode()).hexdigest()[:12]
    d = os.path.join(os.path.expanduser('~'), '.cache', 'fastsar')
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f'libffbp_cpu_{tag}.so')
    if not os.path.exists(so):
        cxx = os.environ.get('CXX', 'g++')
        flags = os.environ.get('FFBP_CPU_FLAGS', '-O3 -march=native -mprefer-vector-width=512 -funroll-loops')
        cmd = [cxx] + flags.split() + ['-fopenmp', '-shared', '-fPIC', '-std=c++17', _SRC, '-o', so + '.tmp']
        subprocess.run(cmd, check=True)
        os.replace(so + '.tmp', so)
    L = ctypes.CDLL(so)
    f32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
    f64 = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
    i, dbl = ctypes.c_int, ctypes.c_double
    L.rot_fir_k.argtypes = [f32, f32, i, i, i, f32, f32, i, f32, i, i, i, i, dbl, f32, f32]
    L.fir_p.argtypes = [f32, f32, i, i, i, f32, i, i, i, i, f32, f32]
    L.final_tiles.argtypes = [f32, f32, i, i, i, i, f64, f64, f64, f64, dbl, dbl, f64, f64, f64, dbl, f32, f32]
    L.ffbp_cpu_threads.restype = i
    _lib = L
    return L


class Pool:
    """Reused intermediates: a fresh np.empty of hundreds of megabytes per call costs a page fault per 4 KB inside the
    parallel regions; the same buffers are handed out again when the shape repeats."""

    def __init__(self):
        self.d = {}

    def get(self, name, shape):
        a = self.d.get(name)
        if a is None or a.shape != tuple(shape):
            a = np.empty(shape, np.float32)
            self.d[name] = a
        return a


def _children(pre, pim, c0, sl, lv, pool, tag=''):
    """pre, pim [Np, P, K] float32; c0, sl [Np, C, P] float32 -> [Np C, Po, Ko] planes (buffers owned by the pool)."""
    L_ = lib()
    Np, P, K = pre.shape
    Cn, Ko, Po, Dk, Dp = c0.shape[1], lv['Ko'], lv['Po'], lv['Dk'], lv['Dp']
    kc = (K - 1) / 2.0
    t0 = time.perf_counter()
    fk = lv['fir_k']
    if Dk > 1:
        taps = np.ascontiguousarray(fk['kern'], np.float32)
        yre = pool.get(f'yre{tag}', (Np * Cn, P, Ko))
        yim = pool.get(f'yim{tag}', (Np * Cn, P, Ko))
        L_.rot_fir_k(pre, pim, Np, P, K, np.ascontiguousarray(c0, np.float32), np.ascontiguousarray(sl, np.float32), Cn,
                     taps, fk['L'], fk['pl'], Dk, Ko, kc, yre, yim)
    else:
        k = np.arange(K, dtype=np.float64) - kc
        cyc = c0[:, :, :, None].astype(np.float64) + k[None, None, None, :] * sl[:, :, :, None]
        ang = (cyc - np.rint(cyc)) * (2 * np.pi)
        cs, sn = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        yre = np.ascontiguousarray((pre[:, None] * cs - pim[:, None] * sn).reshape(Np * Cn, P, K))
        yim = np.ascontiguousarray((pre[:, None] * sn + pim[:, None] * cs).reshape(Np * Cn, P, K))
    _mark('rot_fir_k', t0)
    t0 = time.perf_counter()
    if Dp > 1:
        fp = lv['fir_p']
        taps = np.ascontiguousarray(fp['kern'], np.float32)
        zre = pool.get(f'zre{tag}', (Np * Cn, Po, Ko))
        zim = pool.get(f'zim{tag}', (Np * Cn, Po, Ko))
        L_.fir_p(yre, yim, Np * Cn, P, Ko, taps, fp['L'], fp['pl'], Dp, Po, zre, zim)
        _mark('fir_p', t0)
        return zre, zim
    return yre, yim


def _device_phases(la, refs, lv):
    """Band-center phase and slope of every child of every parent (float64): refs [Np, 3] -> c0, slope [Np, C, P]
    float32. Same formula as ffbp2.device_phases and ffbp_cuda._device_phases."""
    k0c = 2.0 * (lv['f0'] + (lv['K'] - 1) / 2.0 * lv['df']) / C
    k1 = 2.0 * lv['df'] / C
    u, r0, d = la['u'], la['r0'], la['d']                                   # [P, 3], [P], [C, 3]
    uc = refs @ u.T                                                          # [Np, P]
    wn = r0[None, :] * np.sqrt(1 + ((refs * refs).sum(1)[:, None] - 2 * r0[None, :] * uc) / (r0 * r0)[None, :])
    ud = d @ u.T                                                             # [C, P]
    wd = r0[None, None, :] * ud[None] - (d[None, :, :] * refs[:, None, :]).sum(2)[:, :, None]      # [Np, C, P]
    num = (d * d).sum(1)[None, :, None] - 2 * wd
    ddr = num / (np.sqrt(wn[:, None, :] ** 2 + num) + wn[:, None, :])
    c0 = ddr * k0c
    return (c0 - np.rint(c0)).astype(np.float32), (ddr * k1).astype(np.float32)


def make_ffbp_cpu(plan, coll):
    """Build form(S) -> complex64 image [nx, ny] for the plan and the collection arrays (host float64)."""
    L_ = lib()
    levels, T, nlev = plan['levels'], plan['T'], len(plan['levels'])
    fin = plan['final']
    Pf, Qf = fin['P'], fin['K']
    a0, a1, fc2 = 2.0 * fin['f0'] / C, 2.0 * fin['df'] / C, 2.0 * fin['fc'] / C
    e1, e2 = np.asarray(plan['e1'], np.float64), np.asarray(plan['e2'], np.float64)
    en = np.cross(e1, e2)
    dlx = np.ascontiguousarray((np.arange(T) - (T - 1) / 2.0) * plan['spx'], np.float64)
    dly = np.ascontiguousarray((np.arange(T) - (T - 1) / 2.0) * plan['spy'], np.float64)
    host = []
    for i, (lv, cl) in enumerate(zip(levels, coll['levels'])):
        e = {}
        if i < HOST_LEVELS:
            e['c0'], e['slope'] = np.asarray(cl['c0'], np.float32), np.asarray(cl['slope'], np.float32)
        else:
            e.update(u=np.asarray(cl['u'], np.float64), r0=np.asarray(cl['r0'], np.float64), d=np.asarray(lv['d'], np.float64), ref=np.asarray(lv['ref'], np.float64))
        host.append(e)
    fu, fr0 = np.asarray(coll['final']['u'], np.float64), np.asarray(coll['final']['r0'], np.float64)
    fcen = np.asarray(fin['cen'], np.float64)
    sx0, sy0 = levels[0]['sx'], levels[0]['sy']
    G = sx0 * sy0
    mx, my = plan['Nx'] // sx0, plan['Ny'] // sy0
    pool = Pool()
    shape_g = [s for lv in levels[1:] for s in (lv['sx'], lv['sy'])] + [T, T]
    perm_g = [2 * i for i in range(nlev - 1)] + [2 * (nlev - 1)] + [2 * i + 1 for i in range(nlev - 1)] + [2 * (nlev - 1) + 1]

    def final(are, aim, cen):
        """are, aim [B, Pf, Qf]; cen [B, 3] float64 -> (re, im) [B, T, T]."""
        B = are.shape[0]
        t0 = time.perf_counter()
        w = fr0[None, :, None] * fu[None, :, :] - cen[:, None, :]                       # [B, Pf, 3]
        wn = np.sqrt((w * w).sum(2))
        ux = np.ascontiguousarray((w @ e1) / wn)
        uy = np.ascontiguousarray((w @ e2) / wn)
        uz = (w @ en) / wn
        mxv, myv, mzv = ux.mean(1), uy.mean(1), uz.mean(1)
        mn = np.sqrt(mxv * mxv + myv * myv + mzv * mzv)
        ucx, ucy, rc = np.ascontiguousarray(mxv / mn), np.ascontiguousarray(myv / mn), np.ascontiguousarray(wn.mean(1))
        _mark('final_geom', t0)
        t0 = time.perf_counter()
        ore = pool.get('ore', (B, T, T))
        oim = pool.get('oim', (B, T, T))
        L_.final_tiles(np.ascontiguousarray(are), np.ascontiguousarray(aim), B, Pf, Qf, T, ux, uy, dlx, dly, a0, a1, ucx, ucy, rc, fc2, ore, oim)
        _mark('final_tile', t0)
        return ore, oim

    def one_tile(a, b, g):
        """Level-0 child g [Po0, Ko0] through the remaining levels -> its mx x my block."""
        a, b = a[None], b[None]
        for i in range(1, nlev):
            lv, la = levels[i], host[i]
            if i < HOST_LEVELS:
                c0, sl = la['c0'][g][None], la['slope'][g][None]
            else:
                nb_par = a.shape[0]
                refs = la['ref'].reshape(G, nb_par, 3)[g]
                t0 = time.perf_counter()
                c0, sl = _device_phases(la, refs, lv)
                _mark('device_phases', t0)
            a, b = _children(a, b, c0, sl, lv, pool, tag=str(i))
        cen = fcen.reshape(G, -1, 3)[g]
        re, im = final(a, b, cen)
        return (re.reshape(shape_g).transpose(perm_g).reshape(mx, my).copy(), im.reshape(shape_g).transpose(perm_g).reshape(mx, my).copy())

    ox, oy, nx, ny = plan['ox'], plan['oy'], plan['nx'], plan['ny']

    def form(S, ng=8):
        """S [P, K] complex64 (already windowed) -> complex64 image [nx, ny]."""
        S = np.asarray(S)
        scale = float(np.abs(S).max())
        pre = np.ascontiguousarray((S.real / scale).astype(np.float32))[None]
        pim = np.ascontiguousarray((S.imag / scale).astype(np.float32))[None]
        lv0, la0 = levels[0], host[0]
        full = np.empty((plan['Nx'], plan['Ny']), np.complex64)
        for g0 in range(0, G, ng):
            gs = list(range(g0, min(G, g0 + ng)))
            c0 = np.ascontiguousarray(la0['c0'][0, gs][None])
            sl = np.ascontiguousarray(la0['slope'][0, gs][None])
            A, Bm = _children(pre, pim, c0, sl, lv0, pool, tag='0')                  # [ng, Po, Ko]
            for k, g in enumerate(gs):
                re, im = one_tile(A[k], Bm[k], g)
                x, y = g // sy0, g % sy0
                full[x * mx:(x + 1) * mx, y * my:(y + 1) * my] = re + 1j * im
            del A, Bm
        return full[ox:ox + nx, oy:oy + ny] * np.float32(scale)

    return form
