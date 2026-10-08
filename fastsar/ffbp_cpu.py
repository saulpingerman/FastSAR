"""Factorized backprojection on the CPU with the C++/OpenMP kernels of ffbp_cpu.cpp: the same three stages and the
same host-side orchestration as the CUDA pipeline (ffbp_cuda.py), with NumPy for the float64 geometry.

    form = make_ffbp_cpu(plan, coll); img = form(S)        # S [P, K] complex64 (windowed) -> complex64 [nx, ny]

The shared library is compiled on first use with g++ (-O3 -march=native -fopenmp) into ~/.cache/fastsar.
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
    with open(_SRC) as fh:
        src = fh.read()
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
    L.rot_fir_k_cplx.argtypes = [f32, i, i, i, f32, f32, i, f32, i, i, i, i, dbl, f32, f32]
    L.fir_p.argtypes = [f32, f32, i, i, i, f32, i, i, i, i, f32, f32]
    vp = ctypes.c_void_p
    L.final_tiles.argtypes = [f32, f32, i, i, i, i, f64, f64, f64, f64, dbl, dbl, f64, f64, f64, dbl, f32, f32, vp, vp, vp]
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
    """pre, pim [Np, P, K] float32 (or pre a complex64 [Np, P, K] array and pim None); c0, sl [Np, C, P] float32 ->
    [Np C, Po, Ko] planes (buffers owned by the pool)."""
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
        if pim is None:
            L_.rot_fir_k_cplx(pre.view(np.float32), Np, P, K, np.ascontiguousarray(c0, np.float32), np.ascontiguousarray(sl, np.float32),
                              Cn, taps, fk['L'], fk['pl'], Dk, Ko, kc, yre, yim)
        else:
            L_.rot_fir_k(pre, pim, Np, P, K, np.ascontiguousarray(c0, np.float32), np.ascontiguousarray(sl, np.float32), Cn,
                         taps, fk['L'], fk['pl'], Dk, Ko, kc, yre, yim)
    else:
        if pim is None:
            pre, pim = np.ascontiguousarray(pre.real), np.ascontiguousarray(pre.imag)
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


def make_ffbp_cpu(plan, coll, wf=None):
    """Build form(S) -> complex64 image [nx, ny] for the plan and the collection arrays (host float64). wf: final
    subaperture weights per final tile [ntiles, Pf] (api.final_weights), or None."""
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

    def final(are, aim, cen, wts=None):
        """are, aim [B, Pf, Qf]; cen [B, 3] float64; wts None or (w0, gx, gyr) [B, Pf] float32 -> (re, im) [B, T, T]."""
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
        wp = [None] * 3 if wts is None else [np.ascontiguousarray(w, np.float32) for w in wts]
        L_.final_tiles(np.ascontiguousarray(are), np.ascontiguousarray(aim), B, Pf, Qf, T, ux, uy, dlx, dly, a0, a1, ucx, ucy, rc, fc2, ore, oim,
                       *[None if w is None else w.ctypes.data for w in wp])
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
        if wf is not None and wf.ndim == 3:      # weight plus its gradient across the tile, in the final stage
            w = wf.reshape(3, G, -1, Pf)[:, g]
            w0 = w[0]
            gyr = np.where(np.abs(w0) > 1e-6 * max(float(np.abs(w0).max()), 1e-30), w[2] / np.where(w0 == 0, 1, w0), 0.0)
            re, im = final(a, b, cen, (w0, w[1], gyr))
        else:
            if wf is not None:
                w = wf.reshape(G, -1, Pf)[g][:, :, None]
                a, b = a * w, b * w
            re, im = final(a, b, cen)
        return (re.reshape(shape_g).transpose(perm_g).reshape(mx, my).copy(), im.reshape(shape_g).transpose(perm_g).reshape(mx, my).copy())

    ox, oy, nx, ny = plan['ox'], plan['oy'], plan['nx'], plan['ny']

    def form(S, ng=None):
        """S [P, K] complex64 (already windowed) -> complex64 image [nx, ny]. ng: level-0 children per group (default:
        up to 8, fewer when a group's buffers, pulses x range samples per child, would exceed a quarter of the available memory
        or FASTSAR_CPU_GROUP_GB)."""
        S = np.asarray(S)
        if ng is None:
            lv = levels[0]
            per = 8.0 * lv['Ko'] * (lv['P'] + lv['Po'])                       # yre, yim and zre, zim of one child
            from .memory import host_available
            mem = host_available()                                              # available, not installed: a shared host
            budget = float(os.environ.get('FASTSAR_CPU_GROUP_GB') or 0) * 1e9 or 0.25 * mem
            ng = int(max(1, min(8, budget // per)))
            if ng < min(8, G) and not os.environ.get('FASTSAR_CPU_GROUP_GB'):
                from .memory import warn, gb, host_available, children
                warn(f'cpu: first-level groups of {children(ng)} instead of 8 for lack of memory, so the phase history is '
                     f'read {-(-G // ng)} times instead of {-(-G // 8)}; full speed needs about {gb(4 * 8 * per)} of host '
                     f'memory, {gb(host_available())} is available')
        if S.dtype == np.complex64 and S.flags.c_contiguous and levels[0]['Dk'] > 1:
            # the first level reads the complex64 history in place (a long spotlight's history is tens of GB)
            scale, pre, pim = 1.0, S[None], None
        else:
            # in row blocks and in place: each full-size temporary is as large as the history
            scale = max(float(np.abs(S[i:i + 4096]).max()) for i in range(0, S.shape[0], 4096)) or 1.0     # all zero: 1
            pre = np.empty((1,) + S.shape, np.float32); pim = np.empty((1,) + S.shape, np.float32)
            for i in range(0, S.shape[0], 4096):
                np.multiply(S[i:i + 4096].real, np.float32(1.0 / scale), out=pre[0, i:i + 4096])
                np.multiply(S[i:i + 4096].imag, np.float32(1.0 / scale), out=pim[0, i:i + 4096])
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
        return full[ox:ox + nx, oy:oy + ny] if scale == 1.0 else full[ox:ox + nx, oy:oy + ny] * np.float32(scale)

    return form
