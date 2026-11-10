"""Check of an installed FastSAR on the CPU backend, with NumPy, SciPy and JAX only (run by the wheels workflow in
a container without a compiler, and with one):

    python wheel_check.py --expect prebuilt --no-compiler
    python wheel_check.py --expect native

Forms an image of simulated point targets with the factorized, the polar format and the exact algorithm, resamples
one, writes a CPHD file, reads it and processes it to a SICD and a SIDD, decodes a Sentinel-1 packet, and checks
which shared objects the kernels came from. --no-compiler: fail if a C++ compiler is on the path. --expect: the
kernels must all be the native build or all prebuilt."""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--expect', choices=('native', 'prebuilt'))
    ap.add_argument('--no-compiler', action='store_true')
    a = ap.parse_args(argv)
    if a.no_compiler:
        found = [c for c in ('g++', 'c++', 'gcc', 'cc', 'clang++', 'clang') if shutil.which(c)]
        assert not found and not os.environ.get('CXX'), f'a compiler is on the path: {found}, CXX={os.environ.get("CXX")}'
        print('no C++ compiler on the path')
    import fastsar
    from fastsar import _build, resample, sentinel1, sim
    print('fastsar', fastsar.__version__, 'from', os.path.dirname(fastsar.__file__))
    print('levels of this CPU:', ', '.join(_build.status()['levels']) or 'none')

    targets = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 15.0, 0.0], [-18.0, -22.0, 0.0]])
    col = sim.make_collect(res=0.35, scene=170.0, r0=10e3, graze_deg=35.0)
    S = sim.simulate_brute(col, targets, np.ones(len(targets))).astype(np.complex64)
    n, sp = 512, 0.2
    grid = (n, n, sp, sp, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    imgs = {alg: fastsar.form_image(S, col.ant, col.fmin, col.df, *grid, backend='cpu', algorithm=alg,
                                    **(dict(pfa_guard=40.0) if alg == 'pfa' else {})) for alg in ('ffbp', 'pfa', 'bp')}
    err = 20 * np.log10(np.abs(imgs['ffbp'] - imgs['bp']).max() / np.abs(imgs['bp']).max())
    cc = np.corrcoef(np.abs(imgs['pfa']).ravel(), np.abs(imgs['bp']).ravel())[0, 1]      # (polar format: its own scale)
    print(f'form_image: ffbp against exact backprojection {err:.1f} dB, pfa amplitude correlation {cc:.3f}')
    assert all(np.isfinite(img).all() for img in imgs.values()) and err < -30.0 and cc > 0.9
    from fastsar import bp
    v = bp.backproject(S, col.ant, col.fmin, col.df, np.vstack([targets, [[7.3, 31.0, 0.0]]]), backend='cpu')
    print(f'backproject: at the targets {np.abs(v[:-1]).min():.1f} to {np.abs(v[:-1]).max():.1f}, elsewhere {abs(v[-1]):.1f}')
    assert np.abs(v[:-1]).min() > 20 * abs(v[-1])
    # the resampler: a shift by whole pixels reproduces the image
    i = np.arange(n, dtype=np.float64)
    out = resample.warp(imgs['bp'], np.broadcast_to(i + 3.0, (n, n)).copy(), np.broadcast_to(i - 2.0, (n, n)).copy(), (n, n))
    d = np.abs(out[8:-8, 8:-8] - imgs['bp'][11:-5, 6:-10]).max() / np.abs(imgs['bp']).max()
    print(f'resample.warp: shift by whole pixels reproduced to {d:.1e}')
    assert d < 1e-3
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'c.cphd')
        sim.write_cphd(p, col, S, 40.79, -77.86, 351.0, heading=30.0, scene=140.0)
        r = fastsar.io.read_cphd(p)
        assert r['S'].shape == S.shape and np.isfinite(r['S']).all()
        print('read_cphd:', r['S'].shape, r['S'].dtype)
        for product in ('sicd', 'sidd'):
            f = os.path.join(tmp, f'out_{product}.nitf')
            fastsar.process_cphd(p, f, product=product, backend='cpu')
            assert os.path.getsize(f) > 10000
            print(f'process_cphd {product}: {os.path.getsize(f)} bytes')
    # one Sentinel-1 bypass packet (10-bit sign and magnitude samples)
    vals = [[5, -17, 300], [2, 4, -6], [-1, 0, 511], [100, -100, 7]]
    bits = ''
    for ch in vals:
        b = ''.join(('1' if v < 0 else '0') + format(abs(v), '09b') for v in ch)
        bits += b + '0' * (-len(b) % 16)
    user = bytes(int(bits[k:k + 8], 2) for k in range(0, len(bits), 8))
    hdr = dict(quads=np.array([3]), user_offset=np.array([0]), user_length=np.array([len(user)]), baq_mode=np.array([0]))
    dec, bad = sentinel1.decode_user_data(user, hdr, [0])
    assert bad == 0 and np.allclose(dec[0], [5 - 1j, 2 + 100j, -17 + 0j, 4 - 100j, 300 + 511j, -6 + 7j])
    print('sentinel1.decode_user_data: one packet')

    used = {name: (u['path'], u['level']) for name, u in _build.USED.items()}
    for name, (path, level) in sorted(used.items()):
        print(f'  {name:14s} {path}' + (f' ({level})' if level else ''))
    assert {name for name, *_ in _build.KERNELS} <= set(used), sorted(used)
    if a.expect:
        assert {p for p, _ in used.values()} == {a.expect}, f'expected {a.expect} kernels: {used}'
    print('ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
