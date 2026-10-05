"""Form an Umbra spotlight image on the vendor's grid and save it as a complex64 .npy and an amplitude PNG.

    python examples/form_umbra.py scene_CPHD.cphd scene_SICD.nitf out [--backend cpu] [--precision float32]
"""
import argparse, time

import numpy as np

import fastsar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cphd'); ap.add_argument('sicd'); ap.add_argument('out')
    ap.add_argument('--backend', default='auto'); ap.add_argument('--precision', default='float32')
    ap.add_argument('--algorithm', default='ffbp')
    a = ap.parse_args()
    col = fastsar.io.read_cphd(a.cphd, sicd=a.sicd)
    print('pulses, samples', col['S'].shape, 'grid', col['nx'], col['ny'], 'backends', fastsar.available_backends())
    t = time.perf_counter()
    img = fastsar.form_image(**col, algorithm=a.algorithm, backend=a.backend, precision=a.precision)
    print(f'formed in {time.perf_counter() - t:.1f} s (includes compilation on the first call)')
    np.save(a.out + '.npy', img)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        db = 20 * np.log10(np.abs(img) + 1e-12); top = np.percentile(db, 99.9)
        plt.imsave(a.out + '.png', np.clip(db.T, top - 45, top), cmap='gray', vmin=top - 45, vmax=top, origin='lower')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
