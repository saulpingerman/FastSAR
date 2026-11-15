"""A quick look at a spotlight CPHD file: a reduced-resolution picture from the part of the file it needs.

    python examples/quicklook.py scene_CPHD.cphd look.png [--size 2048] [--resolution 1.0 | full] [--uncorrected]
                                 [--looks 2] [--backend cpu|cuda|jax|tpu|auto] [--repeat 2]

Writes the picture (.png) or the complex image as a slant-plane SICD (.nitf) and prints what was read, the notes and
the time of each stage. --repeat forms the picture again: the first call loads or compiles the kernels (and, on a
GPU, makes the context and the FFT plans), the later ones show the time of a look.
"""
import argparse
import time

import fastsar


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cphd')
    ap.add_argument('output', nargs='?')
    ap.add_argument('--size', type=int, default=2048)
    ap.add_argument('--resolution', default=None, help="ground resolution in meters, or 'full'")
    ap.add_argument('--uncorrected', action='store_true')
    ap.add_argument('--looks', type=int, default=1)
    ap.add_argument('--backend', default='auto')
    ap.add_argument('--repeat', type=int, default=1)
    a = ap.parse_args()
    res = a.resolution if a.resolution in (None, 'full') else float(a.resolution)
    for k in range(a.repeat):
        t = time.perf_counter()
        out = fastsar.quicklook(a.cphd, a.output, size=a.size, resolution=res, correct=not a.uncorrected,
                                looks=a.looks, backend=a.backend)
        el = time.perf_counter() - t
        cut = out['cut']
        print(f"call {k + 1}: {el:.3f} s on '{out['backend']}' ({out['algorithm']}); picture {out['display'].shape[1]} by "
              f"{out['display'].shape[0]} pixels of {out['pixel']:.2f} m; read {cut['signal_bytes'] / 1e6:.1f} MB of "
              f"{cut['of_bytes'] / 1e9:.2f} GB")
        print('  stages (s): ' + ', '.join(f'{name} {v:.3f}' for name, v in out['timing'].items()))
    for n in out['notes']:
        print('  -', n)
    print('  corners (latitude, longitude):', out['corners'].round(5).tolist())


if __name__ == '__main__':
    main()
