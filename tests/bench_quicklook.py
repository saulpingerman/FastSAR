"""Times of fastsar.quicklook on a spotlight CPHD file, by size and by stage, as JSON.

    python tests/bench_quicklook.py scene_CPHD.cphd --out quicklook_l4.json
    python tests/bench_quicklook.py scene_CPHD.cphd --backends cuda --sizes 2048,full --repeat 5 --out part.json

For each backend (cpu, and cuda where CuPy has a GPU), each size (1024, 2048, 4096 and 'full', the finest the
collection holds) and the displacement correction on and off: the first call of the process, then `--repeat`
calls, of which the fastest is kept with its stage times (plan, read, reference, geometry, range, azimuth, the
transforms, the correction, the remap, the display and the PNG), the cut (pulses, samples, megabytes read) and the
picture's shape. On cuda one more call is made with the device brought to completion after every step
(`steps`): the upload, the weights and padding, the range transforms, the range gate, the range resampling, the
azimuth transforms, gate and resampling, the image transforms and the read at the apparent positions. Its steps do
not overlap as they do in a normal call, so they sum to more than the stages; their shares say where the device's
time goes. The file is read from the page cache after the first call: the read time is not a cold one.

The JSON also holds what backend='auto' takes for each cut and the thresholds it used (quick.AUTO_CUDA_SAMPLES,
quick.AUTO_CUDA_FIRST_SAMPLES), so the crossover between the CPU and the GPU can be read off the two backends'
times by the cut's samples.
"""
import argparse
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cphd')
    ap.add_argument('--out', default='quicklook_bench.json')
    ap.add_argument('--sizes', default='1024,2048,4096,full')
    ap.add_argument('--backends', default='cpu,cuda')
    ap.add_argument('--repeat', type=int, default=3)
    a = ap.parse_args()
    import fastsar
    from fastsar import quick
    sizes = [s.strip() for s in a.sizes.split(',') if s.strip()]
    backends = []
    for b in a.backends.split(','):
        b = b.strip()
        try:
            backends.append(quick._backend(b))
        except ValueError as e:
            print(f'{b}: skipped ({e})')
    doc = dict(script='bench_quicklook', cphd=os.path.basename(a.cphd), fastsar=fastsar.__version__, python=sys.version.split()[0],
               machine=platform.machine(), cpus=os.cpu_count(), threads=quick._workers(), repeat=a.repeat,
               auto=dict(AUTO_CUDA_SAMPLES=quick.AUTO_CUDA_SAMPLES, AUTO_CUDA_FIRST_SAMPLES=quick.AUTO_CUDA_FIRST_SAMPLES),
               results=[])
    if 'cuda' in backends:
        import cupy as cp
        dev = cp.cuda.Device()
        free, total = dev.mem_info
        doc['gpu'] = dict(name=cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode(), memory_gb=round(total / 1e9, 1),
                          free_gb=round(free / 1e9, 1), cupy=cp.__version__)

    def save():
        tmp = a.out + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, a.out)
    png = os.path.join(tempfile.mkdtemp(prefix='quicklook-bench-'), 'look.png')
    first = {b: True for b in backends}
    for backend in backends:
        for size in sizes:
            for correct in (True, False):
                kw = dict(resolution='full') if size == 'full' else dict(size=int(size))
                row = dict(backend=backend, size=size, correct=correct)
                try:
                    ts, best = [], None
                    for k in range(a.repeat + 1):
                        t = time.perf_counter()
                        out = fastsar.quicklook(a.cphd, png, backend=backend, correct=correct, **kw)
                        el = time.perf_counter() - t
                        if k == 0:
                            row['first_call_seconds'] = round(el, 4)
                            row['first_of_process'] = first[backend]
                            first[backend] = False
                            continue
                        ts.append(el)
                        if best is None or el <= min(ts):
                            best = out['timing']
                    cut = out['cut']
                    P, K = cut['pulses'][1] - cut['pulses'][0], cut['samples'][1] - cut['samples'][0]
                    row.update(seconds=round(min(ts), 4), seconds_median=round(float(np.median(ts)), 4),
                               stages={k: round(float(v), 4) for k, v in best.items()}, picture=list(out['display'].shape),
                               pixel_m=round(out['pixel'], 4), history=[P, K], samples=P * K,
                               megabytes_read=round(cut['signal_bytes'] / 1e6, 1), formed_on=out['backend'],
                               auto_takes=quick._auto(P * K), notes=[n for n in out['notes'] if 'formed on the host' in n])
                    if backend == 'cuda' and out['backend'] == 'cuda':
                        # the steps on the device, one at a time
                        col = quick._collection(a.cphd, backend='cuda', **(dict(resolution='full') if size == 'full' else dict(size=int(size))))
                        tm = dict(detail={})
                        t = time.perf_counter()
                        img = quick._pfa([col.pop('S')], col['ant'], col['fmin'], col['df'], col['nx'], col['ny'], col['spx'],
                                         col['spy'], col['e1'], col['e2'], correct=correct, backend='cuda', complex_image=False,
                                         timing=tm, phase=col['phase'])
                        cp.cuda.get_current_stream().synchronize()
                        total = time.perf_counter() - t
                        del img, col
                        steps = {k: round(float(v), 4) for k, v in tm['detail'].items()}
                        row['steps'] = steps
                        row['steps_seconds'] = round(total, 4)
                        row['step_shares'] = {k: round(v / max(sum(steps.values()), 1e-9), 3) for k, v in steps.items()}
                    print(f"{backend:5s} {size:>5s} {'corrected  ' if correct else 'uncorrected'} {row['seconds']:8.3f} s "
                          f"(first {row['first_call_seconds']:.3f}); {P} x {K} samples, {row['megabytes_read']:.0f} MB; picture "
                          f"{row['picture'][1]} x {row['picture'][0]}" + (f"; formed on {out['backend']}" if out['backend'] != backend else ''),
                          flush=True)
                    if 'step_shares' in row:
                        print('        steps: ' + ', '.join(f'{k} {v:.0%}' for k, v in sorted(row['step_shares'].items(), key=lambda kv: -kv[1])), flush=True)
                    del out
                except Exception as e:
                    row['failed'] = f'{type(e).__name__}: {str(e)[:300]}'
                    print(f'{backend} {size} corrected={correct}: {row["failed"]}', flush=True)
                doc['results'].append(row)
                save()
    save()
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
