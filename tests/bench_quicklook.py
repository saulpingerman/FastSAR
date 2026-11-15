"""Times of polar format on a spotlight collection, the full image first, by backend and by stage, as JSON.

    python tests/bench_quicklook.py scene_CPHD.cphd --out quicklook_l4.json
    python tests/bench_quicklook.py scene_CPHD.cphd --history panama.npz --out full_l4.json
    python tests/bench_quicklook.py scene_CPHD.cphd --backends cuda --sizes full,2048 --device-mb 14000 --out part.json
    python tests/bench_quicklook.py scene_CPHD.cphd --history panama.npz --backends cpu,tpu --program --out v6e.json

For each backend (cpu, and cuda where CuPy has a GPU; --backends cpu,tpu on a TPU VM) and each size ('full', the
whole tasked scene at the finest resolution the collection holds, first; then 4096, 2048 and 1024 pixels), with the
displacement correction on and off: fastsar.quicklook from the file to the PNG, the first call at that size (with
its stage times, `first_call_stages`; the first of the process for the first row of a backend), then
`--repeat` calls, of which the fastest is kept with its stage times (plan, read, reference, geometry, range,
azimuth, the transforms, the correction, the remap, the display and the PNG), the cut (pulses, samples, megabytes
read), the picture's shape and what was kept on the host where the device had no room. On a device one more
formation is made with the device brought to completion after every step (`steps`): on cuda the upload, the
weights and padding, the range transforms, gate and resampling, the azimuth layout, transforms, gate and
resampling, the image transforms, the read at the apparent positions and the blocks that cross to and from the
host; on jax and tpu the programs of fastsar.quick_jax by name. The steps do not overlap as they do in a normal
call, so they sum to more than the stages; their shares say where the device's time goes. On jax and tpu each row
also lists the programs its first call compiled, with the seconds of their first calls (`compiled`); on cuda, the cuFFT plans
the first call and the later ones made (`plans`: count, seconds and the slowest, a length cuFFT compiles at run
time taking tens of milliseconds where the driver's cache does not hold it). The file is read from the page cache
after the first call: the read time is not a cold one.

--history: a prepared collection (.npz with S [pulses, samples], ant, fmin, df, nx, ny, spx, spy, e1, e2) whose
full image form_image(algorithm='pfa') forms on each backend, from the history in memory to the complex image on
the host, with the stages' times and, on a device, the image's error against the cpu backend's; it is measured
before the file's sizes. --program: the JAX program of 0.3.0 (FASTSAR_PFA_PROGRAM=jax) on the same history, twice.
--device-mb holds the device paths to that many megabytes (FASTSAR_CUDA_BLOCK_MB, FASTSAR_JAX_BLOCK_MB), to see a
smaller device's blocks and what it keeps on the host.

The JSON also holds what backend='auto' takes for each cut and the thresholds it used (quick.AUTO_CUDA_SAMPLES,
quick.AUTO_CUDA_FIRST_SAMPLES, quick.AUTO_TPU_SAMPLES), so the crossover between the host's cores and a device can
be read off the two backends' times by the cut's samples. --fresh N: then, for each backend and size, N new
processes that each import fastsar and make one quick look to the PNG (`fresh`: the seconds of that look, which on
cuda include importing CuPy, making the context and loading the kernels; and of the import): the first look of a
process, which quick.AUTO_CUDA_FIRST_SAMPLES is for.
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
    ap.add_argument('--sizes', default='full,4096,2048,1024')
    ap.add_argument('--backends', default='cpu,cuda')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--history', default=None, help='a prepared collection (.npz) for form_image(algorithm=\'pfa\')')
    ap.add_argument('--device-mb', default=None, help='megabytes the device paths may take (FASTSAR_CUDA_BLOCK_MB, FASTSAR_JAX_BLOCK_MB)')
    ap.add_argument('--program', action='store_true', help='with --history: also time the JAX program of 0.3.0')
    ap.add_argument('--fresh', type=int, default=0, help='new processes per backend and size, each timing its first quick look')
    a = ap.parse_args()
    if a.device_mb:
        os.environ['FASTSAR_CUDA_BLOCK_MB'] = os.environ['FASTSAR_JAX_BLOCK_MB'] = a.device_mb
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
               auto=dict(AUTO_CUDA_SAMPLES=quick.AUTO_CUDA_SAMPLES, AUTO_CUDA_FIRST_SAMPLES=quick.AUTO_CUDA_FIRST_SAMPLES,
                         AUTO_TPU_SAMPLES=quick.AUTO_TPU_SAMPLES),
               results=[])
    on_jax = [b for b in backends if b in ('jax', 'tpu')]
    if on_jax:
        import jax
        from fastsar import quick_jax, _xp
        doc['jax'] = dict(version=jax.__version__, devices=[str(d) for d in jax.devices()], memory=_xp._jax_stats(),
                          block=_xp.jax_block(), budget=_xp.jax_budget())

    def compiled(since):
        """The programs compiled since the `since`-th: their count, the seconds of their first calls, and those by name."""
        new, by = quick_jax.COMPILED[since:], {}
        for name, _, sec in new:
            by[name] = round(by.get(name, 0.0) + sec, 2)
        return dict(programs=len(new), seconds=round(sum(c[2] for c in new), 2), by_name=by)
    made = []                                         # the cuFFT plans the cuda path made: (length, batch, seconds)
    if 'cuda' in backends:
        import cupy as cp
        dev = cp.cuda.Device()
        free, total = dev.mem_info
        doc['gpu'] = dict(name=cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode(), memory_gb=round(total / 1e9, 1),
                          free_gb=round(free / 1e9, 1), cupy=cp.__version__, cufft=cp.cuda.cufft.getVersion(),
                          driver=cp.cuda.runtime.driverGetVersion(),
                          shared_points=int(dev.attributes['MaxSharedMemoryPerBlockOptin']) // 8)
        for name in ('_Plan', '_CupyPlan'):
            base = getattr(quick, name, None)
            if base is not None:
                def init(self, n, batch, _base=base):
                    t = time.perf_counter()
                    _base.__init__(self, n, batch)
                    made.append((int(n), int(batch), time.perf_counter() - t))
                setattr(quick, name, type(name, (base,), {'__init__': init}))

    def plans(since, until=None):
        """The cuFFT plans made from the since-th to the until-th: count, seconds, the slowest [length, batch, ms]."""
        new = made[since:until]
        top = max(new, key=lambda m: m[2]) if new else None
        return dict(count=len(new), seconds=round(sum(m[2] for m in new), 4),
                    slowest=[top[0], top[1], round(1e3 * top[2], 1)] if top else None)

    def save():
        tmp = a.out + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, a.out)
    doc['device_mb'] = a.device_mb
    if a.history:
        d = np.load(a.history)
        S, ant, grid = d['S'], d['ant'].astype(np.float64), (int(d['nx']), int(d['ny']), float(d['spx']), float(d['spy']), d['e1'], d['e2'])
        fmin, df = float(d['fmin']), float(d['df'])
        ref = None
        for backend in backends:
            row = dict(mode='form_image', backend=backend, history=list(S.shape), picture=[grid[0], grid[1]])
            try:
                ts, n0, m0 = [], len(quick_jax.COMPILED) if backend in on_jax else 0, len(made)
                for k in range(a.repeat + 1):
                    t = time.perf_counter()
                    img = fastsar.form_image(S, ant, fmin, df, *grid, algorithm='pfa', backend=backend)
                    ts.append(time.perf_counter() - t)
                    if k == 0 and backend in on_jax:
                        row['compiled'] = compiled(n0)
                    if k == 0:
                        m1 = len(made)
                if backend == 'cuda':
                    row['plans'] = dict(first_call=plans(m0, m1), later_calls=plans(m1))
                    if k < a.repeat:
                        del img
                if backend == 'cpu':
                    ref = img[::5, ::5].astype(np.complex128).ravel()
                elif ref is not None:
                    x = img[::5, ::5].astype(np.complex128).ravel()
                    gain = np.vdot(x, ref) / np.vdot(x, x)
                    row['against_cpu_db'] = round(float(10 * np.log10(np.sum(np.abs(gain * x - ref) ** 2) / np.sum(np.abs(ref) ** 2))), 1)
                del img
                tm, info = {}, {}
                t = time.perf_counter()
                img = quick._pfa(S, ant, fmin, df, *grid, backend=backend, timing=tm, info=info, layout='azimuth')
                if backend == 'cuda':
                    quick._host(img)
                total = time.perf_counter() - t
                del img
                row.update(first_call_seconds=round(ts[0], 3), seconds=round(min(ts[1:]), 3), stages={k: round(float(v), 3) for k, v in tm.items()},
                           stages_seconds=round(total, 3), kept_on_host=info.get('stream'))
                steps = None
                if backend != 'cpu':                      # the steps on the device, one at a time
                    tm = dict(detail={})
                    img = quick._pfa(S, ant, fmin, df, *grid, backend=backend, timing=tm, layout='azimuth')
                    if backend == 'cuda':
                        quick._host(img)
                    del img
                    steps = tm['detail']
                if steps:
                    row['steps'] = {k: round(float(v), 3) for k, v in steps.items()}
                    row['step_shares'] = {k: round(v / max(sum(steps.values()), 1e-9), 3) for k, v in steps.items()}
                print(f"{backend:5s} full image by form_image: {row['seconds']:.2f} s (first {row['first_call_seconds']:.2f}); history "
                      f"{S.shape[0]} x {S.shape[1]}, image {grid[0]} x {grid[1]}" + (f"; {info['stream']}" if info.get('stream') else '')
                      + (f"; against the cpu backend {row['against_cpu_db']:.1f} dB" if 'against_cpu_db' in row else ''), flush=True)
                print('        stages: ' + ', '.join(f'{k} {v:.2f}' for k, v in row['stages'].items()), flush=True)
                if steps:
                    print('        steps: ' + ', '.join(f'{k} {v:.0%}' for k, v in sorted(row['step_shares'].items(), key=lambda kv: -kv[1])), flush=True)
                if 'compiled' in row:
                    print(f"        compiled {row['compiled']['programs']} programs, first calls {row['compiled']['seconds']:.1f} s: "
                          + ', '.join(f'{k} {v:.1f}' for k, v in sorted(row['compiled']['by_name'].items(), key=lambda kv: -kv[1])), flush=True)
            except Exception as e:
                row['failed'] = f'{type(e).__name__}: {str(e)[:300]}'
                print(f'{backend} form_image: {row["failed"]}', flush=True)
            doc['results'].append(row)
            save()
            if a.program and backend != 'cpu':
                row = dict(mode='form_image', backend=backend, program='jax', history=list(S.shape), picture=[grid[0], grid[1]])
                os.environ['FASTSAR_PFA_PROGRAM'] = 'jax'
                try:
                    ts = []
                    for k in range(2):
                        t = time.perf_counter()
                        img = fastsar.form_image(S, ant, fmin, df, *grid, algorithm='pfa', backend=backend)
                        ts.append(time.perf_counter() - t)
                        del img
                    row.update(first_call_seconds=round(ts[0], 3), seconds=round(ts[1], 3))
                    print(f"{backend:5s} full image by the JAX program of 0.3.0: {ts[1]:.2f} s (first {ts[0]:.2f})", flush=True)
                except Exception as e:
                    row['failed'] = f'{type(e).__name__}: {str(e)[:300]}'
                    print(f'{backend} JAX program: {row["failed"]}', flush=True)
                finally:
                    del os.environ['FASTSAR_PFA_PROGRAM']
                doc['results'].append(row)
                save()
        del S
    png = os.path.join(tempfile.mkdtemp(prefix='quicklook-bench-'), 'look.png')
    first = {b: True for b in backends}
    for backend in backends:
        for size in sizes:
            for correct in (True, False):
                kw = dict(resolution='full') if size == 'full' else dict(size=int(size))
                row = dict(mode='quicklook', backend=backend, size=size, correct=correct)
                try:
                    ts, best, n0, m0 = [], None, len(quick_jax.COMPILED) if backend in on_jax else 0, len(made)
                    for k in range(a.repeat + 1):
                        t = time.perf_counter()
                        out = fastsar.quicklook(a.cphd, png, backend=backend, correct=correct, **kw)
                        el = time.perf_counter() - t
                        if k == 0:
                            row['first_call_seconds'] = round(el, 4)
                            row['first_call_stages'] = {k: round(float(v), 4) for k, v in out['timing'].items()}
                            row['first_of_process'] = first[backend]
                            first[backend] = False
                            if backend in on_jax:
                                row['compiled'] = compiled(n0)
                            m1 = len(made)
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
                               auto_takes=quick._auto(P * K), kept_on_host=[n for n in out['notes'] if 'kept on the host' in n])
                    if backend == 'cuda':
                        row['plans'] = dict(first_call=plans(m0, m1), later_calls=plans(m1))
                    if backend != 'cpu' and out['backend'] == backend:
                        # the steps on the device, one at a time
                        col = quick._collection(a.cphd, backend=backend, **(dict(resolution='full') if size == 'full' else dict(size=int(size))))
                        tm = dict(detail={})
                        t = time.perf_counter()
                        img = quick._pfa([col.pop('S')], col['ant'], col['fmin'], col['df'], col['nx'], col['ny'], col['spx'],
                                         col['spy'], col['e1'], col['e2'], correct=correct, backend=backend, complex_image=False,
                                         timing=tm, phase=col['phase'])
                        if backend == 'cuda':
                            cp.cuda.get_current_stream().synchronize()
                        total = time.perf_counter() - t
                        del img, col
                        steps = {k: round(float(v), 4) for k, v in tm['detail'].items()}
                        row['steps'] = steps
                        row['steps_seconds'] = round(total, 4)
                        row['step_shares'] = {k: round(v / max(sum(steps.values()), 1e-9), 3) for k, v in steps.items()}
                    print(f"{backend:5s} {size:>5s} {'corrected  ' if correct else 'uncorrected'} {row['seconds']:8.3f} s "
                          f"(first {row['first_call_seconds']:.3f}); {P} x {K} samples, {row['megabytes_read']:.0f} MB; picture "
                          f"{row['picture'][1]} x {row['picture'][0]}" + (f"; {row['kept_on_host'][0]}" if row['kept_on_host'] else ''),
                          flush=True)
                    print('        stages: ' + ', '.join(f'{k} {v:.3f}' for k, v in row['stages'].items()), flush=True)
                    if 'step_shares' in row:
                        print('        steps: ' + ', '.join(f'{k} {v:.0%}' for k, v in sorted(row['step_shares'].items(), key=lambda kv: -kv[1])), flush=True)
                    if row.get('compiled', {}).get('programs'):
                        print(f"        compiled {row['compiled']['programs']} programs, first calls {row['compiled']['seconds']:.1f} s", flush=True)
                    del out
                except Exception as e:
                    row['failed'] = f'{type(e).__name__}: {str(e)[:300]}'
                    print(f'{backend} {size} corrected={correct}: {row["failed"]}', flush=True)
                doc['results'].append(row)
                save()
    if a.fresh:
        import subprocess
        code = ('import json, sys, time\nt0 = time.perf_counter()\nimport fastsar\nt1 = time.perf_counter()\n'
                'out = fastsar.quicklook(sys.argv[1], sys.argv[2], backend=sys.argv[3], **json.loads(sys.argv[4]))\n'
                't2 = time.perf_counter()\nc = out["cut"]\n'
                'print(json.dumps(dict(import_seconds=t1 - t0, seconds=t2 - t1, backend=out["backend"], '
                'samples=(c["pulses"][1] - c["pulses"][0]) * (c["samples"][1] - c["samples"][0]))))\n')
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [os.path.dirname(os.path.dirname(os.path.abspath(fastsar.__file__)))] + [p for p in [os.environ.get('PYTHONPATH')] if p]))
        doc['fresh'] = []
        for backend in backends:
            for size in sizes:
                kw = dict(resolution='full') if size == 'full' else dict(size=int(size))
                row = dict(backend=backend, size=size, seconds=[], import_seconds=[])
                try:
                    for _ in range(a.fresh):
                        r = subprocess.run([sys.executable, '-c', code, a.cphd, png, backend, json.dumps(kw)], env=env,
                                           capture_output=True, text=True, check=True)
                        got = json.loads(r.stdout.strip().splitlines()[-1])
                        row['seconds'].append(round(got['seconds'], 4))
                        row['import_seconds'].append(round(got['import_seconds'], 4))
                        row.update(samples=got['samples'], formed_on=got['backend'])
                    print(f"{backend:5s} {size:>5s} first look of a new process: {min(row['seconds']):.3f} s (of "
                          f"{', '.join(f'{s:.3f}' for s in row['seconds'])}); {row['samples']:.3g} samples", flush=True)
                except Exception as e:
                    row['failed'] = f'{type(e).__name__}: {str(getattr(e, "stderr", "") or e)[-300:]}'
                    print(f'{backend} {size} fresh: {row["failed"]}', flush=True)
                doc['fresh'].append(row)
                save()
    save()
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
