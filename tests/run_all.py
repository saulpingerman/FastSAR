"""Run every tests/test_*.py as its own process and print a summary; exits non-zero if any script fails.

    python tests/run_all.py                    # all scripts, 600 s each
    python tests/run_all.py -t 180 api units   # the scripts whose names contain 'api' or 'units'

A script that needs hardware or a package this machine lacks (a GPU with CuPy, a TPU, finufft, rasterio) is reported
as SKIP with the reason. --max-rss-gb stops a script whose resident memory (with its children) exceeds the limit, so
a runaway test fails instead of exhausting the machine. Each script's output is printed after it finishes (all of it
with -v, else the last lines of a failure).
"""
import argparse, glob, importlib.util, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# what each script needs beyond the library's dependencies
NEEDS = {
    'test_ffbp_cuda.py': ('cuda',),
    'test_insar.py': ('finufft',),
}


def _have(need):
    """None when the requirement is met, else the reason it is not."""
    if need == 'cuda':
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() > 0:
                return None
            return 'no CUDA device'
        except Exception:
            return 'no CuPy with a CUDA device'
    if need == 'tpu':
        try:
            import jax
            if any(d.platform == 'tpu' for d in jax.devices()):
                return None
        except Exception:
            pass
        return 'no TPU'
    return None if importlib.util.find_spec(need) is not None else f'needs {need}'


def _rss(pid):
    """Resident memory (bytes) of pid and its descendants, from /proc (0 where /proc is unavailable)."""
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        try:
            with open(f'/proc/{p}/status') as fh:
                for line in fh:
                    if line.startswith('VmRSS:'):
                        total += int(line.split()[1]) * 1024
            for t in os.listdir(f'/proc/{p}/task'):
                with open(f'/proc/{p}/task/{t}/children') as fh:
                    todo += [int(c) for c in fh.read().split()]
        except (OSError, ValueError):
            pass
    return total


def run(path, timeout, max_rss):
    """-> (status, seconds, peak RSS bytes, output)."""
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'           # the output up to a kill shows where it happened
    env['PYTHONPATH'] = ROOT + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, path], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    import threading
    chunks = []
    reader = threading.Thread(target=lambda: chunks.append(proc.stdout.read()), daemon=True)
    reader.start()
    peak, status = 0, None
    while proc.poll() is None:
        peak = max(peak, _rss(proc.pid))
        if peak > max_rss:
            status = f'FAIL (memory above {max_rss / 1e9:.1f} GB)'
        elif time.perf_counter() - t0 > timeout:
            status = f'FAIL (timeout {timeout:.0f} s)'
        if status:
            os.killpg(proc.pid, 9)
            break
        time.sleep(0.2)
    proc.wait()
    reader.join(10)
    out = (chunks[0] if chunks else b'').decode(errors='replace')
    if status is None:
        status = 'PASS' if proc.returncode == 0 else f'FAIL (exit {proc.returncode})'
    return status, time.perf_counter() - t0, peak, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('names', nargs='*', help='run only the scripts whose names contain one of these')
    ap.add_argument('-t', '--timeout', type=float, default=600.0, help='seconds per script (default 600)')
    ap.add_argument('--max-rss-gb', type=float, default=float(os.environ.get('FASTSAR_TEST_MAX_GB', 6.0)),
                    help='resident memory limit per script (default 6, or FASTSAR_TEST_MAX_GB)')
    ap.add_argument('-v', '--verbose', action='store_true', help='print the full output of every script')
    a = ap.parse_args()
    scripts = sorted(glob.glob(os.path.join(HERE, 'test_*.py')))
    if a.names:
        scripts = [s for s in scripts if any(n in os.path.basename(s) for n in a.names)]
    rows = []
    for s in scripts:
        name = os.path.basename(s)
        why = next((r for r in map(_have, NEEDS.get(name, ())) if r), None)
        if why:
            print(f'SKIP  {name}: {why}', flush=True)
            rows.append((name, 'SKIP', 0.0, 0))
            continue
        status, dt, peak, out = run(s, a.timeout, a.max_rss_gb * 1e9)
        print(f'{status.split()[0]:5s} {name}  {dt:6.1f} s  {peak / 1e9:4.1f} GB' +
              ('' if status == 'PASS' else f'  {status}'), flush=True)
        if a.verbose:
            print(out)
        elif status != 'PASS':
            print('    ' + '\n    '.join(out.rstrip().splitlines()[-25:]))
        rows.append((name, status, dt, peak))
    npass = sum(r[1] == 'PASS' for r in rows)
    nskip = sum(r[1] == 'SKIP' for r in rows)
    bad = [r for r in rows if r[1] not in ('PASS', 'SKIP')]
    print(f'\n{npass} passed, {len(bad)} failed, {nskip} skipped, {sum(r[2] for r in rows):.0f} s')
    for r in bad:
        print(f'  {r[0]}: {r[1]}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
