"""Run the test suite with pytest, with the options of the former script runner; exits non-zero if any test fails.

    python tests/run_all.py                    # every test module, at most 600 s per test
    python tests/run_all.py -t 180 api units   # the modules whose names contain 'api' or 'units'
    python tests/run_all.py --require cuda -t 1500 --max-rss-gb 12

The tests are pytest modules (`uv run pytest`, or `nox`); this wrapper is kept for scripts that call it. It runs
`python -m pytest` on the selected modules with:

  -t SECONDS        --timeout SECONDS per test (pytest-timeout; the former runner timed each script)
  --max-rss-gb GB   the same option of tests/conftest.py: a test whose peak resident memory exceeds it fails, and a
                    process at 1.5 times it is stopped (default $FASTSAR_TEST_MAX_GB, else 12)
  --require LIST    the same option: cuda, tpu (a missing device fails instead of skipping); io (a missing
                    optional test package fails) is always added, as the former runner failed such a script
  -v                -v -s: every test's name and its printed measurements

Any other arguments after `--` go to pytest unchanged. pytest prints the summary (passed, failed, skipped) and
the peak memory of each module.
"""
import argparse
import glob
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], epilog='arguments after -- go to pytest')
    ap.add_argument('names', nargs='*', help='run only the modules whose names contain one of these')
    ap.add_argument('-t', '--timeout', type=float, default=600.0, help='seconds per test (default 600)')
    ap.add_argument('--max-rss-gb', type=float, default=None,
                    help='resident memory limit per test (default $FASTSAR_TEST_MAX_GB, else 12)')
    ap.add_argument('-v', '--verbose', action='store_true', help='print every test and its output')
    ap.add_argument('--require', default='', help='cuda, tpu, io that must be present, e.g. cuda,tpu (absent: failure)')
    argv = sys.argv[1:]
    extra = argv[argv.index('--') + 1:] if '--' in argv else []
    a = ap.parse_args(argv[:argv.index('--')] if '--' in argv else argv)
    files = sorted(glob.glob(os.path.join(HERE, 'test_*.py')))
    if a.names:
        files = [f for f in files if any(n in os.path.basename(f) for n in a.names)]
        if not files:
            sys.exit(f'no test module matches {" ".join(a.names)}')
    cmd = [sys.executable, '-m', 'pytest'] + [os.path.relpath(f, ROOT) for f in files]
    if importlib.util.find_spec('pytest_timeout') is not None:
        cmd += ['--timeout', f'{a.timeout:g}']
    else:
        print('run_all: pytest-timeout is not installed, so tests have no time limit (uv sync --group test)', flush=True)
    if a.max_rss_gb is not None:
        cmd += ['--max-rss-gb', f'{a.max_rss_gb:g}']
    # as with the former runner, a missing test package is a failure, and missing hardware one only if required
    cmd += ['--require', ','.join(['io'] + [r.strip() for r in a.require.split(',') if r.strip() not in ('', 'io')])]
    cmd += ['-v', '-s'] if a.verbose else ['-q']
    cmd += extra
    env = dict(os.environ, PYTHONUNBUFFERED='1')      # the output up to a stop shows where it happened
    print(' '.join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=ROOT, env=env))


if __name__ == '__main__':
    main()
