"""pytest configuration for the FastSAR tests.

Hardware. Tests marked `cuda` need CuPy with an Nvidia GPU, tests marked `tpu` a Cloud TPU. Without the device they
are skipped; `--require cuda,tpu` (a comma list) makes a missing device a failure, for runs on those machines.

Optional packages. Tests marked `needs('sarpy', ...)`, or calling the `need` fixture, are skipped
(pytest.importorskip) when a package of the `test` dependency group or extra is missing; `--require io` makes that
a failure.

Memory. Each test's peak resident memory (the process and its children, from its setup to the end of its call,
module fixtures included) is compared with `--max-rss-gb` (default: $FASTSAR_TEST_MAX_GB, else 12 GB) and the test
fails if it is above. A thread samples the memory five times a second and stops the process if it reaches 1.5
times the limit, so that a runaway test cannot exhaust a shared machine (under pytest-xdist the worker is replaced
and the test reported as crashed). `--max-rss-gb 0` turns both off. The terminal summary lists the peak of each
module.

Isolation. The former runner gave each script its own process. The modules now share one, so each undoes what it
changes: environment variables and patched functions through monkeypatch, the stand-in sarpy modules of
test_cphd.py through a module fixture, and JAX's CPU device (jax_platform_name in the scripts) through the jax_cpu
fixture below, which also sets the default device, so the dense references stay on the CPU of a GPU or TPU host
whatever ran before. `pytest -n 4 --dist loadfile` (pytest-xdist) runs modules in parallel worker processes.
"""
import importlib
import importlib.util
import os
import resource
import signal
import sys
import threading

import pytest

HARDWARE = ('cuda', 'tpu')
REQUIRABLE = HARDWARE + ('io',)
KILL_FACTOR = 1.5
_missing = {}


def pytest_addoption(parser):
    g = parser.getgroup('fastsar')
    g.addoption('--require', default='', metavar='LIST',
                help='comma list of cuda, tpu, io: a missing device (cuda, tpu) or optional test package (io) '
                     'fails the tests that need it instead of skipping them')
    g.addoption('--max-rss-gb', type=float, default=None, metavar='GB',
                help='resident memory limit per test in GB (default $FASTSAR_TEST_MAX_GB, else 12; 0 turns it off); '
                     'a test above it fails, a process at 1.5 times it is stopped')


# ------------------------------------------------------------------------------------------------ hardware, packages

def missing(need):
    """None when the hardware is present, else the reason it is not (cached)."""
    if need not in _missing:
        why = None
        if need == 'cuda':
            try:
                import cupy
                if cupy.cuda.runtime.getDeviceCount() == 0:
                    why = 'no CUDA device'
            except Exception:
                why = 'no CuPy with a CUDA device'
        elif need == 'tpu':
            why = 'no TPU'
            try:
                import jax
                if any(d.platform == 'tpu' for d in jax.devices()):
                    why = None
            except Exception:
                pass
        _missing[need] = why
    return _missing[need]


def pytest_configure(config):
    req = {r.strip() for r in config.getoption('require').split(',') if r.strip()}
    unknown = req - set(REQUIRABLE)
    if unknown:
        raise pytest.UsageError(f'--require: unknown {", ".join(sorted(unknown))} (one of {", ".join(REQUIRABLE)})')
    config._fastsar_require = req
    gb = config.getoption('max_rss_gb')
    if gb is None:
        gb = float(os.environ.get('FASTSAR_TEST_MAX_GB', 12.0))
    config._fastsar_cap = gb * 1e9
    config._fastsar_peaks = {}
    config._fastsar_hw_skips = {}
    # the pytest-xdist controller runs no tests, and its workers are its children: each worker watches itself
    xdist_controller = not hasattr(config, 'workerinput') and (config.getoption('numprocesses', None) or 0) != 0
    config._fastsar_watch = _Watchdog(gb * 1e9 * KILL_FACTOR, config) if gb > 0 and not xdist_controller else None


def pytest_report_header(config):
    cap = config._fastsar_cap
    req = ', '.join(sorted(config._fastsar_require)) or 'none'
    return (f'fastsar: memory limit {cap / 1e9:g} GB per test' if cap else 'fastsar: no memory limit') + \
           f'; required: {req}'


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    req = item.config._fastsar_require
    for hw in HARDWARE:
        if item.get_closest_marker(hw):
            why = missing(hw)
            if why:
                if hw in req:
                    pytest.fail(f'{why} (--require {hw})', pytrace=False)
                pytest.skip(f'{why} (marked {hw})')
    for mark in item.iter_markers('needs'):
        for name in mark.args:
            _import_or_skip(name, 'io' in req)


def _import_or_skip(name, required):
    """The module name; if it is missing, the test is skipped (pytest.importorskip) or, when required, fails."""
    if not required:
        return pytest.importorskip(name, reason=f'needs {name} (uv sync --group test, or pip install -e ".[io,test]")')
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.fail(f'needs {name} (--require io): {e}; install the test group (uv sync --group test) or extra '
                    '(pip install -e ".[io,test]")', pytrace=False)


@pytest.fixture
def need(request):
    """need('module') inside a test: the module, or a skip (a failure with --require io) when it is missing; for
    imports that apply only to some cases of a test, where the needs marker would cover them all."""
    return lambda name: _import_or_skip(name, 'io' in request.config._fastsar_require)


@pytest.fixture(scope='module')
def jax_cpu():
    """JAX on its CPU device for a module (the scripts set jax_platform_name to cpu when they started): the
    platform setting, which takes effect only before JAX picks its default backend, and the default device, which
    takes effect at any time; both restored afterwards."""
    import jax
    old = jax.config.values.get('jax_platform_name')
    jax.config.update('jax_platform_name', 'cpu')
    try:
        with jax.default_device(jax.devices('cpu')[0]):
            yield
    finally:
        jax.config.update('jax_platform_name', old)


# ------------------------------------------------------------------------------------------------ memory

def _proc_rss(pid):
    """Resident memory (bytes) of pid and its descendants, from /proc (None where /proc is unavailable)."""
    total, todo, seen = 0, [pid], False
    while todo:
        p = todo.pop()
        try:
            with open(f'/proc/{p}/status') as fh:
                for line in fh:
                    if line.startswith('VmRSS:'):
                        total += int(line.split()[1]) * 1024
                        seen = True
            for t in os.listdir(f'/proc/{p}/task'):
                with open(f'/proc/{p}/task/{t}/children') as fh:
                    todo += [int(c) for c in fh.read().split()]
        except (OSError, ValueError):
            pass
    return total if seen else None


def _hwm():
    """Peak resident memory (bytes) of this process since the last _reset_hwm (VmHWM), or since its start
    (getrusage) where /proc is unavailable."""
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == 'darwin' else r * 1024


def _reset_hwm():
    try:
        with open('/proc/self/clear_refs', 'w') as fh:       # 5: reset the peak resident set size (Linux 4.0)
            fh.write('5')
    except OSError:
        pass


class _Watchdog:
    """Samples the resident memory of this process and its children; records the peak since reset() and stops the
    process when it reaches `kill` bytes."""

    def __init__(self, kill, config):
        self.kill, self.config, self.peak, self.test = kill, config, 0, None
        if _proc_rss(os.getpid()) is not None:
            threading.Thread(target=self._run, daemon=True, name='fastsar-rss').start()

    def reset(self, test):
        self.peak, self.test = 0, test

    def _run(self):
        ev = threading.Event()
        while not ev.wait(0.2):
            r = _proc_rss(os.getpid()) or 0
            self.peak = max(self.peak, r)
            if r >= self.kill:
                try:                                    # past pytest's capture of the test's output
                    self.config.pluginmanager.getplugin('capturemanager').suspend_global_capture()
                except Exception:
                    pass
                sys.stderr.write(f'\nfastsar: {self.test} holds {r / 1e9:.1f} GB of memory, 1.5 times the limit '
                                 '(--max-rss-gb, FASTSAR_TEST_MAX_GB): stopping the process\n')
                sys.stderr.flush()
                os.kill(os.getpid(), signal.SIGKILL)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    _reset_hwm()
    w = item.config._fastsar_watch
    if w is not None:
        w.reset(item.nodeid)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when != 'call':
        return
    w = item.config._fastsar_watch
    peak = max(_hwm(), w.peak if w is not None else 0)
    rep.user_properties.append(('peak_rss_bytes', peak))
    cap = item.config._fastsar_cap
    if cap and peak > cap and rep.passed:
        rep.outcome = 'failed'
        rep.longrepr = (f'peak resident memory {peak / 1e9:.1f} GB above the limit of {cap / 1e9:g} GB '
                        '(--max-rss-gb, FASTSAR_TEST_MAX_GB)')


def pytest_runtest_logreport(report):
    """Collects the peak memory of each module (on the controller under pytest-xdist) and the hardware skips."""
    cfg = _config
    if cfg is None:
        return
    for k, v in report.user_properties:
        if k == 'peak_rss_bytes':
            mod = report.nodeid.split('::')[0]
            cfg._fastsar_peaks[mod] = max(cfg._fastsar_peaks.get(mod, 0), v)
    if report.skipped and isinstance(report.longrepr, tuple):
        msg = str(report.longrepr[2])
        for hw in HARDWARE:
            if f'(marked {hw})' in msg:
                cfg._fastsar_hw_skips[hw] = cfg._fastsar_hw_skips.get(hw, 0) + 1


_config = None


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    global _config
    _config = session.config


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    peaks = config._fastsar_peaks
    if peaks:
        tr = terminalreporter
        tr.section('peak resident memory per test module')
        for mod, v in sorted(peaks.items()):
            tr.write_line(f'{v / 1e9:6.2f} GB  {mod}')
    for hw, n in sorted(config._fastsar_hw_skips.items()):
        terminalreporter.write_line(f'not tested on this machine: {n} {hw} tests skipped '
                                    f'(run on such a machine with --require {hw})')
