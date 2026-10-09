"""Checks of the planning that runs before or around image formation, on the CPU (JAX on its CPU device):

  patches.choose_buckets       against an exhaustive search over bucket sets; empty, single and equal inputs, zero
                               counts, the compile-cost trade-off, and its time for thousands of distinct lengths
  memory (CUDA sizing)         full_speed('cuda') is the least free memory at which ffbp_cuda's sizing (streaming
                               threshold, group size, image) runs at full speed, for float32 and float16 storage
  memory (TPU sizing)          _jax_group from the device's limit, not its free memory; full_speed('tpu')
  ImageFormer.memory()         full_speed against the free memory reported for cuda and tpu (patched)
  TPU out-of-memory retry      a program that raises RESOURCE_EXHAUSTED (at once, or when its result is read) is
                               rebuilt for half its effective group, with a MemoryWarning; groups of 1 raise
                               MemoryError; other errors pass through
  program cache                formers built on several threads at once build one program per signature
  environment variables        integer and float FASTSAR_* settings below their minimum or not numbers raise ValueError

Each check prints its value and limit; a test fails if any of its checks fails."""
import itertools
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

import fastsar
from fastsar import patches as pt
from fastsar import memory as fmem, api, ffbp2, sim


def check(bad, name, value, limit, fmt='{:.0f}'):
    ok = value <= limit
    print(f'  {name:66s} {fmt.format(value):>10s}  (limit {fmt.format(limit)})' + ('' if ok else '  FAIL'))
    if not ok:
        bad.append(name)


def done(bad):
    assert not bad, 'FAILED: ' + '; '.join(bad)


# ---------------------------------------------------------------------------------------------- choose_buckets

def cost(buckets, counts, cc, step):
    c = step * -(-np.asarray(counts, np.int64) // step)
    b = np.asarray(buckets)
    return cc * len(b) + float(np.sum(b[np.searchsorted(b, c)] / c - 1.0))


def brute(counts, cc, step):
    """The cheapest bucket set by enumeration: any subset of the distinct counts that holds the largest."""
    u = np.unique(step * -(-np.asarray(counts, np.int64) // step))
    best = None
    for r in range(len(u)):
        for sub in itertools.combinations(u[:-1], r):
            b = list(sub) + [u[-1]]
            c = cost(b, counts, cc, step)
            if best is None or c < best - 1e-12:
                best = c
    return best


def test_choose_buckets():
    print('choose_buckets')
    bad = []
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        n, step = int(rng.integers(1, 30)), int(rng.choice([1, 2, 256]))
        counts = rng.integers(1, int(rng.choice([20, 3000])), n)
        cc = float(rng.choice([0.0, 0.3, 3.0, 11.0, rng.uniform(0, 5)]))
        if len(np.unique(step * -(-counts // step))) > 9:
            continue
        worst = max(worst, cost(pt.choose_buckets(counts, cc, step), counts, cc, step) - brute(counts, cc, step))
    check(bad, 'cost above the exhaustive optimum (random inputs)', worst, 1e-9, '{:.1e}')
    check(bad, 'empty input: no buckets', len(pt.choose_buckets([], 11)), 0)
    check(bad, 'zero and negative counts are ignored', abs(len(pt.choose_buckets([0, -3, 0], 11))), 0)
    check(bad, 'a single count: one bucket, rounded up to the step', abs(pt.choose_buckets([1000], 11)[0] - 1024), 0)
    check(bad, 'equal counts: one bucket', abs(len(pt.choose_buckets([5000] * 40, 11)) - 1), 0)
    check(bad, 'zero compile cost: a bucket per distinct count', abs(len(pt.choose_buckets([256, 512, 768, 1024], 0.0)) - 4), 0)
    check(bad, 'large compile cost: one bucket at the largest count', abs(len(pt.choose_buckets([256, 512, 768, 1024], 1e9)) - 1), 0)
    c = rng.integers(1000, 7000, 4000)
    t = time.perf_counter()
    pt.choose_buckets(c, 11.0, step=2)
    dt = time.perf_counter() - t
    print(f'  ({len(np.unique(2 * -(-c // 2)))} distinct gate lengths planned in {dt:.3f} s)')
    check(bad, 'time for thousands of distinct gate lengths (s)', dt, 1.0, '{:.3f}')
    done(bad)


# ---------------------------------------------------------------------------------------------- CUDA sizing

def fake_plan(P, K, Ko, Po, C, Nx, Ny):
    return dict(levels=[dict(P=P, K=K, Ko=Ko, Po=Po, C=C)], Nx=Nx, Ny=Ny)


def cuda_run(plan, K, isz, free_total):
    """The sizing steps of ffbp_cuda's form for a device with free_total bytes free: -> (stream, children per group)."""
    lv0 = plan['levels'][0]
    planes = isz * lv0['P'] * K
    free = free_total - 8.0 * plan['Nx'] * plan['Ny']          # the image, allocated first
    stream = fmem.cuda_streams(planes, free)
    return stream, fmem.cuda_group(lv0, K, isz, free - (0 if stream else planes), stream)


def test_cuda_sizing():
    print('\nCUDA memory model (memory.full_speed, cuda_streams, cuda_group)')
    bad = []
    nbad, ncase = 0, 0
    for P, K, Ko, Po, C, Nx, Ny in ((2000, 2000, 500, 250, 16, 2048, 2048), (74203, 17282, 2200, 9300, 64, 20000, 20000),
                                    (52000, 12000, 3000, 6500, 8, 12207, 8808), (40000, 30000, 600, 5000, 4, 1000, 1000),
                                    (8000, 4000, 1000, 1000, 2, 30000, 30000)):
        plan = fake_plan(P, K, Ko, Po, C, Nx, Ny)
        for isz in (8, 4):
            ncase += 1
            need = fmem.full_speed(plan, K, 'cuda', isz)[0]
            full = cuda_run(plan, K, isz, need * 1.0001) == (False, min(fmem.CUDA_GROUP, C))
            short = cuda_run(plan, K, isz, need * 0.995) != (False, min(fmem.CUDA_GROUP, C))
            if not (full and short):
                nbad += 1
                print(f'    P {P} K {K} isz {isz}: need {need / 1e9:.2f} GB, at need {cuda_run(plan, K, isz, need * 1.0001)}, '
                      f'below {cuda_run(plan, K, isz, need * 0.995)}')
    check(bad, f'full_speed is the threshold of the sizing ({ncase} plans and storage types)', nbad, 0)
    # a 10.3 GB history with 4 GB of groups fits 16 GB as planes plus groups, but streams (10.3 GB > half of 16 GB)
    plan = fake_plan(74203, 17282, 470, 9300, 64, 4000, 4000)
    need, parts = fmem.full_speed(plan, 17282, 'cuda')
    print(f'  (history {parts["history"] / 1e9:.1f} GB, groups {parts["first_level"] / 1e9:.1f} GB: full speed needs '
          f'{need / 1e9:.1f} GB; with 16 GB free the sizing streams: {cuda_run(plan, 17282, 8, 16e9)[0]})')
    check(bad, 'a history above half the free memory is not reported as full speed', float(need <= 16e9), 0)
    check(bad, 'float16 storage halves the history part', abs(fmem.full_speed(plan, 17282, 'cuda', 4)[1]['history'] * 2 - parts['history']), 0)
    done(bad)


# ---------------------------------------------------------------------------------------------- TPU sizing

TPU_PLAN = fake_plan(74203, 17282, 2200, 9300, 64, 20000, 20000)


def test_tpu_sizing(monkeypatch):
    print('\nTPU memory model (api._jax_group, memory.full_speed)')
    bad = []
    plan = TPU_PLAN
    need = fmem.full_speed(plan, 17282, 'tpu')[0]
    with monkeypatch.context() as mp:
        mp.setattr(fmem, 'device_available', lambda b: (_ for _ in ()).throw(AssertionError('sized from the free memory')))
        mp.setattr(fmem, 'tpu_capacity', lambda: need * 1.0001)
        g_full = api._jax_group(plan, 17282)
        mp.setattr(fmem, 'tpu_capacity', lambda: need * 0.98)
        g_short = api._jax_group(plan, 17282)
        mp.setattr(fmem, 'tpu_capacity', lambda: 1e9)
        g_none = api._jax_group(plan, 17282)
    print(f'  (full speed needs {need / 1e9:.1f} GB; groups at that limit {g_full}, 2% below {g_short}, at 1 GB {g_none})')
    check(bad, 'groups of TPU_GROUP at the full-speed limit', abs(g_full - fmem.TPU_GROUP), 0)
    check(bad, 'smaller groups below it', float(g_short >= fmem.TPU_GROUP), 0)
    check(bad, 'groups of 1 when nothing fits', abs(g_none - 1), 0)
    done(bad)


# ---------------------------------------------------------------------------------------------- ImageFormer.memory()

@pytest.fixture(scope='module')
def jax_former():
    """A JAX former of a small scene, shared in order by the tests below as one former was in the script."""
    rng = np.random.default_rng(3)
    col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
    tg = np.stack([rng.uniform(-25, 25, 20), rng.uniform(-25, 25, 20), np.zeros(20)], 1)
    S = sim.simulate_brute(col, tg, rng.standard_normal(20) + 1j * rng.standard_normal(20)).astype(np.complex64)
    grid = (128, 128, 0.5, 0.5, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    fj = fastsar.ImageFormer(col.ant, col.fmin, col.df, col.K, *grid, backend='jax', window=False)
    return SimpleNamespace(S=S, fj=fj)


def test_former_memory(jax_former, monkeypatch):
    print('\nImageFormer.memory() on cuda and tpu (patched free memory)')
    bad = []
    fj = jax_former.fj
    try:
        with monkeypatch.context() as mp:
            fj.backend = 'cuda'                    # the bookkeeping of a cuda former, without a GPU
            need32 = fmem.full_speed(fj._plan, fj.K, 'cuda')[0]
            mp.setattr(fmem, 'cuda_free', lambda: need32)
            m_at = fj.memory()
            mp.setattr(fmem, 'cuda_free', lambda: need32 * 0.99)
            m_below = fj.memory()
            fj.precision = 'float16'
            m16 = fj.memory()
            fj.backend, fj.precision = 'tpu', 'float32'
            need_t = fmem.full_speed(fj._plan, fj.K, 'tpu')[0]
            mp.setattr(fmem, '_jax_stats', lambda: dict(bytes_limit=need_t + 1e9, bytes_in_use=0.5e9))
            m_tpu = fj.memory()
            mp.setattr(fmem, '_jax_stats', lambda: dict(bytes_limit=need_t + 1e9, bytes_in_use=2e9))
            m_tpu_busy = fj.memory()
    finally:
        fj.backend, fj.precision = 'jax', 'float32'
    check(bad, 'cuda: full speed with exactly the memory it needs', float(not m_at['full_speed']), 0)
    check(bad, 'cuda: not full speed 1% below it', float(m_below['full_speed']), 0)
    check(bad, 'cuda float16: needs less than float32', float(m16['needed'] >= m_at['needed']), 0)
    check(bad, 'tpu: full speed when the free memory covers the need', float(not m_tpu['full_speed']), 0)
    check(bad, 'tpu: not full speed when other arrays take the margin', float(m_tpu_busy['full_speed']), 0)
    done(bad)


# ---------------------------------------------------------------------------------------------- TPU retry

def test_tpu_out_of_memory_retry(jax_former):
    print('\nout-of-memory retry of the JAX/TPU former (stub programs)')
    from jax.errors import JaxRuntimeError
    bad = []
    fj, S = jax_former.fj, jax_former.S
    img_ref = fj(S)
    real = fj._fn

    class Stub:
        """A program that runs the real one, reporting group size ng, failing as told: 'now' at the call, 'later'
        when its result is read (asynchronous dispatch), or with another error."""

        def __init__(self, ng, fail=None, msg='RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.0G'):
            self.stages, self.pad, self.fail, self.msg = dict(real.stages, ng=ng), real.pad, fail, msg

        def __call__(self, hre, him, arrs):
            if self.fail == 'now':
                raise JaxRuntimeError(self.msg)
            if self.fail == 'later':
                msg = self.msg

                class Pending:
                    def block_until_ready(self):
                        raise JaxRuntimeError(msg)
                return Pending(), Pending()
            return real(hre, him, arrs)

    def run(first, next_fail=None):
        """fj(S) with fj._fn = first and every rebuilt program a Stub failing as next_fail says (by its group): ->
        (image or exception, groups asked for, MemoryWarning messages)."""
        asked = []

        def program(ng):
            asked.append(ng)
            return Stub(ng, (next_fail or {}).get(ng)), None
        fj._fn, fj._program = first, program
        try:
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                try:
                    out = fj(S)
                except Exception as e:
                    out = e
        finally:
            fj._fn = real
            del fj._program
        return out, asked, [str(w.message) for w in wl if issubclass(w.category, fastsar.MemoryWarning)]

    out, asked, msgs = run(Stub(4, 'now'))
    print(f'  groups of 4 out of memory: rebuilt for {asked}; {msgs}')
    check(bad, 'retried once, for half the group', float(asked != [2]), 0)
    check(bad, 'one MemoryWarning naming groups of 2', float(len(msgs) != 1 or 'groups of 2' not in msgs[0]), 0)
    check(bad, 'the retried image equals the image', float(np.abs(out - img_ref).max()) if isinstance(out, np.ndarray) else 1.0, 0, '{:.1e}')
    out, asked, msgs = run(Stub(4, 'later'))
    check(bad, 'an error raised when the result is read is retried too', float(asked != [2] or not isinstance(out, np.ndarray)), 0)
    out, asked, msgs = run(Stub(3, 'now'), {1: 'now'})
    print(f'  effective group of 3 out of memory, then 1: rebuilt for {asked}; {type(out).__name__}')
    check(bad, 'halves the effective group (3 -> 1), then MemoryError at 1', float(asked != [1] or not isinstance(out, MemoryError)), 0)
    out, asked, msgs = run(Stub(1, 'now'))
    check(bad, 'groups of 1 out of memory: MemoryError, no rebuild', float(asked != [] or not isinstance(out, MemoryError)), 0)
    out, asked, msgs = run(Stub(4, 'now', 'RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem'))
    check(bad, 'a VMEM error passes through, no rebuild', float(asked != [] or not isinstance(out, JaxRuntimeError)), 0)
    out, asked, msgs = run(Stub(4, 'now', 'INTERNAL: something else'))
    check(bad, 'another runtime error passes through', float(asked != [] or not isinstance(out, JaxRuntimeError)), 0)
    done(bad)


# ---------------------------------------------------------------------------------------------- program cache

def test_program_cache_threads(jax_former, monkeypatch):
    print('\nJAX program cache under concurrent formers')
    bad = []
    fj = jax_former.fj
    calls = []
    real_make = ffbp2.make_ffbp

    def slow_make(*a, **k):
        calls.append(1)
        time.sleep(0.2)                       # a build long enough for the other threads to look up the same key
        return real_make(*a, **k)

    api._JAX_PROGRAMS.clear()
    with monkeypatch.context() as mp:
        mp.setattr(ffbp2, 'make_ffbp', slow_make)
        with ThreadPoolExecutor(4) as ex:
            got = list(ex.map(lambda _: fj._program(1)[0], range(4)))
    check(bad, '4 threads, one signature: programs built', abs(len(calls) - 1), 0)
    check(bad, 'every thread got the same program', float(any(g is not got[0] for g in got)), 0)
    done(bad)


# ---------------------------------------------------------------------------------------------- environment

def test_environment_variables(monkeypatch):
    print('\nenvironment variables')
    bad = []
    nbad = 0
    for name, value, least, kind, ok in (('FASTSAR_TPU_GROUP', '0', 1, int, False), ('FASTSAR_TPU_GROUP', '2', 1, int, True),
                                         ('FASTSAR_CUDA_GROUP', 'eight', 1, int, False),
                                         ('FASTSAR_MOSAIC_PREFETCH', '-1', 0, int, False),
                                         ('FASTSAR_MOSAIC_PREFETCH', '0', 0, int, True),
                                         ('FASTSAR_PULSE_SLACK', '-0.1', 0.0, float, False),
                                         ('FASTSAR_CPU_GROUP_GB', '1.5', 0.0, float, True)):
        with monkeypatch.context() as mp:
            mp.setenv(name, value)
            try:
                fmem._env_number(name, least, kind)
                res = 'accepted'
            except ValueError as e:
                res = f'ValueError: {e}'
        nbad += (res == 'accepted') != ok
        print(f'    {name}={value}: {res}')
    check(bad, 'settings outside their range raise ValueError, others are accepted', nbad, 0)
    with monkeypatch.context() as mp:
        mp.setenv('FASTSAR_TPU_GROUP', '0')
        try:
            api._jax_group(TPU_PLAN, 17282)
            r = 'no error'
        except ValueError:
            r = 'ValueError'
    check(bad, 'FASTSAR_TPU_GROUP=0 raises ValueError in the TPU sizing', float(r != 'ValueError'), 0)
    done(bad)
