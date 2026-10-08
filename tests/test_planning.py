"""Checks of the planning that runs before or around image formation, on the CPU (JAX on its CPU device):

  patches.choose_buckets       against an exhaustive search over bucket sets; empty, single and equal inputs, zero
                               counts, the compile-cost trade-off, and its time for thousands of distinct lengths
  memory (CUDA sizing)         full_speed('cuda') is the least free memory at which ffbp_cuda's sizing (streaming
                               threshold, group size, image) runs at full speed, for float32 and float16 storage
  ImageFormer.memory()         full_speed against the free memory reported for cuda and tpu (patched)
  environment variables        integer and float FASTSAR_* settings below their minimum or not numbers raise ValueError

Each check prints its value and limit; the script exits non-zero if any fails."""
import itertools, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from fastsar import patches as pt

bad = []


def check(name, value, limit, fmt='{:.0f}'):
    ok = value <= limit
    print(f'  {name:66s} {fmt.format(value):>10s}  (limit {fmt.format(limit)})' + ('' if ok else '  FAIL'))
    if not ok:
        bad.append(name)


t_start = time.perf_counter()

# ---------------------------------------------------------------------------------------------- choose_buckets
print('choose_buckets')


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


rng = np.random.default_rng(1)
worst = 0.0
for _ in range(300):
    n, step = int(rng.integers(1, 30)), int(rng.choice([1, 2, 256]))
    counts = rng.integers(1, int(rng.choice([20, 3000])), n)
    cc = float(rng.choice([0.0, 0.3, 3.0, 11.0, rng.uniform(0, 5)]))
    if len(np.unique(step * -(-counts // step))) > 9:
        continue
    worst = max(worst, cost(pt.choose_buckets(counts, cc, step), counts, cc, step) - brute(counts, cc, step))
check('cost above the exhaustive optimum (random inputs)', worst, 1e-9, '{:.1e}')
check('empty input: no buckets', len(pt.choose_buckets([], 11)), 0)
check('zero and negative counts are ignored', abs(len(pt.choose_buckets([0, -3, 0], 11))), 0)
check('a single count: one bucket, rounded up to the step', abs(pt.choose_buckets([1000], 11)[0] - 1024), 0)
check('equal counts: one bucket', abs(len(pt.choose_buckets([5000] * 40, 11)) - 1), 0)
check('zero compile cost: a bucket per distinct count', abs(len(pt.choose_buckets([256, 512, 768, 1024], 0.0)) - 4), 0)
check('large compile cost: one bucket at the largest count', abs(len(pt.choose_buckets([256, 512, 768, 1024], 1e9)) - 1), 0)
c = rng.integers(1000, 7000, 4000)
t = time.perf_counter()
pt.choose_buckets(c, 11.0, step=2)
dt = time.perf_counter() - t
print(f'  ({len(np.unique(2 * -(-c // 2)))} distinct gate lengths planned in {dt:.3f} s)')
check('time for thousands of distinct gate lengths (s)', dt, 1.0, '{:.3f}')

# ---------------------------------------------------------------------------------------------- CUDA sizing
print('\nCUDA memory model (memory.full_speed, cuda_streams, cuda_group)')
import warnings
import fastsar
from fastsar import memory as fmem, api, ffbp2, sim


def fake_plan(P, K, Ko, Po, C, Nx, Ny):
    return dict(levels=[dict(P=P, K=K, Ko=Ko, Po=Po, C=C)], Nx=Nx, Ny=Ny)


def cuda_run(plan, K, isz, free_total):
    """The sizing steps of ffbp_cuda's form for a device with free_total bytes free: -> (stream, children per group)."""
    lv0 = plan['levels'][0]
    planes = isz * lv0['P'] * K
    free = free_total - 8.0 * plan['Nx'] * plan['Ny']          # the image, allocated first
    stream = fmem.cuda_streams(planes, free)
    return stream, fmem.cuda_group(lv0, K, isz, free - (0 if stream else planes), stream)


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
check(f'full_speed is the threshold of the sizing ({ncase} plans and storage types)', nbad, 0)
# a 10.3 GB history with 4 GB of groups fits 16 GB as planes plus groups, but streams (10.3 GB > half of 16 GB)
plan = fake_plan(74203, 17282, 470, 9300, 64, 4000, 4000)
need, parts = fmem.full_speed(plan, 17282, 'cuda')
print(f'  (history {parts["history"] / 1e9:.1f} GB, groups {parts["first_level"] / 1e9:.1f} GB: full speed needs '
      f'{need / 1e9:.1f} GB; with 16 GB free the sizing streams: {cuda_run(plan, 17282, 8, 16e9)[0]})')
check('a history above half the free memory is not reported as full speed', float(need <= 16e9), 0)
check('float16 storage halves the history part', abs(fmem.full_speed(plan, 17282, 'cuda', 4)[1]['history'] * 2 - parts['history']), 0)

# ---------------------------------------------------------------------------------------------- ImageFormer.memory()
print('\nImageFormer.memory() on cuda and tpu (patched free memory)')
rng = np.random.default_rng(3)
col = sim.make_collect(res=0.5, scene=60.0, r0=5e3)
tg = np.stack([rng.uniform(-25, 25, 20), rng.uniform(-25, 25, 20), np.zeros(20)], 1)
S = sim.simulate_brute(col, tg, rng.standard_normal(20) + 1j * rng.standard_normal(20)).astype(np.complex64)
grid = (128, 128, 0.5, 0.5, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
fj = fastsar.ImageFormer(col.ant, col.fmin, col.df, col.K, *grid, backend='jax', window=False)
keep_free, keep_stats = fmem.cuda_free, fmem._jax_stats
try:
    fj.backend = 'cuda'                    # the bookkeeping of a cuda former, without a GPU
    need32 = fmem.full_speed(fj._plan, fj.K, 'cuda')[0]
    fmem.cuda_free = lambda: need32
    m_at = fj.memory()
    fmem.cuda_free = lambda: need32 * 0.99
    m_below = fj.memory()
    fj.precision = 'float16'
    m16 = fj.memory()
    fj.backend, fj.precision = 'tpu', 'float32'
    need_t = fmem.full_speed(fj._plan, fj.K, 'tpu')[0]
    fmem._jax_stats = lambda: dict(bytes_limit=need_t + 1e9, bytes_in_use=0.5e9)
    m_tpu = fj.memory()
    fmem._jax_stats = lambda: dict(bytes_limit=need_t + 1e9, bytes_in_use=2e9)
    m_tpu_busy = fj.memory()
finally:
    fmem.cuda_free, fmem._jax_stats = keep_free, keep_stats
    fj.backend, fj.precision = 'jax', 'float32'
check('cuda: full speed with exactly the memory it needs', float(not m_at['full_speed']), 0)
check('cuda: not full speed 1% below it', float(m_below['full_speed']), 0)
check('cuda float16: needs less than float32', float(m16['needed'] >= m_at['needed']), 0)
check('tpu: full speed when the free memory covers the need', float(not m_tpu['full_speed']), 0)
check('tpu: not full speed when other arrays take the margin', float(m_tpu_busy['full_speed']), 0)

# ---------------------------------------------------------------------------------------------- environment
print('\nenvironment variables')
nbad = 0
for name, value, least, kind, ok in (('FASTSAR_TPU_GROUP', '0', 1, int, False), ('FASTSAR_TPU_GROUP', '2', 1, int, True),
                                     ('FASTSAR_CUDA_GROUP', 'eight', 1, int, False),
                                     ('FASTSAR_MOSAIC_PREFETCH', '-1', 0, int, False),
                                     ('FASTSAR_MOSAIC_PREFETCH', '0', 0, int, True),
                                     ('FASTSAR_PULSE_SLACK', '-0.1', 0.0, float, False),
                                     ('FASTSAR_CPU_GROUP_GB', '1.5', 0.0, float, True)):
    os.environ[name] = value
    try:
        fmem._env_number(name, least, kind)
        res = 'accepted'
    except ValueError as e:
        res = f'ValueError: {e}'
    finally:
        os.environ.pop(name)
    nbad += (res == 'accepted') != ok
    print(f'    {name}={value}: {res}')
check('settings outside their range raise ValueError, others are accepted', nbad, 0)

print(f'\n{time.perf_counter() - t_start:.0f} s')
if bad:
    sys.exit('FAILED: ' + '; '.join(bad))
print('ok')
