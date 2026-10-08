"""Checks of the planning that runs before or around image formation, on the CPU (JAX on its CPU device):

  patches.choose_buckets       against an exhaustive search over bucket sets; empty, single and equal inputs, zero
                               counts, the compile-cost trade-off, and its time for thousands of distinct lengths

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

print(f'\n{time.perf_counter() - t_start:.0f} s')
if bad:
    sys.exit('FAILED: ' + '; '.join(bad))
print('ok')
