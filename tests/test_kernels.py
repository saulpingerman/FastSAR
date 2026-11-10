"""Where the CPU kernels' shared objects come from (fastsar._build): the native build where there is a compiler, the
prebuilt objects of a platform wheel otherwise, chosen by the CPU's instruction-set level and tied to the source by
a hash. The selection is tested on made-up installations (a manifest and stand-in objects in a temporary
directory); the objects themselves are built once, where there is a compiler, and one image is formed with each
level this CPU runs."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from fastsar import _build

HASWELL = 'fpu sse sse2 pni ssse3 cx16 sse4_1 sse4_2 popcnt lahf_lm avx avx2 bmi1 bmi2 f16c fma abm movbe xsave'.split()
AVX512 = HASWELL + 'avx512f avx512bw avx512cd avx512dq avx512vl'.split()
X86 = ('linux', 'x86_64')


def test_levels_from_cpu_flags():
    names = lambda have, plat=X86: [n for n, _ in _build.cpu_levels(have, plat)]
    assert names(AVX512) == ['v4', 'v3', 'v2']
    assert names(HASWELL) == ['v3', 'v2']
    assert names([f for f in HASWELL if f != 'movbe']) == ['v2']          # one missing flag of a level denies it
    assert names([f for f in AVX512 if f != 'avx512vl']) == ['v3', 'v2']    # AVX-512 F alone is not x86-64-v4
    assert names([f for f in AVX512 if f != 'fma']) == ['v2']               # and no level above a missing one
    assert names(['fpu', 'sse', 'sse2']) == []
    assert names(['fp', 'asimd'], ('linux', 'aarch64')) == ['neon'] and names(['fp'], ('linux', 'aarch64')) == []
    assert names(AVX512, ('linux', 'riscv64')) == []
    here = [n for n, _ in _build.cpu_levels()]                              # this machine: consistent with its flags
    if _build._platform() == X86:
        assert ('v3' in here) == ({'avx2', 'fma', 'bmi2'} <= _build._cpu_flags() and 'v2' in here)


def test_source_hash_ignores_tuning_flags_only():
    h = _build.source_hash
    assert h('int f();', ['-O3', '-march=native', '-mprefer-vector-width=512', '-funroll-loops', '-std=c++17']) == \
        h('int f();', ['-O3', '-march=x86-64-v2', '-std=c++17'])
    assert h('int f();', ['-std=c++17']) != h('int f();', ['-std=c++17', '-ffast-math'])
    assert h('int f();', ['-std=c++17']) != h('int g();', ['-std=c++17'])
    assert h('int f();', []) != h('int f();', ['-DX=1'])


def _call_sites(monkeypatch):
    """(name, source, flags) of every shared_object call of the package, each module made to ask again."""
    from fastsar import bp, exact, ffbp_cpu, io, pfa2, resample, sentinel1
    seen = []

    def record(name, src, flags):
        seen.append((name, src, list(flags)))
        raise RuntimeError('recorded')
    monkeypatch.setattr(_build, 'shared_object', record)
    monkeypatch.delenv('FFBP_CPU_FLAGS', raising=False)
    for mod, var, empty, load in ((ffbp_cpu, '_lib', None, ffbp_cpu.lib), (exact, '_cpu_lib', None, exact._cpu),
                                  (bp, '_cpu_lib', None, bp._cpu), (pfa2, '_CPU_LIB', None, pfa2._cpu_gather_lib),
                                  (resample, '_LIB', type(resample._LIB)(), resample._lib),
                                  (io, '_LIB', type(io._LIB)(), io._lib), (sentinel1, '_lib', None, sentinel1._decoder)):
        monkeypatch.setattr(mod, var, empty)
        try:
            load()
        except RuntimeError:
            pass
    return seen


def test_kernel_table_matches_the_call_sites(monkeypatch):
    """KERNELS, from which the prebuilt objects are built, lists every kernel a module asks for, with the source and
    the flags that module passes: otherwise a wheel's objects would not be found by their hash."""
    seen = _call_sites(monkeypatch)
    table = {name: (spec, extra, wide) for name, spec, extra, wide, _ in _build.KERNELS}
    assert sorted(n for n, _, _ in seen) == sorted(table)
    for name, src, flags in seen:
        spec, extra, wide = table[name]
        assert _build.kernel_source(spec) == src, name
        assert _build.source_hash(src, flags) == _build.source_hash(src, list(extra)), name
        if _build._platform() == X86:
            assert ('-mprefer-vector-width=512' in flags) == wide, name


@pytest.fixture
def install(tmp_path, monkeypatch):
    """A made-up installation: prebuilt objects of the kernel 'k' (source 'int k;', flags -std=c++17) at the levels
    v2 and v3 of an AVX-512 x86-64 Linux machine with no compiler and an empty cache. The objects are stand-in
    files, which is all the selection needs; one "loads" unless its text says otherwise."""
    d = tmp_path / 'prebuilt'
    d.mkdir()
    objects = {}
    for level in ('v2', 'v3'):
        (d / f'k.{level}.so').write_text('object')
        objects[level] = dict(file=f'k.{level}.so', hash=_build.source_hash('int k;', ['-std=c++17']))
    (d / 'manifest.json').write_text(json.dumps(dict(platform='linux-x86_64', compiler='g++', threads='std::thread',
                                                     objects=dict(k=objects))))
    monkeypatch.setattr(_build, 'PREBUILT_DIR', str(d))
    monkeypatch.setattr(_build, '_load_error', lambda path: None if open(path).read() == 'object' else 'invalid ELF header')
    monkeypatch.setattr(_build, '_MANIFEST', [])
    monkeypatch.setattr(_build, '_platform', lambda: X86)
    monkeypatch.setattr(_build, '_supported', lambda: True)
    monkeypatch.setattr(_build, '_cpu_flags', lambda: set(AVX512))
    monkeypatch.setattr(_build, 'USED', {})
    monkeypatch.setattr(_build, 'compiler', lambda: ('no-such-compiler', ['-fopenmp'], ['-fopenmp']))
    monkeypatch.setenv('FASTSAR_CACHE_DIR', str(tmp_path / 'cache'))
    for v in ('FASTSAR_KERNELS', 'FASTSAR_KERNEL_LEVEL', 'CXX'):
        monkeypatch.delenv(v, raising=False)
    return d


FLAGS = ['-O3', '-march=native', '-std=c++17']


def test_no_compiler_takes_the_best_prebuilt_level(install):
    so = _build.shared_object('k', 'int k;', FLAGS)
    assert so == str(install / 'k.v3.so')                                   # v4 is not carried: the best below it
    assert _build.USED['k']['path'] == 'prebuilt' and _build.USED['k']['level'] == 'x86-64-v3'
    assert 'was not found' in _build.USED['k']['note']
    st = _build.status()
    assert st['levels'][0] == 'x86-64-v4' and st['prebuilt']['threads'] == 'std::thread'


def test_level_never_above_the_cpu(install, monkeypatch):
    monkeypatch.setattr(_build, '_cpu_flags', lambda: {'pni', 'ssse3', 'cx16', 'sse4_1', 'sse4_2', 'popcnt', 'lahf_lm'})
    assert _build.shared_object('k', 'int k;', FLAGS) == str(install / 'k.v2.so')
    monkeypatch.setenv('FASTSAR_KERNEL_LEVEL', 'v3')
    with pytest.raises(RuntimeError, match='this CPU runs the levels v2'):
        _build.shared_object('k', 'int k;', FLAGS)
    monkeypatch.delenv('FASTSAR_KERNEL_LEVEL')
    monkeypatch.setattr(_build, '_cpu_flags', lambda: {'sse2'})
    with pytest.raises(RuntimeError, match='no prebuilt k kernel for this CPU'):
        _build.shared_object('k', 'int k;', FLAGS)


def test_level_override_and_mode(install, monkeypatch):
    monkeypatch.setenv('FASTSAR_KERNEL_LEVEL', 'v2')
    assert _build.kernels_mode() == 'prebuilt'
    assert _build.shared_object('k', 'int k;', FLAGS) == str(install / 'k.v2.so')
    monkeypatch.setenv('FASTSAR_KERNEL_LEVEL', 'v4')                        # the CPU runs it, the installation lacks it
    with pytest.raises(RuntimeError, match='no prebuilt k kernel'):
        _build.shared_object('k', 'int k;', FLAGS)
    monkeypatch.delenv('FASTSAR_KERNEL_LEVEL')
    monkeypatch.setenv('FASTSAR_KERNELS', 'fastest')
    with pytest.raises(ValueError, match='native or prebuilt'):
        _build.shared_object('k', 'int k;', FLAGS)
    monkeypatch.setenv('FASTSAR_KERNELS', 'native')                         # no fallback when native is asked for
    with pytest.raises(RuntimeError, match='was not found'):
        _build.shared_object('k', 'int k;', FLAGS)


def test_another_source_is_never_served(install, monkeypatch):
    for src, flags in (('int k2;', FLAGS), ('int k;', FLAGS + ['-ffast-math'])):
        with pytest.raises(RuntimeError, match='another source or with other flags'):
            _build.shared_object('k', src, flags)
    monkeypatch.setenv('FASTSAR_KERNELS', 'prebuilt')
    with pytest.raises(RuntimeError, match='FASTSAR_KERNELS=prebuilt'):
        _build.shared_object('k', 'int k2;', FLAGS)
    with pytest.raises(RuntimeError, match='no prebuilt other kernel'):
        _build.shared_object('other', 'int k;', FLAGS)


def test_an_object_that_does_not_load_is_skipped(install):
    (install / 'k.v3.so').write_text('not a shared object')
    assert _build.shared_object('k', 'int k;', FLAGS) == str(install / 'k.v2.so')


def test_no_compiler_and_no_prebuilt_objects(install, monkeypatch):
    os.remove(install / 'manifest.json')
    with pytest.raises(RuntimeError) as e:
        _build.shared_object('k', 'int k;', FLAGS)
    assert 'was not found' in str(e.value) and 'carries no prebuilt kernels for this platform' in str(e.value)


def test_compiler_present(install, monkeypatch, tmp_path):
    """A compiler on the path: the native build, as without prebuilt objects; when it fails, the prebuilt object, and
    the failure is noted so that later processes do not compile again."""
    cxx = tmp_path / 'cxx'
    cxx.write_text('#!/bin/sh\nfor a; do o="$a"; done\necho built > "$o"\n')
    cxx.chmod(0o755)
    monkeypatch.setattr(_build, 'compiler', lambda: (str(cxx), ['-fopenmp'], ['-fopenmp']))
    so = _build.shared_object('k', 'int k;', FLAGS)
    assert so.startswith(str(tmp_path / 'cache')) and _build.USED['k']['path'] == 'native'
    assert _build.shared_object('k', 'int k;', FLAGS) == so
    monkeypatch.setattr(_build, 'compiler', lambda: ('no-such-compiler', ['-fopenmp'], ['-fopenmp']))
    assert _build.shared_object('k', 'int k;', FLAGS) != so                 # the cache is per compiler
    cxx.write_text('#!/bin/sh\necho "no OpenMP here" >&2\nexit 1\n')
    monkeypatch.setattr(_build, 'compiler', lambda: (str(cxx), ['-fopenmp'], ['-fopenmp']))
    flags = FLAGS + ['-O2']                                                 # (another cache key, the same source hash)
    assert _build.shared_object('k', 'int k;', flags) == str(install / 'k.v3.so')
    assert 'no OpenMP here' not in _build.USED['k']['note'] and 'compiling the k kernel failed' in _build.USED['k']['note']
    marks = [f for f in os.listdir(tmp_path / 'cache') if f.endswith('.failed')]
    assert len(marks) == 1
    cxx.write_text('#!/bin/sh\nfor a; do o="$a"; done\necho built > "$o"\n')    # repaired, but not tried again
    assert _build.shared_object('k', 'int k;', flags) == str(install / 'k.v3.so')
    assert 'earlier native build failed' in _build.USED['k']['note']
    os.remove(tmp_path / 'cache' / marks[0])
    assert _build.shared_object('k', 'int k;', flags).startswith(str(tmp_path / 'cache'))
    # without a prebuilt object the failure is the error, as before
    cxx.write_text('#!/bin/sh\necho "no OpenMP here" >&2\nexit 1\n')
    with pytest.raises(RuntimeError, match='no OpenMP here'):
        _build.shared_object('k', 'int k2;', FLAGS)
    assert not [f for f in os.listdir(tmp_path / 'cache') if f.endswith('.failed')]


def test_installed_prebuilt_objects_match_the_sources():
    """In an installation from a platform wheel: every kernel has an object at every level of the platform, with
    the hash of the source the modules ask for."""
    m = _build.prebuilt_manifest()
    if m is None:
        pytest.skip('this installation carries no prebuilt kernels (a checkout, or the source distribution)')
    levels = [lv[0] for lv in _build.LEVELS[_build._platform()]]
    for name, spec, extra, _, _ in _build.KERNELS:
        for level in levels:
            o = m['objects'][name][level]
            assert os.path.isfile(os.path.join(_build.PREBUILT_DIR, o['file'])), o['file']
            assert o['hash'] == _build.source_hash(_build.kernel_source(spec), list(extra)), (name, level)


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    """The prebuilt objects of this platform, built as the wheel's build step builds them."""
    if _build._platform() not in _build.LEVELS:
        pytest.skip('no prebuilt levels for this platform')
    cxx = os.environ.get('CXX') or ('clang++' if sys.platform == 'darwin' else 'g++')
    if shutil.which(cxx) is None:
        pytest.skip(f'building the prebuilt kernels needs a compiler ({cxx})')
    d = tmp_path_factory.mktemp('prebuilt')
    return str(d), _build.build_prebuilt(str(d), log=lambda line: None)


def test_build_prebuilt(built):
    d, m = built
    with open(os.path.join(d, 'manifest.json')) as fh:
        assert json.load(fh) == m
    assert m['platform'] == '-'.join(_build._platform()) and m['threads'] == 'std::thread'
    levels = [lv[0] for lv in _build.LEVELS[_build._platform()]]
    files = set(os.listdir(d)) - {'manifest.json'}
    assert files == {m['objects'][name][level]['file'] for name, *_ in _build.KERNELS for level in levels}
    if sys.platform.startswith('linux') and shutil.which('objdump'):
        for f in sorted(files):                                             # only the C library: no OpenMP, no libstdc++
            out = subprocess.run(['objdump', '-p', os.path.join(d, f)], capture_output=True, text=True).stdout
            needed = {line.split()[1].split('.so')[0] for line in out.splitlines() if 'NEEDED' in line}
            assert needed <= {'libc', 'libm', 'ld-linux-x86-64', 'ld-linux-aarch64', 'libpthread', 'libdl'}, (f, needed)


_IMAGE = '''
import sys, numpy as np, fastsar
from fastsar import _build, sim
if sys.argv[1]:
    _build.PREBUILT_DIR = sys.argv[1]
col = sim.make_collect(res=0.5, scene=30.0, r0=5e3)
S = sim.simulate_brute(col, np.array([[3.0, -2.0, 0.0], [-5.0, 4.0, 0.0]]), np.ones(2)).astype(np.complex64)
out = []
for alg in ('ffbp', 'pfa', 'bp'):
    kw = dict(T=16, levels=2) if alg == 'ffbp' else dict(pfa_guard=2.0) if alg == 'pfa' else {}
    out.append(fastsar.form_image(S, col.ant, col.fmin, col.df, 64, 64, 0.5, 0.5, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
                                  backend='cpu', algorithm=alg, **kw))
np.save(sys.argv[2], np.stack(out))
print(' '.join(sorted({f"{v['path']}:{v['level']}" for v in _build.USED.values()})))
'''


def _images(tmp_path, tag, prebuilt_dir, **env):
    out = str(tmp_path / f'{tag}.npy')
    e = {k: v for k, v in os.environ.items() if k not in ('FASTSAR_KERNELS', 'FASTSAR_KERNEL_LEVEL')}
    root = os.path.dirname(os.path.dirname(os.path.abspath(_build.__file__)))
    e.update(env, PYTHONPATH=os.pathsep.join([root] + ([e['PYTHONPATH']] if e.get('PYTHONPATH') else [])),
             JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([sys.executable, '-c', _IMAGE, prebuilt_dir, out], capture_output=True, text=True, env=e, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    return np.load(out), r.stdout.strip().splitlines()[-1]


def test_images_of_each_prebuilt_level_match_the_native_build(built, tmp_path):
    """The three image formation algorithms with each level's objects that this CPU runs, each in its own process,
    against the native build: the same images to -80 dB of the peak (the levels differ in vector width and in
    fused multiply-adds, so in float32 rounding, not in the arithmetic)."""
    d, _ = built
    ref, used = _images(tmp_path, 'native', '', FASTSAR_KERNELS='native')
    assert used == 'native:None'
    levels = _build.cpu_levels()
    assert levels, 'this CPU runs no level of its platform'
    for name, label in levels:
        img, used = _images(tmp_path, name, d, FASTSAR_KERNEL_LEVEL=name)
        assert used == f'prebuilt:{label}'
        err = np.abs(img - ref).max(axis=(1, 2)) / np.abs(ref).max(axis=(1, 2))
        print(f'{label}: relative difference from the native build {err.max():.1e}')
        assert err.max() < 1e-4, (label, err)
