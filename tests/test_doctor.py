"""The installation check (fastsar.doctor): on this machine, and on machines made up by replacing what it looks at
(no compiler, a GPU without CuPy, a TPU VM without the TPU build of JAX), the commands it recommends."""
import json
import sys

import pytest

from fastsar import doctor


def test_this_machine(capsys):
    c = doctor.check(probe=False)
    assert set(c) >= {'fastsar', 'python', 'cpu', 'compiler', 'kernels', 'nvidia', 'tpu', 'packages', 'backends', 'todo', 'hints'}
    assert c['packages']['numpy'] and c['packages']['scipy'] and c['cpu']['cores'] >= 1
    assert all(set(t) == {'what', 'command', 'pip'} for t in c['todo'])
    text = doctor.report(c)
    assert 'Backends:' in text and all(name in text for name in c['backends'])
    assert doctor.main(['--json', '--no-probe']) in (0, 1)
    assert json.loads(capsys.readouterr().out)['fastsar'] == c['fastsar']


def machine(monkeypatch, *, compiler=True, openmp=True, nvidia=None, tpu=False, missing=(), prebuilt=None, mode='',
            simd=('AVX2',)):
    """prebuilt: the level label of the prebuilt kernels the made-up installation carries (None: none)."""
    from fastsar import _build
    monkeypatch.setattr(doctor, '_compiler', lambda: dict(
        command='g++', found=compiler, version='g++ 13' if compiler else None, openmp=openmp if compiler else None,
        install=None if compiler and openmp else 'sudo apt-get install -y g++'))
    monkeypatch.setattr(doctor, '_nvidia', lambda: nvidia)
    monkeypatch.setattr(doctor, '_tpu', lambda: tpu)
    cpu = dict(doctor._cpu(), machine='x86_64', system='linux', simd=list(simd))
    monkeypatch.setattr(doctor, '_cpu', lambda: cpu)
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(_build, 'status', lambda: dict(
        mode=mode, levels=[prebuilt] if prebuilt else [], prebuilt=dict(platform='linux-x86_64') if prebuilt else None,
        kernels={name: dict(what=what, prebuilt=prebuilt, used=None,
                            why=None if prebuilt else 'this installation carries no prebuilt kernels for this platform')
                 for name, _, _, _, what in _build.KERNELS}))
    real = doctor._has
    monkeypatch.setattr(doctor, '_has', lambda m: False if m in missing else (True if m in ('jax', 'numpy', 'scipy') else real(m)))
    return doctor.check(probe=False)


def test_no_compiler(monkeypatch):
    c = machine(monkeypatch, compiler=False)
    assert not c['backends']['cpu']['ok'] and c['backends']['jax']['ok']
    t = [t for t in c['todo'] if 'compiler' in t['what']]
    assert len(t) == 1 and not t[0]['pip'] and 'g++' in t[0]['command']
    assert 'sudo apt-get install -y g++' in doctor.report(c)
    assert 'no prebuilt kernels' in c['backends']['cpu']['why'] and 'CPU kernels: none' in doctor.report(c)
    c = machine(monkeypatch, openmp=False)
    assert not c['backends']['cpu']['ok'] and any('OpenMP' in t['what'] for t in c['todo'])


def test_kernel_paths(monkeypatch):
    """Which shared objects the CPU backend uses: the native build with a compiler, the prebuilt objects of a
    platform wheel without one (the backend is then ready and nothing is missing), and a hint where a compiler
    would use vector extensions the prebuilt level leaves out."""
    c = machine(monkeypatch, prebuilt='x86-64-v3')
    assert c['backends']['cpu']['ok'] and 'CPU kernels: native build' in doctor.report(c) and not c['hints']
    assert all(f['path'] == 'native' for f in c['kernels']['families'].values())
    c = machine(monkeypatch, compiler=False, prebuilt='x86-64-v3')
    assert c['backends']['cpu']['ok'] and 'prebuilt kernels (x86-64-v3); no C++ compiler' in c['backends']['cpu']['why']
    assert not [t for t in c['todo'] if 'compiler' in t['what']] and not c['hints']
    assert 'CPU kernels: prebuilt (x86-64-v3)' in doctor.report(c)
    c = machine(monkeypatch, openmp=False, prebuilt='x86-64-v3')              # a compiler that cannot build them
    assert c['backends']['cpu']['ok'] and 'cannot link OpenMP' in c['backends']['cpu']['why']
    c = machine(monkeypatch, compiler=False, prebuilt='x86-64-v3', simd=('AVX-512', 'AVX2'))
    assert c['backends']['cpu']['ok'] and len(c['hints']) == 1 and 'AVX-512' in c['hints'][0]['what']
    assert 'Optional:' in doctor.report(c) and 'sudo apt-get install -y g++' in doctor.report(c)
    c = machine(monkeypatch, compiler=False, prebuilt='x86-64-v4', simd=('AVX-512', 'AVX2'))
    assert c['backends']['cpu']['ok'] and not c['hints']
    c = machine(monkeypatch, prebuilt='x86-64-v3', mode='prebuilt')            # FASTSAR_KERNELS=prebuilt with a compiler
    assert c['backends']['cpu']['ok'] and 'FASTSAR_KERNELS=prebuilt' in doctor.report(c)
    assert all(f['text'] == 'prebuilt (x86-64-v3)' for f in c['kernels']['families'].values())
    c = machine(monkeypatch, compiler=False, prebuilt='x86-64-v3', mode='native')
    assert not c['backends']['cpu']['ok'] and 'FASTSAR_KERNELS=native' in c['backends']['cpu']['why']


@pytest.mark.parametrize('cuda, wheel', [((12, 4), 'cupy-cuda12x[ctk]'), ((13, 2), 'cupy-cuda13x[ctk]'),
                                         ((11, 8), 'cupy-cuda11x'), (None, 'cupy-cuda12x[ctk]')])
def test_gpu_without_cupy(monkeypatch, cuda, wheel):
    c = machine(monkeypatch, nvidia=dict(gpus=['NVIDIA L4, 23034 MiB'], cuda=cuda), missing=('cupy',))
    assert not c['backends']['cuda']['ok'] and 'CuPy is not installed' in c['backends']['cuda']['why']
    t = [t for t in c['todo'] if 'CuPy' in t['what']]
    assert len(t) == 1 and t[0]['pip'] and wheel in t[0]['command'] and ' -m pip install ' in t[0]['command']


def test_no_gpu_and_optional_packages(monkeypatch):
    c = machine(monkeypatch, missing=('rasterio', 'h5py'))
    assert not c['backends']['cuda']['ok'] and not any('CuPy' in t['what'] for t in c['todo'])
    cmds = ' '.join(t['command'] for t in c['todo'])
    assert 'fastsar[geo]' in cmds and 'fastsar[raw]' in cmds and 'fastsar[io]' not in cmds
    assert c['backends']['cpu']['ok']


def test_tpu_vm_without_the_tpu_build(monkeypatch):
    c = machine(monkeypatch, tpu=True, missing=('libtpu',))
    assert not c['backends']['tpu']['ok']
    t = [t for t in c['todo'] if 'TPU' in t['what']]
    assert len(t) == 1 and 'jax[tpu]' in t[0]['command'] and 'libtpu_releases' in t[0]['command']
    assert not machine(monkeypatch)['backends']['tpu']['ok']


def test_available_memory_honors_a_control_group_limit(tmp_path, monkeypatch):
    """host_available under a memory limit (a container, a systemd scope): the limit less the usage, not the host's."""
    import builtins
    from fastsar import memory
    files = {'/proc/self/cgroup': '0::/user.slice/job.scope\n',
             '/sys/fs/cgroup/user.slice/job.scope/memory.max': '8589934592\n',
             '/sys/fs/cgroup/user.slice/job.scope/memory.current': '1073741824\n',
             '/sys/fs/cgroup/user.slice/memory.max': 'max\n',
             '/proc/meminfo': 'MemTotal: 131072000 kB\nMemAvailable: 104857600 kB\n'}
    real = builtins.open

    def fake(path, *a, **k):
        if path in files:
            p = tmp_path / str(abs(hash(path)))
            p.write_text(files[path])
            return real(p, *a, **k)
        if str(path).startswith(('/sys/fs/cgroup', '/proc/')):
            raise FileNotFoundError(path)
        return real(path, *a, **k)
    monkeypatch.setattr(builtins, 'open', fake)
    assert memory._cgroup_available() == 7 * 2 ** 30
    assert memory.host_available() == 7 * 2 ** 30
    # the group's page cache is not counted against it: a 4 GiB file just read is in the usage and can be given back
    files['/sys/fs/cgroup/user.slice/job.scope/memory.current'] = f'{5 * 2 ** 30}\n'
    files['/sys/fs/cgroup/user.slice/job.scope/memory.stat'] = (f'anon {2 ** 30}\nfile {4 * 2 ** 30}\ninactive_anon 0\n'
                                                                 f'inactive_file {3 * 2 ** 30}\nactive_file {2 ** 30}\n')
    assert memory._cgroup_available() == 7 * 2 ** 30 and memory.host_available() == 7 * 2 ** 30
    files['/sys/fs/cgroup/user.slice/job.scope/memory.stat'] = 'inactive_file many\n'       # unreadable: the usage as it is
    assert memory._cgroup_available() == 3 * 2 ** 30
    files['/sys/fs/cgroup/user.slice/job.scope/memory.stat'] = (f'inactive_file 1\nactive_file 1\n'
                                                                 f'total_inactive_file {2 ** 30}\ntotal_active_file {2 ** 30}\n')
    assert memory._cgroup_available() == 5 * 2 ** 30                                         # a v1 hierarchy: the totals
    del files['/sys/fs/cgroup/user.slice/job.scope/memory.stat']
    files['/sys/fs/cgroup/user.slice/job.scope/memory.current'] = '1073741824\n'
    files['/sys/fs/cgroup/user.slice/job.scope/memory.max'] = 'max\n'
    assert memory._cgroup_available() is None and memory.host_available() == 104857600 * 1024
