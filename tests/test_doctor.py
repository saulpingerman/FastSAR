"""The installation check (fastsar.doctor): on this machine, and on machines made up by replacing what it looks at
(no compiler, a GPU without CuPy, a TPU VM without the TPU build of JAX), the commands it recommends."""
import json

import pytest

from fastsar import doctor


def test_this_machine(capsys):
    c = doctor.check(probe=False)
    assert set(c) >= {'fastsar', 'python', 'cpu', 'compiler', 'nvidia', 'tpu', 'packages', 'backends', 'todo'}
    assert c['packages']['numpy'] and c['packages']['scipy'] and c['cpu']['cores'] >= 1
    assert all(set(t) == {'what', 'command', 'pip'} for t in c['todo'])
    text = doctor.report(c)
    assert 'Backends:' in text and all(name in text for name in c['backends'])
    assert doctor.main(['--json', '--no-probe']) in (0, 1)
    assert json.loads(capsys.readouterr().out)['fastsar'] == c['fastsar']


def machine(monkeypatch, *, compiler=True, openmp=True, nvidia=None, tpu=False, missing=()):
    monkeypatch.setattr(doctor, '_compiler', lambda: dict(
        command='g++', found=compiler, version='g++ 13' if compiler else None, openmp=openmp if compiler else None,
        install=None if compiler and openmp else 'sudo apt-get install -y g++'))
    monkeypatch.setattr(doctor, '_nvidia', lambda: nvidia)
    monkeypatch.setattr(doctor, '_tpu', lambda: tpu)
    real = doctor._has
    monkeypatch.setattr(doctor, '_has', lambda m: False if m in missing else (True if m in ('jax', 'numpy', 'scipy') else real(m)))
    return doctor.check(probe=False)


def test_no_compiler(monkeypatch):
    c = machine(monkeypatch, compiler=False)
    assert not c['backends']['cpu']['ok'] and c['backends']['jax']['ok']
    t = [t for t in c['todo'] if 'compiler' in t['what']]
    assert len(t) == 1 and not t[0]['pip'] and 'g++' in t[0]['command']
    assert 'sudo apt-get install -y g++' in doctor.report(c)
    c = machine(monkeypatch, openmp=False)
    assert not c['backends']['cpu']['ok'] and any('OpenMP' in t['what'] for t in c['todo'])


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
    files['/sys/fs/cgroup/user.slice/job.scope/memory.max'] = 'max\n'
    assert memory._cgroup_available() is None and memory.host_available() == 104857600 * 1024
