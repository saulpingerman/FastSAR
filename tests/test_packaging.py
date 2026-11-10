"""Every kernel source that the package reads at run time is listed as package data, so that a wheel carries it;
a platform wheel also carries the prebuilt kernels and is tagged for its platform, not for a Python version."""
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

import fastsar
from fastsar import _build


def test_kernel_sources_are_package_data():
    here = os.path.dirname(os.path.abspath(fastsar.__file__))
    pyproject = os.path.join(os.path.dirname(here), 'pyproject.toml')
    globs = ['*.cpp', '*.hpp']                                       # an installed package: what its wheel was given
    if os.path.exists(pyproject):
        with open(pyproject) as fh:                                  # (tomllib is Python 3.11's)
            block = re.search(r'\[tool\.setuptools\.package-data\]\s*\nfastsar\s*=\s*\[([^\]]*)\]', fh.read())
        globs = re.findall(r'"([^"]+)"', block.group(1))
    assert globs
    sources = [f for f in os.listdir(here) if f.endswith(('.cpp', '.hpp', '.mm', '.metal', '.cu'))]
    assert 'ffbp_cpu.cpp' in sources and 'par.hpp' in sources
    missing = [f for f in sources if not any(fnmatch.fnmatch(f, g) for g in globs)]
    assert not missing, f'not in [tool.setuptools.package-data]: {missing}'
    # and every such file a module opens by name exists
    for mod in os.listdir(here):
        if mod.endswith('.py'):
            with open(os.path.join(here, mod)) as fh:
                for name in re.findall(r"'([a-z0-9_]+\.(?:cpp|hpp|mm|metal|cu))'", fh.read()):
                    assert name in sources, f'{mod} reads {name}, which is not in the package'
    for _, spec, _, _, _ in _build.KERNELS:                         # and the prebuilt kernels' table names real sources
        assert _build.kernel_source(spec).strip()


def _checkout():
    root = os.path.dirname(os.path.dirname(os.path.abspath(fastsar.__file__)))
    return root if os.path.exists(os.path.join(root, 'setup.py')) and os.path.exists(os.path.join(root, 'pyproject.toml')) else None


def _wheel(tmp_path, name, **env):
    """The wheel setup.py builds from a copy of the checkout's package, with env added to the environment."""
    root, src = _checkout(), tmp_path / name
    src.mkdir()
    for f in ('setup.py', 'pyproject.toml', 'README.md', 'LICENSE'):
        shutil.copy(os.path.join(root, f), src)
    shutil.copytree(os.path.join(root, 'fastsar'), src / 'fastsar', ignore=shutil.ignore_patterns('__pycache__', '_prebuilt'))
    e = {k: v for k, v in os.environ.items() if k != 'FASTSAR_PREBUILT'}
    r = subprocess.run([sys.executable, 'setup.py', '-q', 'bdist_wheel', '-d', str(tmp_path / (name + '-dist'))],
                       cwd=src, env=dict(e, **env), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    (whl,) = os.listdir(tmp_path / (name + '-dist'))
    return whl, zipfile.ZipFile(tmp_path / (name + '-dist') / whl)


def test_platform_wheel_carries_the_prebuilt_kernels(tmp_path):
    """FASTSAR_PREBUILT=1 (the wheels workflow): a py3-none wheel of this platform with every kernel at every level,
    in the package itself; without it the pure wheel, as before, with the sources only."""
    try:
        import setuptools
        from setuptools.command import bdist_wheel  # noqa: F401  (setuptools 70.1 and later)
    except ImportError:
        pytest.skip('building a wheel needs setuptools')
    if _checkout() is None:
        pytest.skip('not a checkout (the installed package has no setup.py)')
    whl, z = _wheel(tmp_path, 'pure')
    assert whl.endswith('-py3-none-any.whl') and 'fastsar/ffbp_cpu.cpp' in z.namelist()
    assert not [n for n in z.namelist() if '_prebuilt' in n]
    if _build._platform() not in _build.LEVELS or shutil.which(os.environ.get('CXX') or ('clang++' if sys.platform == 'darwin' else 'g++')) is None:
        pytest.skip('the prebuilt kernels need a compiler and a platform with levels')
    whl, z = _wheel(tmp_path, 'platform', FASTSAR_PREBUILT='1')
    assert '-py3-none-' in whl and not whl.endswith('-any.whl'), whl
    names = z.namelist()
    info = next(n for n in names if n.endswith('.dist-info/WHEEL'))
    assert 'Root-Is-Purelib: false' in z.read(info).decode()
    m = json.loads(z.read('fastsar/_prebuilt/manifest.json'))
    for name, spec, extra, _, _ in _build.KERNELS:
        assert 'fastsar/' + spec.split(':')[0] in names
        for level, *_ in _build.LEVELS[_build._platform()]:
            o = m['objects'][name][level]
            assert 'fastsar/_prebuilt/' + o['file'] in names
            assert o['hash'] == _build.source_hash(_build.kernel_source(spec), list(extra))
