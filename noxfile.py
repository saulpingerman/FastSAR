"""Test sessions (https://nox.thea.codes), run with uv: `uv tool install nox`, then

    nox                     # tests on Python 3.10 to 3.13 and on the lowest versions pyproject.toml allows (CPU)
    nox -s tests-3.12       # one Python
    nox -s lowest           # numpy, scipy and jax at their lower bounds, on the oldest Python
    nox -s cuda             # on a machine with an Nvidia GPU: the CUDA tests must run (--require cuda)
    nox -s tpu              # on a Cloud TPU VM: the TPU tests must run (--require tpu)

Arguments after `--` go to pytest, e.g. `nox -s tests-3.12 -- -k exact -s`. Every session installs the `test`
dependency group and passes `--require io`, so a missing test package fails instead of skipping.
"""
import os

import nox

nox.needs_version = '>=2025.2.9'
nox.options.default_venv_backend = 'uv'
nox.options.sessions = ['tests', 'lowest']

PYTHONS = ['3.10', '3.11', '3.12', '3.13']
PYPROJECT = nox.project.load_toml('pyproject.toml')
TEST = nox.project.dependency_groups(PYPROJECT, 'test')
PYTEST = ['pytest', '--timeout', '600']


def _pytest(session, require):
    session.run(*PYTEST, '--require', require, *session.posargs)


@nox.session(python=PYTHONS)
def tests(session):
    """The CPU suite: hardware tests are skipped."""
    session.install('-e', '.', *TEST)
    _pytest(session, 'io')


@nox.session(python=PYTHONS[0])
def lowest(session):
    """The library's dependencies at the lowest versions pyproject.toml allows (uv --resolution lowest-direct),
    then the test packages at versions compatible with them."""
    session.install('--resolution', 'lowest-direct', '-e', '.')
    pins = session.run('python', '-c', 'import importlib.metadata as m; '
                       'print("\\n".join(f"{p}=={m.version(p)}" for p in ("numpy", "scipy", "jax", "jaxlib")))',
                       silent=True)
    constraints = os.path.join(session.create_tmp(), 'lowest.txt')
    with open(constraints, 'w') as fh:
        fh.write(pins)
    session.log('lowest versions:\n' + pins.strip())
    session.install('-c', constraints, *TEST)
    _pytest(session, 'io')


@nox.session
def cuda(session):
    """On a machine with an Nvidia GPU and CUDA 12: CuPy installed and the CUDA tests required."""
    session.install('-e', '.[cuda]', *TEST)
    _pytest(session, 'cuda,io')


@nox.session
def tpu(session):
    """On a Cloud TPU VM: jax[tpu] installed and the TPU tests required."""
    session.install('-e', '.[tpu]', *TEST)
    _pytest(session, 'tpu,io')
