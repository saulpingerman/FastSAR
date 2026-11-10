"""Platform wheels with prebuilt CPU kernels. pyproject.toml holds the metadata; this file adds one build step.

With FASTSAR_PREBUILT=1 in the environment (the cibuildwheel configuration of pyproject.toml sets it) the C++
kernels are compiled at each instruction-set level of the platform into fastsar/_prebuilt of the wheel
(fastsar/_build.py: build_prebuilt), and the wheel is tagged py3-none-<platform>: the objects are loaded with
ctypes, so they do not depend on the Python version. Without it the wheel is pure, as an installation from the
source distribution is: the kernels are then compiled on first use and need a C++ compiler on the machine."""
import importlib.util
import os

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py

PREBUILT = os.environ.get('FASTSAR_PREBUILT', '') == '1'


def _build_module():
    """fastsar/_build.py loaded by path: importing the package would need NumPy in the build environment."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fastsar', '_build.py')
    spec = importlib.util.spec_from_file_location('_fastsar_build', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BuildPy(build_py):
    def run(self):
        super().run()
        if PREBUILT and not getattr(self, 'editable_mode', False):
            _build_module().build_prebuilt(os.path.join(self.build_lib, 'fastsar', '_prebuilt'))


class BdistWheel(bdist_wheel):
    def get_tag(self):
        if not PREBUILT:
            return super().get_tag()
        return 'py3', 'none', super().get_tag()[2]


class PlatformDistribution(Distribution):
    """With the prebuilt kernels the package is installed as platform-specific (platlib), not pure."""

    def has_ext_modules(self):
        return PREBUILT or super().has_ext_modules()


setup(cmdclass=dict(build_py=BuildPy, bdist_wheel=BdistWheel), distclass=PlatformDistribution)
