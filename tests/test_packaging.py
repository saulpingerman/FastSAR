"""Every kernel source that the package reads at run time is listed as package data, so that a wheel carries it."""
import fnmatch
import os
import re

import fastsar


def test_kernel_sources_are_package_data():
    here = os.path.dirname(os.path.abspath(fastsar.__file__))
    with open(os.path.join(os.path.dirname(here), 'pyproject.toml')) as fh:        # (tomllib is Python 3.11's)
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
