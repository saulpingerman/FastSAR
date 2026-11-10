"""What this machine can run, and what to install so that it runs more:

    python -m fastsar.doctor              # the report, and the commands that would complete the installation
    python -m fastsar.doctor --install    # the same, running the pip commands among them
    python -m fastsar.doctor --json

pip installs FastSAR with NumPy, SciPy and JAX. The rest depends on the machine and cannot be chosen by pip: the
C++ compiler that builds the CPU kernels for the exact processor on first use (a system package; without one the
prebuilt kernels of a platform wheel run), CuPy for the CUDA version of an Nvidia GPU's driver, the TPU build of
JAX, and the readers' packages. check() looks at the processor, the compiler, the kernels, the GPU, the TPU and the
installed packages, and lists what is missing with the command that installs it. Commands that need the system's
package manager are printed, never run.
"""
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys

__all__ = ['check', 'report', 'main']

_SYSTEM = (('apt-get', 'sudo apt-get install -y g++'), ('dnf', 'sudo dnf install -y gcc-c++'),
           ('yum', 'sudo yum install -y gcc-c++'), ('zypper', 'sudo zypper install -y gcc-c++'),
           ('pacman', 'sudo pacman -S --needed gcc'), ('apk', 'apk add g++'))
_OPTIONAL = (('rasterio', 'geo', 'GeoTIFF output'), ('h5py', 'raw', 'NISAR and other HDF5 raw data'))
_TPU_INDEX = 'https://storage.googleapis.com/jax-releases/libtpu_releases.html'


def _run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _has(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _cpu():
    flags = set()
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith(('flags', 'Features')):
                    flags = set(line.split(':', 1)[1].split())
                    break
    except OSError:
        pass
    simd = [n for n, f in (('AVX-512', 'avx512f'), ('AVX2', 'avx2'), ('SVE', 'sve'), ('NEON', 'asimd')) if f in flags]
    if sys.platform == 'darwin' and platform.machine() == 'arm64':
        simd = ['NEON']
    return dict(machine=platform.machine(), system=sys.platform, cores=os.cpu_count(), simd=simd)


def _compiler():
    """The C++ compiler the CPU kernels would be built with: dict(command, found, version, openmp, install)."""
    from . import _build
    try:
        cxx, omp_c, omp_l = _build.compiler()
    except RuntimeError:
        cxx, omp_c, omp_l = 'clang++', [], []
    path = shutil.which(cxx) if _build.compiler_found(cxx) else None
    out = dict(command=cxx, found=path is not None, version=None, openmp=None, install=None)
    if path is None:
        if sys.platform == 'darwin':
            out['install'] = 'xcode-select --install'
        else:
            out['install'] = next((c for tool, c in _SYSTEM if shutil.which(tool)), 'install g++ with the system package manager')
        return out
    v = _run([cxx, '--version'])
    out['version'] = v.splitlines()[0].strip() if v else None
    if '-fopenmp' in omp_c:                                          # a compiler without its OpenMP runtime fails at the link
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, 't.cc')
            with open(src, 'w') as fh:
                fh.write('#include <omp.h>\nint main() { return omp_get_max_threads() > 0 ? 0 : 1; }\n')
            out['openmp'] = _run([cxx, '-fopenmp', src, '-o', os.path.join(d, 't')], timeout=60) is not None
            if not out['openmp']:
                out['install'] = next((c.replace('g++', 'g++ libgomp1') if 'apt' in tool else c.replace('gcc-c++', 'gcc-c++ libgomp')
                                       for tool, c in _SYSTEM if shutil.which(tool)), 'install the OpenMP runtime (libgomp)')
    return out


def _kernels(cc):
    """Where each CPU kernel's shared object comes from on this machine, given the compiler found (cc):
    dict(mode (FASTSAR_KERNELS, '' when unset), levels (the CPU's instruction-set levels, best first), prebuilt (what
    the installation carries, or None), families: name -> dict(what, path 'native', 'prebuilt' or None, level,
    text: 'native build', 'prebuilt (x86-64-v3)' or the reason there is no object))."""
    from . import _build
    try:
        st = _build.status()
    except ValueError as e:                                         # FASTSAR_KERNELS is neither native nor prebuilt
        return dict(mode=os.environ.get('FASTSAR_KERNELS'), levels=[], prebuilt=None, families={}, error=str(e))
    native = cc['found'] and cc['openmp'] is not False
    fam = {}
    for name, k in st['kernels'].items():
        if k['used']:
            path, level = k['used']['path'], k['used']['level']
        elif st['mode'] != 'prebuilt' and native:
            path, level = 'native', None
        elif st['mode'] != 'native' and k['prebuilt']:
            path, level = 'prebuilt', k['prebuilt']
        else:
            path, level = None, None
        text = 'native build' if path == 'native' else f'prebuilt ({level})' if path else \
            ('none: ' + ('no C++ compiler' if st['mode'] == 'native' or not k['why'] else k['why']))
        fam[name] = dict(what=k['what'], path=path, level=level, text=text)
    return dict(mode=st['mode'], levels=st['levels'], prebuilt=st['prebuilt'], families=fam)


def _beyond(cpu, levels):
    """The vector extension of this CPU that the prebuilt levels in use leave unused, or None."""
    for ext, covered in (('AVX-512', ('x86-64-v4',)), ('AVX2', ('x86-64-v4', 'x86-64-v3')), ('SVE', ())):
        if ext in cpu['simd']:
            return None if all(lv in covered for lv in levels) else ext
    return None


def _nvidia():
    """The Nvidia GPUs and the CUDA version their driver supports, from nvidia-smi: dict(gpus, cuda) or None."""
    if shutil.which('nvidia-smi') is None:
        return None
    q = _run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])
    if not q or not q.strip():
        return None
    head = _run(['nvidia-smi']) or ''
    m = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', head)
    return dict(gpus=[line.strip() for line in q.strip().splitlines()], cuda=(int(m.group(1)), int(m.group(2))) if m else None)


def _tpu():
    """Whether this looks like a Cloud TPU VM (the accelerator's device nodes, or the TPU environment variables)."""
    try:
        if any(n.startswith('accel') for n in os.listdir('/dev')):
            return True
    except OSError:
        pass
    return any(k in os.environ for k in ('TPU_ACCELERATOR_TYPE', 'TPU_WORKER_ID'))


def check(probe=True):
    """Look at the machine. probe: also ask CuPy and JAX for their devices (imports them, a few seconds).

    -> dict(fastsar (version), python, cpu, compiler, kernels (see _kernels), nvidia, tpu, packages (name ->
    installed), backends (name -> dict(ok, why)), todo (a list of dict(what, command, pip: whether --install may run
    it)), hints (the same, for what is optional: a compiler where the prebuilt kernels leave speed unused))."""
    from . import __version__
    cpu, cc, nv, tpu = _cpu(), _compiler(), _nvidia(), _tpu()
    pk = {m: _has(m) for m in ('numpy', 'scipy', 'jax', 'cupy') + tuple(m for m, _, _ in _OPTIONAL)}
    todo, hints, backends = [], [], {}
    supported = (sys.platform == 'darwin' or sys.platform.startswith('linux')) and cpu['machine'] in ('x86_64', 'AMD64', 'aarch64', 'arm64')
    kern = _kernels(cc)
    pre = sorted({k['level'] for k in kern['families'].values() if k['path'] == 'prebuilt'})
    # cpu
    if not supported:
        backends['cpu'] = dict(ok=False, why=f"the C++ kernels are for Linux and macOS on x86-64 or 64-bit ARM; use backend='jax' here")
    elif pre and all(k['path'] for k in kern['families'].values()):
        why = 'no C++ compiler' if not cc['found'] else f"{cc['command']} cannot link OpenMP programs" if cc['openmp'] is False else None
        backends['cpu'] = dict(ok=True, why=f"prebuilt kernels ({', '.join(pre)})" + (f'; {why}' if why else f"; FASTSAR_KERNELS={kern['mode']}"))
        more = _beyond(cpu, pre)
        if why and more:
            hints.append(dict(what=f'a C++ compiler would make the CPU backend faster on this machine: the prebuilt kernels use '
                                   f"{', '.join(pre)}, a native build also uses {more}", command=cc['install'], pip=False))
    elif not cc['found']:
        backends['cpu'] = dict(ok=False, why=f"no C++ compiler ({cc['command']}) and " + (
            'FASTSAR_KERNELS=native' if kern['mode'] == 'native' else 'no prebuilt kernels for this processor in this '
            'installation (the platform wheels carry them)'))
        todo.append(dict(what='a C++ compiler for the CPU kernels', command=cc['install'], pip=False))
    elif cc['openmp'] is False:
        backends['cpu'] = dict(ok=False, why=f"{cc['command']} cannot link OpenMP programs")
        todo.append(dict(what='the OpenMP runtime of the C++ compiler', command=cc['install'], pip=False))
    else:
        backends['cpu'] = dict(ok=True, why=f"{cc['version'] or cc['command']}; {', '.join(cpu['simd']) or 'no vector extensions found'}")
    # cuda
    if nv is None:
        backends['cuda'] = dict(ok=False, why='no Nvidia GPU found (nvidia-smi)')
    else:
        # the driver runs programs built for its CUDA version or an earlier one; the [ctk] extra (CuPy 14, CUDA 12
        # and 13) brings the CUDA runtime libraries with it, so that no system toolkit is needed
        major = min(nv['cuda'][0], 13) if nv['cuda'] else 12
        wheel = f'cupy-cuda{major}x[ctk]>=14' if major >= 12 else f'cupy-cuda{major}x>=13'
        if not pk['cupy']:
            backends['cuda'] = dict(ok=False, why=f"{nv['gpus'][0]}: CuPy is not installed")
            todo.append(dict(what=f"CuPy for the GPU (driver CUDA {'.'.join(map(str, nv['cuda'])) if nv['cuda'] else 'version unknown'})",
                             command=f'{sys.executable} -m pip install "{wheel}"', pip=True))
        else:
            ok, why = True, nv['gpus'][0]
            if probe:
                try:
                    import cupy
                    n = cupy.cuda.runtime.getDeviceCount()
                    cupy.arange(4).sum().item()                      # the libraries load and a kernel runs
                    ok, why = n > 0, f"{nv['gpus'][0]}, CuPy {cupy.__version__}"
                except Exception as e:                               # a CuPy built for another CUDA version, missing libraries
                    ok, why = False, f'CuPy is installed but does not run on this GPU ({type(e).__name__}: {str(e)[:120]})'
                    todo.append(dict(what=f'CuPy built for CUDA {major}', pip=True,
                                     command=f'{sys.executable} -m pip install --force-reinstall "{wheel}"'))
            backends['cuda'] = dict(ok=ok, why=why)
    # jax and tpu
    if not pk['jax']:
        backends['jax'] = dict(ok=False, why='JAX is not installed')
        todo.append(dict(what='JAX', command=f'{sys.executable} -m pip install "jax>=0.6.2"', pip=True))
    else:
        backends['jax'] = dict(ok=True, why='any device JAX has')
    if tpu:
        ok, why = False, 'a TPU VM, but JAX does not see the TPU'
        if pk['jax'] and probe:
            try:
                import jax
                if any(d.platform == 'tpu' for d in jax.devices()):
                    ok, why = True, f'{len(jax.devices())} TPU device(s), JAX {jax.__version__}'
            except Exception as e:
                why = f'JAX cannot start the TPU ({type(e).__name__}: {str(e)[:120]})'
        elif pk['jax'] and _has('libtpu'):
            ok, why = True, 'libtpu is installed'
        if not ok:
            todo.append(dict(what='the TPU build of JAX', pip=True,
                             command=f'{sys.executable} -m pip install "jax[tpu]>=0.6.2" -f {_TPU_INDEX}'))
        backends['tpu'] = dict(ok=ok, why=why)
    else:
        backends['tpu'] = dict(ok=False, why='not a Cloud TPU VM')
    if sys.platform == 'darwin':
        try:
            from . import metal
            backends['metal'] = dict(ok=bool(metal.available()), why='Apple GPU')
        except Exception as e:
            backends['metal'] = dict(ok=False, why=str(e)[:120])
    for mod, extra, use in _OPTIONAL:
        if not pk[mod]:
            todo.append(dict(what=f'{mod} ({use})', command=f'{sys.executable} -m pip install "fastsar[{extra}]"', pip=True))
    return dict(fastsar=__version__, python=platform.python_version(), cpu=cpu, compiler=cc, kernels=kern, nvidia=nv,
                tpu=tpu, packages=pk, backends=backends, todo=todo, hints=hints)


def report(c):
    """The check as text."""
    lines = [f"FastSAR {c['fastsar']}, Python {c['python']}, {c['cpu']['machine']} {c['cpu']['system']}, "
             f"{c['cpu']['cores']} cores", '', 'Backends:']
    for name, b in c['backends'].items():
        lines.append(f"  {name:6s} {'ready  ' if b['ok'] else 'missing'}  {b['why']}")
    k = c.get('kernels') or {}
    if k.get('families'):
        texts = sorted({f['text'] for f in k['families'].values()})
        if len(texts) == 1:
            lines += ['', f"CPU kernels: {texts[0]}" + (f" (FASTSAR_KERNELS={k['mode']})" if k['mode'] else '')]
        else:
            lines += ['', 'CPU kernels:'] + [f"  {f['what']:26s} {f['text']}" for f in k['families'].values()]
    lines += ['', 'Packages: ' + ', '.join(f"{m} {'yes' if ok else 'no'}" for m, ok in c['packages'].items())]
    if c['todo']:
        lines += ['', 'To complete the installation:']
        lines += [f"  {t['what']}:\n      {t['command']}" for t in c['todo']]
        if any(t['pip'] for t in c['todo']):
            lines += ['', 'python -m fastsar.doctor --install runs the pip commands above.']
    else:
        lines += ['', 'Nothing is missing.']
    if c.get('hints'):
        lines += ['', 'Optional:'] + [f"  {t['what']}:\n      {t['command']}" for t in c['hints']]
    return '\n'.join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m fastsar.doctor', description=__doc__.split('\n\n')[0])
    ap.add_argument('--install', action='store_true', help='run the pip commands that complete the installation')
    ap.add_argument('--json', action='store_true', help='print the check as JSON')
    ap.add_argument('--no-probe', action='store_true', help='do not import CuPy and JAX to ask for their devices')
    a = ap.parse_args(argv)
    c = check(probe=not a.no_probe)
    if a.install:
        import shlex
        for t in [t for t in c['todo'] if t['pip']]:
            print(f"installing {t['what']}: {t['command']}", flush=True)
            if subprocess.run(shlex.split(t['command'])).returncode != 0:
                print('  failed; run it by hand to see why (an environment made by uv has no pip: use `uv pip install`)')
        c = check(probe=not a.no_probe)
    print(json.dumps(c, indent=1) if a.json else report(c))
    return 0 if any(b['ok'] for b in c['backends'].values()) else 1


if __name__ == '__main__':
    sys.exit(main())
