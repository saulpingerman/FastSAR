"""Compiling the C++ kernels on first use: one shared object per source, compiler, flags and host, cached in
$FASTSAR_CACHE_DIR, else $XDG_CACHE_HOME/fastsar, else ~/.cache/fastsar. The source is compiled with the given flags and linked in a separate step without them, so that
-ffast-math, when a kernel is compiled with it, does not link crtfastmath.o (which would switch the whole process to
flush-to-zero on load). Concurrent first builds each write their own temporary files and the last os.replace wins,
with identical contents."""
import hashlib
import os
import platform
import shutil
import subprocess
import sys


MACHINES = ('x86_64', 'AMD64', 'aarch64', 'arm64')


def default_flags():
    """The optimization flags of the C++ kernels for this machine: -O3 -march=native on Linux, and on x86-64 a
    preference for 512-bit vectors (the kernels' 16-float vector type maps to one AVX-512 register; on ARM the
    compiler splits it over NEON or SVE registers); -O3 alone on macOS, where Apple's clang has no -march=native
    and NEON is the baseline of Apple silicon. FFBP_CPU_FLAGS replaces them."""
    env = os.environ.get('FFBP_CPU_FLAGS')
    if env:
        return env.split()
    if sys.platform == 'darwin':
        return ['-O3']
    flags = ['-O3', '-march=native']
    if platform.machine() in ('x86_64', 'AMD64'):
        flags.append('-mprefer-vector-width=512')
    return flags


def compiler():
    """The C++ compiler and the OpenMP flags for compiling and linking: $CXX; on macOS a Homebrew GCC (g++-15 to
    g++-12) if one is on the PATH, else clang++ with Homebrew's libomp. -> (cxx, compile flags, link flags)."""
    cxx = os.environ.get('CXX')
    if cxx:
        return cxx, ['-fopenmp'], ['-fopenmp']
    if sys.platform != 'darwin':
        return 'g++', ['-fopenmp'], ['-fopenmp']
    for v in range(15, 11, -1):
        if shutil.which(f'g++-{v}'):
            return f'g++-{v}', ['-fopenmp'], ['-fopenmp']
    prefix = None
    try:
        prefix = subprocess.run(['brew', '--prefix', 'libomp'], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    if prefix and os.path.isdir(prefix) and shutil.which('clang++'):
        return 'clang++', ['-Xpreprocessor', '-fopenmp', f'-I{prefix}/include'], [f'-L{prefix}/lib', '-lomp']
    raise RuntimeError("fastsar's cpu backend needs a C++ compiler with OpenMP: on macOS install GCC (brew install gcc) "
                       "or libomp for Apple's clang (brew install libomp), or set CXX")


def _check_gcc(cxx):
    """On ARM the vector extensions need GCC 12 or later (earlier releases mis-lower the 64-byte vector type)."""
    try:
        out = subprocess.run([cxx, '-dumpfullversion'], capture_output=True, text=True, timeout=30).stdout.strip()
        major = int(out.split('.')[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return
    if major < 12:
        raise RuntimeError(f"fastsar's cpu backend needs GCC 12 or later on ARM; {cxx} is {out}")


def _host_tag():
    """The machine and its CPU's instruction set extensions (a -march=native build must not be loaded on another
    CPU sharing the same home directory)."""
    tag = platform.machine()
    if sys.platform == 'darwin':
        try:
            return tag + subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return tag + platform.processor()
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith(('flags', 'Features')):
                    return tag + line
    except OSError:
        pass
    return tag + platform.processor()


def cache_dir():
    """The directory of the compiled kernels: $FASTSAR_CACHE_DIR, else $XDG_CACHE_HOME/fastsar, else
    ~/.cache/fastsar, created if needed."""
    d = os.environ.get('FASTSAR_CACHE_DIR') or os.path.join(
        os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache'), 'fastsar')
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f'fastsar cannot create its kernel cache directory {d!r} ({e.strerror}); set '
                           'FASTSAR_CACHE_DIR to a writable directory') from None
    return d


def shared_object(name, src, flags):
    """Path of the shared object built from the C++ source text src with the compiler flags (a list), building it
    if it is not cached. The compiler is $CXX (default g++); OpenMP and -fPIC are added."""
    if not (sys.platform == 'darwin' or sys.platform.startswith('linux')) or platform.machine() not in MACHINES:
        raise RuntimeError(f"fastsar's cpu backend compiles C++ kernels for Linux and macOS on x86-64 or 64-bit ARM; "
                           f'this is {platform.machine()} {sys.platform}. Use backend="jax" (any device JAX supports) '
                           'or run on a machine of those kinds')
    cxx, omp_c, omp_l = compiler()
    if platform.machine() in ('aarch64', 'arm64') and 'g++' in os.path.basename(cxx):
        _check_gcc(cxx)
    key = '\0'.join([src, cxx, ' '.join(flags), _host_tag()])
    tag = hashlib.sha1(key.encode()).hexdigest()[:12]
    d = cache_dir()
    so = os.path.join(d, f'lib{name}_{tag}.so')
    if os.path.exists(so):
        return so
    stem = os.path.join(d, f'{name}_{tag}.{os.getpid()}')
    cpp, obj, tmp = stem + '.cpp', stem + '.o', stem + '.so.tmp'
    with open(cpp, 'w') as fh:
        fh.write(src)
    try:
        for cmd in ([cxx] + list(flags) + omp_c + ['-fPIC', '-c', cpp, '-o', obj],
                    [cxx, '-shared'] + omp_l + [obj, '-o', tmp]):
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except FileNotFoundError:
                raise RuntimeError(f"fastsar's cpu backend compiles its C++ kernels on first use and needs a C++ compiler "
                                   f'with OpenMP: {cxx!r} was not found. Install g++ (Debian/Ubuntu: apt install g++; '
                                   'RHEL/Fedora: dnf install gcc-c++) or point CXX at one') from None
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f'compiling the {name} kernel failed ({" ".join(cmd)}):\n{e.stderr.strip()}') from None
        os.replace(tmp, so)
    finally:
        for f in (cpp, obj, tmp):
            try:
                os.remove(f)
            except OSError:
                pass
    return so
