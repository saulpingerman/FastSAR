"""Compiling the C++ kernels on first use: one shared object per source, compiler, flags and host, cached in
$FASTSAR_CACHE_DIR, else $XDG_CACHE_HOME/fastsar, else ~/.cache/fastsar. The source is compiled with the given flags and linked in a separate step without them, so that
-ffast-math, when a kernel is compiled with it, does not link crtfastmath.o (which would switch the whole process to
flush-to-zero on load). Concurrent first builds each write their own temporary files and the last os.replace wins,
with identical contents."""
import hashlib
import os
import platform
import subprocess
import sys


def _host_tag():
    """The machine and its CPU's instruction set extensions (a -march=native build must not be loaded on another
    CPU sharing the same home directory)."""
    tag = platform.machine()
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
    if not sys.platform.startswith('linux') or platform.machine() not in ('x86_64', 'AMD64'):
        raise RuntimeError(f"fastsar's cpu backend compiles C++ kernels for x86-64 Linux; this is "
                           f'{platform.machine()} {sys.platform}. Use backend="jax" (any device JAX supports) '
                           'or run on an x86-64 Linux machine')
    cxx = os.environ.get('CXX', 'g++')
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
        for cmd in ([cxx] + list(flags) + ['-fopenmp', '-fPIC', '-c', cpp, '-o', obj],
                    [cxx, '-shared', '-fopenmp', obj, '-o', tmp]):
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
