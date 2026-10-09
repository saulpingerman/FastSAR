"""Compiling the C++ kernels on first use: one shared object per source, compiler, flags and host, cached in
~/.cache/fastsar. The source is compiled with the given flags and linked in a separate step without them, so that
-ffast-math, when a kernel is compiled with it, does not link crtfastmath.o (which would switch the whole process to
flush-to-zero on load). Concurrent first builds each write their own temporary files and the last os.replace wins,
with identical contents."""
import hashlib
import os
import platform
import subprocess


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


def shared_object(name, src, flags):
    """Path of the shared object built from the C++ source text src with the compiler flags (a list), building it
    if it is not cached. The compiler is $CXX (default g++); OpenMP and -fPIC are added."""
    cxx = os.environ.get('CXX', 'g++')
    key = '\0'.join([src, cxx, ' '.join(flags), _host_tag()])
    tag = hashlib.sha1(key.encode()).hexdigest()[:12]
    d = os.path.join(os.path.expanduser('~'), '.cache', 'fastsar')
    os.makedirs(d, exist_ok=True)
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
