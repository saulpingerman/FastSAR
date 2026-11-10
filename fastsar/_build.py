"""The shared objects of the C++ kernels: compiled on this machine on first use, or prebuilt in a platform wheel.

Native build: one shared object per source, compiler, flags and host, cached in $FASTSAR_CACHE_DIR, else
$XDG_CACHE_HOME/fastsar, else ~/.cache/fastsar. The source is compiled with the given flags and linked in a separate
step without them, so that -ffast-math, when a kernel is compiled with it, does not link crtfastmath.o (which would
switch the whole process to flush-to-zero on load). Concurrent first builds each write their own temporary files and
the last os.replace wins, with identical contents.

Prebuilt objects: a platform wheel carries every kernel compiled at a few instruction-set levels
(fastsar/_prebuilt, with a manifest of the source hash of each). shared_object uses them where there is no
compiler or the native build fails; FASTSAR_KERNELS=native or prebuilt forces one path, and
FASTSAR_KERNEL_LEVEL one level (for tests)."""
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
    compiler splits it over NEON or SVE registers); -O3 -mcpu=native on macOS (Apple's clang has no -march=native).
    FFBP_CPU_FLAGS replaces them."""
    env = os.environ.get('FFBP_CPU_FLAGS')
    if env:
        return env.split()
    if sys.platform == 'darwin':
        return ['-O3', '-mcpu=native']
    flags = ['-O3', '-march=native']
    if platform.machine() in ('x86_64', 'AMD64'):
        flags.append('-mprefer-vector-width=512')
    return flags


def compiler():
    """The C++ compiler and the threading flags for compiling and linking: $CXX with OpenMP; on Linux g++ with
    OpenMP; on macOS Apple's clang++ with the simd pragmas only (-fopenmp-simd) and no OpenMP runtime, since a second
    runtime next to the one a wheel such as finufft's bundles deadlocks (par.hpp then runs the loops on std::thread).
    -> (cxx, compile flags, link flags)."""
    cxx = os.environ.get('CXX')
    if cxx:
        return cxx, ['-fopenmp'], ['-fopenmp']
    if sys.platform != 'darwin':
        return 'g++', ['-fopenmp'], ['-fopenmp']
    if shutil.which('clang++'):
        return 'clang++', ['-fopenmp-simd'], []
    for v in range(20, 11, -1):
        if shutil.which(f'g++-{v}'):
            return f'g++-{v}', ['-fopenmp'], ['-fopenmp']
    raise RuntimeError("fastsar's cpu backend needs a C++ compiler: on macOS install the command line tools "
                       "(xcode-select --install), or set CXX")


def shared_object_mm(name, src):
    """Path of the shared library built from the Objective-C++ source text src with Apple's clang against the Metal
    and Foundation frameworks (macOS only), cached as shared_object caches."""
    if sys.platform != 'darwin':
        raise RuntimeError('Metal is available on macOS only')
    cxx = 'clang++'
    flags = ['-O2', '-std=c++17', '-ObjC++', '-fobjc-arc']
    key = '\0'.join([src, cxx, ' '.join(flags), _host_tag()])
    tag = hashlib.sha1(key.encode()).hexdigest()[:12]
    d = cache_dir()
    so = os.path.join(d, f'lib{name}_{tag}.dylib')
    if os.path.exists(so):
        return so
    stem = os.path.join(d, f'{name}_{tag}.{os.getpid()}')
    mm, tmp = stem + '.mm', stem + '.dylib.tmp'
    with open(mm, 'w') as fh:
        fh.write(src)
    try:
        cmd = [cxx] + flags + ['-shared', '-fPIC', '-framework', 'Metal', '-framework', 'Foundation', mm, '-o', tmp]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError("the Metal bridge needs Apple's clang (xcode-select --install)") from None
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'compiling the Metal bridge failed ({" ".join(cmd)}):\n{e.stderr.strip()}') from None
        os.replace(tmp, so)
    finally:
        for f in (mm, tmp):
            try:
                os.remove(f)
            except OSError:
                pass
    return so


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


def _supported():
    return (sys.platform == 'darwin' or sys.platform.startswith('linux')) and platform.machine() in MACHINES


def compiler_found(cxx):
    """Whether the compiler command exists. On macOS /usr/bin/clang++ exists on every machine: without the command
    line tools it is a stub that opens an installation dialog and fails, so there the tools are asked for
    (xcode-select -p, which opens nothing)."""
    path = shutil.which(cxx)
    if path is None:
        return False
    if sys.platform == 'darwin' and os.path.dirname(os.path.abspath(path)) == '/usr/bin':
        try:
            return subprocess.run(['/usr/bin/xcode-select', '-p'], capture_output=True, timeout=10).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    return True


def _no_compiler(cxx):
    return (f"fastsar's cpu backend compiles its C++ kernels on first use and needs a C++ compiler "
            f'with OpenMP: {cxx!r} was not found. Install g++ (Debian/Ubuntu: apt install g++; '
            'RHEL/Fedora: dnf install gcc-c++) or point CXX at one. python -m fastsar.doctor lists '
            'what this machine is missing')


def _native(name, src, flags, build=True):
    """The shared object compiled on this machine with its compiler and the flags, from the cache or built now.
    build False: the cached object's path or None, and the path of the note an earlier failed build left,
    -> (so, note path)."""
    cxx, omp_c, omp_l = compiler()
    if platform.machine() in ('aarch64', 'arm64') and 'g++' in os.path.basename(cxx):
        _check_gcc(cxx)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, 'par.hpp')) as fh:
        par = fh.read()
    key = '\0'.join([src, par, cxx, ' '.join(flags), _host_tag()])
    tag = hashlib.sha1(key.encode()).hexdigest()[:12]
    d = cache_dir()
    so = os.path.join(d, f'lib{name}_{tag}.so')
    if not build:
        return (so if os.path.exists(so) else None), so + '.failed'
    if os.path.exists(so):
        return so
    stem = os.path.join(d, f'{name}_{tag}.{os.getpid()}')
    cpp, obj, tmp = stem + '.cpp', stem + '.o', stem + '.so.tmp'
    with open(cpp, 'w') as fh:
        fh.write(src)
    try:
        for cmd in ([cxx] + list(flags) + omp_c + ['-I' + here, '-fPIC', '-c', cpp, '-o', obj],
                    [cxx, '-shared'] + omp_l + [obj, '-o', tmp]):
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except FileNotFoundError:
                raise RuntimeError(_no_compiler(cxx)) from None
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


# ----------------------------------------------------------------------------------------------- prebuilt kernels
#
# A platform wheel carries each kernel compiled at a few instruction-set levels (fastsar/_prebuilt, written by
# build_prebuilt when the wheel is built). They are used where there is no compiler; a compiler, when present,
# still builds for the exact CPU.

# name, source (a file of the package, or module.py:NAME for a string in a module), flags the call site adds to
# the base flags, whether the call site prefers 512-bit vectors (default_flags does; backproject's flags do not),
# what the kernel serves
KERNELS = (('ffbp_cpu', 'ffbp_cpu.cpp', ('-funroll-loops', '-std=c++17'), True, 'factorized backprojection'),
           ('exact_cpu', 'exact.py:_CPU_SRC', ('-ffast-math',), True, 'exact backprojection'),
           ('bp_cpu', 'bp.py:_CPU_SRC', (), False, 'backprojection to points'),
           ('pfa_cpu', 'pfa_cpu.cpp', ('-std=c++17',), True, 'polar format'),
           ('resample_cpu', 'resample_cpu.cpp', ('-std=c++17',), True, 'resampling'),
           ('io_cpu', 'io_cpu.cpp', ('-std=c++17', '-fopenmp-simd'), True, 'file readers'),
           ('s1_decode', 'sentinel1.py:_SRC', ('-std=c++17',), True, 'Sentinel-1 raw decoder'))

# per platform, lowest level first: name, what reports call it, compiler flags, the CPU flags it needs
# (/proc/cpuinfo names). x86-64-v4 is AVX-512 F, BW, CD, DQ and VL: Intel Skylake-X and later servers, Ice, Tiger
# and Rocket Lake, AMD Zen 4 and later
LEVELS = {
    ('linux', 'x86_64'): (
        ('v2', 'x86-64-v2', ('-march=x86-64-v2',), ('cx16', 'lahf_lm', 'popcnt', 'pni', 'sse4_1', 'sse4_2', 'ssse3')),
        ('v3', 'x86-64-v3', ('-march=x86-64-v3',),
         ('avx', 'avx2', 'bmi1', 'bmi2', 'f16c', 'fma', 'abm', 'movbe', 'xsave')),
        ('v4', 'x86-64-v4', ('-march=x86-64-v4',), ('avx512f', 'avx512bw', 'avx512cd', 'avx512dq', 'avx512vl'))),
    ('linux', 'aarch64'): (('neon', 'armv8-a', ('-march=armv8-a',), ('asimd',)),),
    ('darwin', 'arm64'): (('apple', 'Apple silicon', ('-mcpu=apple-m1',), ()),),
}
PREBUILT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_prebuilt')
USED = {}        # kernel name -> dict(path 'native' or 'prebuilt', level, file, note): what this process loaded
_MANIFEST = []


def _platform():
    m = {'AMD64': 'x86_64'}.get(platform.machine(), platform.machine())
    return ('darwin' if sys.platform == 'darwin' else 'linux' if sys.platform.startswith('linux') else sys.platform), m


def kernel_source(spec):
    """The source text of a KERNELS entry's source field, read without importing the module that holds it."""
    here = os.path.dirname(os.path.abspath(__file__))
    file, _, var = spec.partition(':')
    with open(os.path.join(here, file)) as fh:
        text = fh.read()
    if not var:
        return text
    import ast
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == var for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f'{file} has no string {var}')


def source_hash(src, flags):
    """What ties a prebuilt object to its source: the hash of the source text, par.hpp and the flags that change
    what the code means (-std, -ffast-math, -fopenmp-simd, definitions). Flags that only tune (-O, -m...,
    -funroll-loops) are left out: they differ between the levels by design."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'par.hpp')) as fh:
        par = fh.read()
    keep = [f for f in flags if not f.startswith(('-O', '-m', '-funroll'))]
    return hashlib.sha1('\0'.join([src, par, ' '.join(keep)]).encode()).hexdigest()[:16]


def _cpu_flags():
    """The flags (x86) or features (ARM) line of /proc/cpuinfo as a set; empty where there is none."""
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith(('flags', 'Features')):
                    return set(line.split(':', 1)[1].split())
    except OSError:
        pass
    return set()


def cpu_levels(have=None, plat=None):
    """The instruction-set levels this CPU runs, best first, as (name, label): from the flags line of /proc/cpuinfo
    on Linux (the kernel lists AVX and AVX-512 only when it saves their registers), from sysctl on macOS. A level is
    listed only if every level below it is: its objects assume them. have, plat: the flags and the (system,
    machine) of another machine, for tests."""
    plat = plat or _platform()
    if have is None:
        have = set()
        if plat[0] == 'linux':
            have = _cpu_flags()
        elif plat == ('darwin', 'arm64'):
            try:                                                   # (a PATH without /usr/sbin still finds it)
                out = subprocess.run([shutil.which('sysctl') or '/usr/sbin/sysctl', '-n', 'hw.optional.arm64'],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                out = ''
            if out not in ('1', ''):                               # every arm64 Mac is Apple silicon; 0 would deny it
                return []
    out = []
    for name, label, _, need in LEVELS.get(plat, ()):
        if not set(need) <= set(have):
            break
        out.insert(0, (name, label))
    return out


def prebuilt_manifest():
    """The manifest of the prebuilt kernels in this installation (fastsar/_prebuilt/manifest.json: platform,
    compiler, threads, objects[kernel][level] = dict(file, hash)), or None: an installation from the source
    distribution or from a checkout has none."""
    if not _MANIFEST:
        m = None
        try:
            import json
            with open(os.path.join(PREBUILT_DIR, 'manifest.json')) as fh:
                m = json.load(fh)
            if m.get('platform') != '-'.join(_platform()):
                m = None
        except (OSError, ValueError):
            m = None
        _MANIFEST.append(m)
    return _MANIFEST[0]


def _level_wanted():
    """FASTSAR_KERNEL_LEVEL (for tests: one level's prebuilt objects), checked against the CPU: an object of a
    level the CPU lacks would stop the process with an illegal instruction."""
    want = os.environ.get('FASTSAR_KERNEL_LEVEL', '').strip().lower()
    if want and want not in [n for n, _ in cpu_levels()]:
        raise RuntimeError(f'FASTSAR_KERNEL_LEVEL={want}: this CPU runs the levels '
                           f'{", ".join(n for n, _ in cpu_levels()) or "(none)"} of this platform')
    return want


def _load_error(path):
    """None if the shared object loads (its system libraries are present), else the loader's message."""
    try:
        import ctypes
        ctypes.CDLL(path)
    except OSError as e:
        return str(e)


def prebuilt(name, src, flags, load=True):
    """The prebuilt object of the kernel for the best level this CPU runs (or FASTSAR_KERNEL_LEVEL), whose recorded
    hash is that of the source and flags asked for. load: also check that the object loads (its system libraries
    are present). -> (path, level label, None) or (None, None, the reason there is none)."""
    m = prebuilt_manifest()
    if m is None:
        return None, None, ('this installation carries no prebuilt kernels for this platform (the platform wheels do; '
                            'an installation from the source distribution or from a checkout does not)')
    want = _level_wanted()
    h = source_hash(src, flags)
    why = f'this installation carries no prebuilt {name} kernel for this CPU'
    for level, label in cpu_levels():
        o = m.get('objects', {}).get(name, {}).get(level)
        if o is None or (want and level != want):
            continue
        if o['hash'] != h:
            why = (f'the prebuilt {name} kernel was built from another source or with other flags than the ones asked '
                   'for (FFBP_CPU_FLAGS with options beyond -O and -m needs a compiler)')
            continue
        path = os.path.join(PREBUILT_DIR, o['file'])
        if load:
            e = _load_error(path)
            if e:
                why = f'the prebuilt {name} kernel ({label}) does not load: {e}'
                continue
        return path, label, None
    return None, None, why


def kernels_mode():
    """FASTSAR_KERNELS: 'native', 'prebuilt' or '' (unset: a native build where there is a compiler, else the
    prebuilt objects). FASTSAR_KERNEL_LEVEL alone means prebuilt."""
    mode = os.environ.get('FASTSAR_KERNELS', '').strip().lower()
    if mode not in ('', 'native', 'prebuilt'):
        raise ValueError(f'FASTSAR_KERNELS must be native or prebuilt, not {mode!r}')
    return mode or ('prebuilt' if os.environ.get('FASTSAR_KERNEL_LEVEL', '').strip() else '')


def shared_object(name, src, flags):
    """Path of the shared object of the kernel with the C++ source text src and the compiler flags (a list).

    FASTSAR_KERNELS unset: the object built on this machine (cached; the compiler is $CXX, default g++; the
    threading flags, the include path of par.hpp and -fPIC are added) where there is a compiler, else the prebuilt
    object a platform wheel carries for this CPU's instruction-set level; a failed native build falls back to the
    prebuilt object too. FASTSAR_KERNELS=native: the native build only. FASTSAR_KERNELS=prebuilt: the prebuilt
    object only. USED[name] records which was taken."""
    if not _supported():
        raise RuntimeError(f"fastsar's cpu backend compiles C++ kernels for Linux and macOS on x86-64 or 64-bit ARM; "
                           f'this is {platform.machine()} {sys.platform}. Use backend="jax" (any device JAX supports) '
                           'or run on a machine of those kinds')
    mode = kernels_mode()
    if mode == 'native':
        so = _native(name, src, flags)
        USED[name] = dict(path='native', level=None, file=so, note=None)
        return so
    if mode == 'prebuilt':
        so, label, why = prebuilt(name, src, flags)
        if so is None:
            raise RuntimeError(f'FASTSAR_KERNELS=prebuilt (or FASTSAR_KERNEL_LEVEL): {why}')
        USED[name] = dict(path='prebuilt', level=label, file=so, note=None)
        return so
    cxx, failed, err, tried = None, None, None, False
    try:
        so, failed = _native(name, src, flags, build=False)
        if so is not None:
            USED[name] = dict(path='native', level=None, file=so, note=None)
            return so
        cxx = compiler()[0]
    except RuntimeError as e:                                      # macOS without a compiler, GCC before 12 on ARM
        err = e
    if cxx is not None and not compiler_found(cxx):
        err = RuntimeError(_no_compiler(cxx))
    elif cxx is not None and os.path.exists(failed) and prebuilt(name, src, flags)[0] is not None:
        err = RuntimeError(f'an earlier native build failed (remove {failed} to try again)')
    elif cxx is not None:
        try:
            so = _native(name, src, flags)
            USED[name] = dict(path='native', level=None, file=so, note=None)
            return so
        except RuntimeError as e:
            err, tried = e, True
    pre, label, why = prebuilt(name, src, flags)                   # looked up only now: a native build loads no other
    if pre is None:
        raise RuntimeError(f'{err}\n{why[0].upper() + why[1:]}.') from None
    if tried:                                                      # so that later processes do not compile again
        try:
            with open(failed, 'w') as fh:
                fh.write(str(err))
        except OSError:
            pass
    USED[name] = dict(path='prebuilt', level=label, file=pre, note=str(err).splitlines()[0])
    return pre


def status():
    """What decides each kernel's object in this process, without building anything: dict(mode (kernels_mode()),
    levels (this CPU's labels, best first), prebuilt (the manifest's platform, compiler and threads, or None),
    kernels: name -> dict(what (the kernel's use), prebuilt (the label of the level whose object matches the
    source, or None), why (the reason there is none), used (the USED entry once the kernel is loaded)))."""
    m = prebuilt_manifest()
    out = dict(mode=kernels_mode(), levels=[label for _, label in cpu_levels()], kernels={},
               prebuilt=None if m is None else {k: m.get(k) for k in ('platform', 'compiler', 'threads')})
    for name, spec, extra, _, what in KERNELS:
        try:
            if not _supported():
                raise RuntimeError('the C++ kernels are for Linux and macOS on x86-64 or 64-bit ARM')
            _, label, why = prebuilt(name, kernel_source(spec), list(extra), load=False)
        except RuntimeError as e:
            label, why = None, str(e)
        out['kernels'][name] = dict(what=what, prebuilt=label, why=why, used=USED.get(name))
    return out


def build_prebuilt(dest, cxx=None, openmp=None, jobs=None, log=print):
    """Compile every kernel of KERNELS at every level of this platform into the directory dest and write
    dest/manifest.json (run when a platform wheel is built; setup.py calls it).

    The objects must load on a machine with no compiler and no OpenMP runtime, so they are built without OpenMP
    (openmp, or FASTSAR_PREBUILT_OPENMP=1, builds them with it for comparison: they then need libgomp): the simd
    pragmas by -fopenmp-simd and par.hpp's std::thread loops, on Linux with its kept threads (FASTSAR_PAR_POOL;
    measured as fast as the OpenMP build). On Linux they carry their own copy of the C++ runtime
    (-static-libstdc++ -static-libgcc, its symbols hidden) and need only the C library.
    -> the manifest."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    plat = _platform()
    levels = LEVELS.get(plat)
    if not levels:
        raise RuntimeError(f'no prebuilt kernel levels are defined for {"-".join(plat)}')
    if openmp is None:
        openmp = os.environ.get('FASTSAR_PREBUILT_OPENMP', '0') == '1'
    darwin = plat[0] == 'darwin'
    cxx = cxx or os.environ.get('CXX') or ('clang++' if darwin else 'g++')
    if plat[1] in ('aarch64', 'arm64') and 'g++' in os.path.basename(cxx):
        _check_gcc(cxx)
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(dest, exist_ok=True)
    thr_c = ['-fopenmp'] if openmp else ['-fopenmp-simd'] + ([] if darwin else ['-pthread', '-DFASTSAR_PAR_POOL'])
    link = ['-fopenmp'] if openmp else ([] if darwin else ['-pthread'])
    if not darwin:
        link += ['-static-libstdc++', '-static-libgcc', '-Wl,--exclude-libs,ALL', '-Wl,--gc-sections', '-Wl,-s',
                 '-Wl,-z,noexecstack']
    ext = '.dylib' if darwin else '.so'
    version = subprocess.run([cxx, '--version'], capture_output=True, text=True).stdout.splitlines()[0].strip()
    manifest = dict(platform='-'.join(plat), compiler=version, threads='OpenMP' if openmp else 'std::thread', objects={})
    jobs_ = []
    for name, spec, extra, wide, _ in KERNELS:
        src = kernel_source(spec)
        manifest['objects'][name] = {}
        for level, label, march, _ in levels:
            flags = ['-O3'] + list(march) + (['-mprefer-vector-width=512'] if wide and level == 'v4' else []) + list(extra)
            file = f'{name}.{level}{ext}'
            manifest['objects'][name][level] = dict(file=file, hash=source_hash(src, flags), flags=' '.join(flags + thr_c))
            jobs_.append((name, src, flags, file))

    def one(job):
        name, src, flags, file = job
        stem = os.path.join(dest, file)
        cpp, obj = stem + '.cpp', stem + '.o'
        with open(cpp, 'w') as fh:
            fh.write(src)
        try:
            for cmd in ([cxx] + flags + thr_c + ['-I' + here, '-fPIC', '-ffunction-sections', '-c', cpp, '-o', obj],
                        [cxx, '-shared', obj, '-o', stem] + link):
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f'building the prebuilt {file} failed ({" ".join(cmd)}):\n{r.stderr.strip()}')
        finally:
            for f in (cpp, obj):
                try:
                    os.remove(f)
                except OSError:
                    pass
        log(f'prebuilt kernel {file}: {os.path.getsize(stem)} bytes')

    with ThreadPoolExecutor(jobs or min(8, os.cpu_count() or 1)) as ex:
        list(ex.map(one, jobs_))
    with open(os.path.join(dest, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')
    return manifest
