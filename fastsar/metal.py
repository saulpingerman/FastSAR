"""Exact backprojection on Apple GPUs through Metal (backend='metal' of ExactFormer): the CUDA tile kernel of
fastsar.exact as a Metal compute kernel, one threadgroup per tile of TX by TY pixels, PX pixels per thread, the tile
center's per-pulse terms staged in threadgroup memory. Apple GPUs have no float64, so the terms the CUDA kernel
computes in float64 per pulse and tile (the unit vector to the antenna, the range curvature factors, the sample
position and the phase of the tile center) are computed by numpy in float64 on the CPU per chunk of pulses and
handed to the kernel as float32, which then does the same float32 work per pixel as the CUDA and C++ kernels.

The bridge to Metal is a small Objective-C++ library compiled on first use with Apple's clang (the command line
tools; no Xcode): the shader source is compiled at run time by the Metal framework. Buffers are shared (unified
memory); the image accumulates on the device across chunks and is copied out once.
"""
import ctypes
import os
import sys

import numpy as np

_BRIDGE = r'''
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <string>

struct Ctx {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipe;
    int tb;
};
struct Image {
    id<MTLBuffer> buf;
    size_t n;
};

static void put_err(char* err, int len, NSError* e, const char* what)
{
    if (!err || len <= 0) return;
    std::string s = what;
    if (e) { s += ": "; s += [[e localizedDescription] UTF8String]; }
    std::strncpy(err, s.c_str(), len - 1); err[len - 1] = 0;
}

extern "C" int fsm_available(char* name, int len)
{
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (!d) return 0;
        if (name && len > 0) { std::strncpy(name, [[d name] UTF8String], len - 1); name[len - 1] = 0; }
        return 1;
    }
}

// compile the kernel source (with its defines already in the text) and build the pipeline of function `fn`
extern "C" void* fsm_init(const char* src, const char* fn, int tb, char* err, int len)
{
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (!d) { put_err(err, len, nil, "no Metal device"); return nullptr; }
        NSError* e = nil;
        MTLCompileOptions* opt = [MTLCompileOptions new];
        opt.fastMathEnabled = YES;
        id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:src] options:opt error:&e];
        if (!lib) { put_err(err, len, e, "shader compilation failed"); return nullptr; }
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
        if (!f) { put_err(err, len, nil, "kernel function not found"); return nullptr; }
        id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:f error:&e];
        if (!p) { put_err(err, len, e, "pipeline creation failed"); return nullptr; }
        Ctx* c = new Ctx; c->dev = d; c->queue = [d newCommandQueue]; c->pipe = p; c->tb = tb;
        return c;
    }
}

extern "C" void* fsm_image_begin(void* ctx, long n)
{
    Ctx* c = (Ctx*)ctx;
    Image* im = new Image;
    im->n = (size_t)n;
    im->buf = [c->dev newBufferWithLength:(NSUInteger)(n * 8) options:MTLResourceStorageModeShared];
    if (!im->buf) { delete im; return nullptr; }
    std::memset([im->buf contents], 0, (size_t)n * 8);
    return im;
}

// out (complex64 [n]) += the image buffer; frees it
extern "C" void fsm_image_end(void* image, float* out)
{
    Image* im = (Image*)image;
    const float* src = (const float*)[im->buf contents];
    for (size_t i = 0; i < im->n * 2; ++i) out[i] += src[i];
    im->buf = nil;
    delete im;
}

struct Params { int n, W, nty, nx, ny, TX, TY, cubic; float sx, sy, fdr, fk; float e1[3]; float e2[3]; };

// rc: complex64 [n, W] windowed profiles of the chunk; terms float32 [ntiles, n, 8]; ti int32 [ntiles, n]
extern "C" int fsm_bp_tiles(void* ctx, void* image, const float* rc, int n, int W, const float* terms, const int* ti,
                            int ntiles, int nty, int TX, int TY, int nx, int ny, float sx, float sy, const float* e1,
                            const float* e2, float fdr, float fk, int cubic, char* err, int len)
{
    @autoreleasepool {
        Ctx* c = (Ctx*)ctx; Image* im = (Image*)image;
        const NSUInteger opt = MTLResourceStorageModeShared;
        id<MTLBuffer> brc = [c->dev newBufferWithBytes:rc length:(NSUInteger)n * W * 8 options:opt];
        id<MTLBuffer> bt = [c->dev newBufferWithBytes:terms length:(NSUInteger)ntiles * n * 32 options:opt];
        id<MTLBuffer> bi = [c->dev newBufferWithBytes:ti length:(NSUInteger)ntiles * n * 4 options:opt];
        if (!brc || !bt || !bi) { put_err(err, len, nil, "buffer allocation failed"); return 1; }
        Params prm; prm.n = n; prm.W = W; prm.nty = nty; prm.nx = nx; prm.ny = ny; prm.TX = TX; prm.TY = TY; prm.cubic = cubic;
        prm.sx = sx; prm.sy = sy; prm.fdr = fdr; prm.fk = fk;
        for (int i = 0; i < 3; ++i) { prm.e1[i] = e1[i]; prm.e2[i] = e2[i]; }
        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:c->pipe];
        [enc setBuffer:brc offset:0 atIndex:0];
        [enc setBuffer:bt offset:0 atIndex:1];
        [enc setBuffer:bi offset:0 atIndex:2];
        [enc setBytes:&prm length:sizeof(prm) atIndex:3];
        [enc setBuffer:im->buf offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(ntiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(c->tb, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] != MTLCommandBufferStatusCompleted) { put_err(err, len, [cb error], "kernel failed"); return 2; }
        return 0;
    }
}
'''

_MSL = r'''
#include <metal_stdlib>
using namespace metal;
#define TB __TB__
#define PX __PX__
#define CUBIC __CUBIC__
struct Params { int n, W, nty, nx, ny, TX, TY, cubic; float sx, sy, fdr, fk; float e1[3]; float e2[3]; };

kernel void bp_tiles(device const float2* rc [[buffer(0)]], device const float* terms [[buffer(1)]],
                     device const int* ti [[buffer(2)]], constant Params& prm [[buffer(3)]],
                     device float2* out [[buffer(4)]], uint b [[threadgroup_position_in_grid]],
                     uint j [[thread_index_in_threadgroup]])
{
    threadgroup float s_ux[TB], s_uy[TB], s_uz[TB], s_ir[TB], s_i2[TB], s_tf[TB], s_ph[TB];
    threadgroup int s_ti[TB];
    const int TX = prm.TX, TY = prm.TY, n = prm.n, W = prm.W;
    const int bx = (int)b / prm.nty, by = (int)b % prm.nty;
    float dx[PX], dy[PX], dz[PX], dd[PX], are[PX], aim[PX];
    int ix[PX], iy[PX];
    for (int m = 0; m < PX; ++m) {
        const int q = m * TB + (int)j, li = q / TY, lj = q % TY;
        ix[m] = bx * TX + li; iy[m] = by * TY + lj;
        const float a = (li - 0.5f * (TX - 1)) * prm.sx, c = (lj - 0.5f * (TY - 1)) * prm.sy;
        dx[m] = a * prm.e1[0] + c * prm.e2[0]; dy[m] = a * prm.e1[1] + c * prm.e2[1]; dz[m] = a * prm.e1[2] + c * prm.e2[2];
        dd[m] = dx[m] * dx[m] + dy[m] * dy[m] + dz[m] * dz[m];
        are[m] = 0.f; aim[m] = 0.f;
    }
    const float fdr = prm.fdr, fk = prm.fk;
    device const float* tb_terms = terms + (size_t)b * n * 8;
    device const int* tb_ti = ti + (size_t)b * n;
    for (int p0 = 0; p0 < n; p0 += TB) {
        const int p = p0 + (int)j;
        if (p < n) {
            device const float* t = tb_terms + (size_t)p * 8;
            s_ux[j] = t[0]; s_uy[j] = t[1]; s_uz[j] = t[2]; s_ir[j] = t[3]; s_i2[j] = t[4]; s_tf[j] = t[5]; s_ph[j] = t[6];
            s_ti[j] = tb_ti[p];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int np = min(TB, n - p0);
        for (int k = 0; k < np; ++k) {
            const float ux = s_ux[k], uy = s_uy[k], uz = s_uz[k], hir = s_ir[k], hi2 = s_i2[k], tfk = s_tf[k], phk = s_ph[k];
            device const float2* row = rc + (size_t)(p0 + k) * W + s_ti[k];
            for (int m = 0; m < PX; ++m) {
                const float du = dx[m] * ux + dy[m] * uy + dz[m] * uz;
                const float del = du + (dd[m] - du * du) * (hir - du * hi2);
                const float t = tfk + del * fdr, fl = floor(t), w = t - fl;
                const int i = (int)fl;
#if CUBIC
                const float2 a0 = row[i - 1], a1 = row[i], a2 = row[i + 1], a3 = row[i + 2];
                const float wm1 = w - 1.f, wm2 = w - 2.f, wp1 = w + 1.f;
                const float c0 = -w * wm1 * wm2 * (1.f / 6.f), c1 = wp1 * wm1 * wm2 * 0.5f;
                const float c2 = -wp1 * w * wm2 * 0.5f, c3 = wp1 * w * wm1 * (1.f / 6.f);
                const float vre = c0 * a0.x + c1 * a1.x + c2 * a2.x + c3 * a3.x, vim = c0 * a0.y + c1 * a1.y + c2 * a2.y + c3 * a3.y;
#else
                const float2 r0 = row[i], r1 = row[i + 1];
                const float vre = fma(w, r1.x - r0.x, r0.x), vim = fma(w, r1.y - r0.y, r0.y);
#endif
                float ph = fma(fk, del, phk);
                ph -= rint(ph);
                const float ang = 6.283185307179586f * ph;
                const float s = fast::sin(ang), c = fast::cos(ang);
                are[m] = fma(vre, c, fma(-vim, s, are[m]));
                aim[m] = fma(vre, s, fma(vim, c, aim[m]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int m = 0; m < PX; ++m)
        if (ix[m] < prm.nx && iy[m] < prm.ny) {
            const size_t o = (size_t)ix[m] * prm.ny + iy[m];
            out[o] = float2(out[o].x + are[m], out[o].y + aim[m]);
        }
}
'''

_lib = None
_ctx = {}


def _bridge():
    """The compiled bridge library (ctypes), built on first use."""
    global _lib
    if _lib is not None:
        return _lib
    if sys.platform != 'darwin':
        raise RuntimeError("backend 'metal' needs macOS on Apple silicon")
    from ._build import shared_object_mm
    L = ctypes.CDLL(shared_object_mm('fastsar_metal', _BRIDGE))
    vp, ci, cf = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    cp = ctypes.c_char_p
    f32 = np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')
    i32 = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
    L.fsm_available.argtypes = [cp, ci]; L.fsm_available.restype = ci
    L.fsm_init.argtypes = [cp, cp, ci, cp, ci]; L.fsm_init.restype = vp
    L.fsm_image_begin.argtypes = [vp, ctypes.c_long]; L.fsm_image_begin.restype = vp
    L.fsm_image_end.argtypes = [vp, f32]; L.fsm_image_end.restype = None
    L.fsm_bp_tiles.argtypes = [vp, vp, f32, ci, ci, f32, i32, ci, ci, ci, ci, ci, ci, cf, cf, f32, f32, cf, cf, ci, cp, ci]
    L.fsm_bp_tiles.restype = ci
    _lib = L
    return L


def available():
    """Whether a Metal device is present (and the bridge compiles)."""
    try:
        name = ctypes.create_string_buffer(256)
        return bool(_bridge().fsm_available(name, 256))
    except Exception:
        return False


def device_name():
    name = ctypes.create_string_buffer(256)
    if _bridge().fsm_available(name, 256):
        return name.value.decode()
    return None


def pipeline(TX, TY, cubic):
    """The compiled kernel for a tile size and interpolation: (ctx, TB)."""
    key = (TX, TY, bool(cubic))
    if key not in _ctx:
        L = _bridge()
        tb = min(128, TX * TY)
        px = (TX * TY) // tb
        if tb * px != TX * TY:
            raise ValueError(f'tile {TX}x{TY} is not a multiple of {tb} threads')
        src = _MSL.replace('__TB__', str(tb)).replace('__PX__', str(px)).replace('__CUBIC__', '1' if cubic else '0')
        err = ctypes.create_string_buffer(4096)
        ctx = L.fsm_init(src.encode(), b'bp_tiles', tb, err, 4096)
        if not ctx:
            raise RuntimeError(f'Metal: {err.value.decode()}')
        _ctx[key] = (ctx, tb)
    return _ctx[key]


def tile_terms(cen, ant, ref, lo, inv_dr, kcyc):
    """The tile centers' per-pulse terms in float64 -> (terms float32 [ntiles, n, 8], ti int32 [ntiles, n]):
    the unit vector to the antenna, 1/(2r), 1/(2r^2), the fractional sample position, the phase in cycles (the
    integer sample position separately)."""
    w = cen[:, None, :] - ant[None, :, :]                      # [ntiles, n, 3]
    r = np.sqrt((w * w).sum(-1))
    ir = 1.0 / r
    dR = r - ref[None, :]
    t = dR * inv_dr - lo[None, :]
    tf = np.floor(t)
    ph = kcyc * dR
    terms = np.empty(w.shape[:2] + (8,), np.float32)
    terms[..., 0] = w[..., 0] * ir; terms[..., 1] = w[..., 1] * ir; terms[..., 2] = w[..., 2] * ir
    terms[..., 3] = 0.5 * ir; terms[..., 4] = 0.5 * ir * ir
    terms[..., 5] = t - tf; terms[..., 6] = ph - np.rint(ph); terms[..., 7] = 0.0
    return terms, tf.astype(np.int32)
