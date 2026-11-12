"""From a CPHD file to a product on disk, with the choices a production processor offers:

    fastsar.process_cphd('scene_CPHD.cphd', 'scene_SICD.nitf')                    # complex, slant plane, autofocus
    fastsar.process_cphd('scene_CPHD.cphd', 'scene_SIDD.nitf', product='sidd')    # detected, ground plane, SVA, NRL

Product. 'sicd': the complex image (SICD); 'sidd': a detected, remapped 8-bit image (SIDD), laid out shadows down.
Area. 'tasked': the image area of the file (or of the vendor SICD given). 'tasked_quality': everything at least as
good as the tasked area; a number: everything at or above that RNIIRS; 'black': everything with a signal-to-noise
ratio of at least 1 (fastsar.rgiqe: the radar generalized image quality equation on the collection's own antenna
pattern and noise). The impulse response is the same over the whole image, the noise-equivalent sigma zero rises
toward the edges, and pixels of the rectangular array outside the area are 0.
Plane. 'slant' or 'ground'. The image is always formed on the ground plane, where backprojection is exact for any
track, on axes along the line of sight; a slant-plane product is that image resampled (fastsar.planes), a
ground-plane SICD is the image as formed, and a ground-plane SIDD is detected on the slant plane and projected.
Autofocus. 'quadratic' (one quadratic phase error over the aperture, by map drift), 'pga', 'spatially_variant'
or 'spatially_variant_pga' (a phase error that varies over the scene), or False (fastsar.autofocus.image_autofocus).
Weighting. 'taylor' (nbar 4, 35 dB, on the phase history), 'uniform', or 'sva' (spatially variant apodization,
fastsar.weighting, on the slant plane at twice Nyquist: the unweighted resolution without the sidelobes).
Remap (SIDD). A name of fastsar.remap.REMAPS (the NGA MATLAB SAR Toolbox's: 'nrl', 'density', 'brighter', 'darker',
'highcontrast', 'pedf', 'log') or a function.

Defaults: the tasked area; a SICD on the slant plane with Taylor weighting and quadratic autofocus; for a SIDD the
ground plane, spatially variant apodization and the NRL remap.

A moving-beam collection (stripmap, sliding spotlight) is formed on the ground plane on axes along the track and
brought to baseband there (fastsar.moving). Weighting: 'taylor' is the Taylor window along frequency and over each
pixel's processed azimuth band, 'uniform' a rectangle over that band and no frequency window, 'sva' the uniform
image at twice Nyquist of both bands, cut to the same rectangle at every pixel and apodized on the ground grid
(refused where the squint or the swath leaves no rectangle). Autofocus: 'quadratic' or 'spatially_variant', by map
drift on blocks along the track, tied together by a smooth fit (the phase gradient modes are for one aperture).
Area: the tasked extent along the track, and in ground range as far as the criterion holds. Defaults: Taylor,
quadratic autofocus, the tasked area. The slant plane is for spotlight collections.
"""
import contextlib
import time

import numpy as np

from . import _xp, memory, planes, weighting as _weighting

__all__ = ['process_cphd']

_MARGIN = 24                  # ground pixels formed beyond the area on each side: the interpolation's edge


def _choice(name, value, allowed):
    if value not in allowed:
        raise ValueError(f'{name} must be one of {", ".join(repr(a) for a in allowed)}, got {value!r}')
    return value


def process_cphd(cphd, output=None, *, product='sicd', area='tasked', plane=None, autofocus=None, weighting=None,
                 remap='nrl', remap_args=None, spacing=None, extent=None, sicd=None, mode='auto', backend='auto',
                 precision='float32', shadows_down=True, nesz_db=None, scene_sigma0_db=-10.0, height=None, channel=0,
                 troposphere=None, reader_options=None, target_db=-40.0):
    """Form the image of a CPHD collection and write it as a SICD or a SIDD (see the module docstring).

    cphd: path. output: the file to write (NITF), or None to return the product without writing it.
    product: 'sicd' or 'sidd'. area: 'tasked', 'tasked_quality', an RNIIRS (number) or 'black'.
    plane: 'slant' or 'ground'; None: slant for a SICD, ground for a SIDD (and for any moving-beam collection).
    autofocus: 'quadratic', 'pga', 'spatially_variant', 'spatially_variant_pga' or False; None: 'quadratic'. A
        moving beam takes 'quadratic' and 'spatially_variant' (fastsar.moving.autofocus).
    weighting: 'taylor', 'uniform' or 'sva'; None: 'taylor' for a SICD, 'sva' for a SIDD ('taylor' for a moving beam).
    remap, remap_args: the SIDD's remap and its parameters (fastsar.sidd.to_display). shadows_down: lay a SIDD out
        with range increasing down the rows (False: the grid's own layout).
    spacing: the product's pixel spacing in meters: (azimuth, range) on the slant plane, (across, along the line of
        sight) for a ground-plane SICD, one number (square pixels) for a ground-plane SIDD, (along, across) the
        track for a moving beam. None: 0.96 of the
        reciprocal of the bandwidth for a complex product (half of it with 'sva'), 0.7 of the finer ground
        resolution for a SIDD. Without a spacing or a vendor sicd, a spotlight uses the central part of the
        aperture that makes the azimuth resolution the ground range resolution (form_cphd's aperture='auto').
    extent: the tasked area's sides in meters about the scene center, for a collection that states no image area
        (and in place of the one it states): (across, along) the line of sight for a spotlight, (along, across)
        the track for a moving beam.
    sicd: the vendor's SICD (path, .xml or SICD metadata), whose footprint is then the tasked area.
    nesz_db, scene_sigma0_db: the noise level of the quality areas when the files give none (fastsar.rgiqe).
    mode, backend, precision, height, channel, troposphere, reader_options, target_db: form_cphd's.

    -> the product dict (form_cphd's entries on the product's grid: image, origin, e1, e2, spx, spy, plane, meta,
    notes, ...), with product, area, weighting, autofocus (the mode run), autofocus_info, timing (seconds of each
    stage), and, when written, path and metadata (fastsar._sicdxml.SICD or fastsar._siddxml.SIDD). The image of a SIDD product is
    the detected amplitude (float32) on the product's own grid; the remap and the layout are applied when the file
    is written, and fastsar.sidd.to_display(out, out['remap'], **out['remap_args']) gives the same display values."""
    from .api import _backend
    from .cphd import form_cphd
    backend = _backend(backend)
    _choice('product', product, ('sicd', 'sidd'))
    _choice('plane', plane, (None, 'slant', 'ground'))
    _choice('weighting', weighting, (None, 'taylor', 'uniform', 'sva'))
    if autofocus is True:
        autofocus = 'quadratic'
    _choice('autofocus', autofocus, (None, False, 'quadratic', 'pga', 'spatially_variant', 'spatially_variant_pga'))
    if not (isinstance(area, str) and area in ('tasked', 'tasked_quality', 'black')) and not (
            isinstance(area, (int, float, np.integer, np.floating)) and not isinstance(area, bool)):
        raise ValueError(f"area must be 'tasked', 'tasked_quality', 'black' or an RNIIRS, got {area!r}")
    if extent is not None and (np.size(extent) != 2 or np.min(extent) <= 0):
        raise ValueError(f'extent must be two positive lengths in meters, got {extent!r}')
    if spacing is not None and (np.size(spacing) not in (1, 2) or np.min(spacing) <= 0):
        raise ValueError(f'spacing must be one or two positive lengths in meters, got {spacing!r}')
    if product == 'sidd' and spacing is not None and np.size(spacing) != 1:
        raise ValueError('a SIDD has square pixels: give spacing as one number')
    if product == 'sidd':
        from . import remap as _remap
        if not callable(remap) and (remap not in _remap.REMAPS or remap == 'linear'):
            raise ValueError(f"remap {remap!r}: one of {sorted(k for k in _remap.REMAPS if k != 'linear')}, or a function")
    if weighting == 'sva' and product == 'sicd' and spacing is not None:
        raise ValueError("weighting='sva' sets the slant spacing itself (twice Nyquist): leave spacing=None")
    timing, st = {}, {}
    t0 = time.perf_counter()
    st['start'] = t0

    cuda, jaxdev = backend == 'cuda', backend in ('jax', 'tpu')

    def lap(name, image=None):
        nonlocal t0
        if cuda:                                                     # the stage's kernels have run
            import cupy
            cupy.cuda.Stream.null.synchronize()
        elif _xp.kind(image) == 'jax':                               # and the stage's programs, whose result this is
            image.block_until_ready()
        t1 = time.perf_counter()
        timing[name] = timing.get(name, 0.0) + t1 - t0
        t0 = t1

    def plan(g):
        """The formation grid, from the geometry form_cphd read: the area's bounds and the sampling its spectrum asks."""
        st['read'] = time.perf_counter()
        spot = g['mode'] == 'spotlight'
        pl = plane or ('slant' if product == 'sicd' and spot else 'ground')
        wt = weighting or ('sva' if product == 'sidd' and spot else 'taylor')
        af = ('quadratic' if spot else False) if autofocus is None else autofocus
        if not spot:
            if pl != 'ground':
                raise ValueError(f'plane={pl!r} is for spotlight collections; this one has a moving beam')
            from . import moving
            return moving.plan(g, st, dict(
                product=product, weighting=wt, autofocus='quadratic' if autofocus is None else autofocus, area=area,
                spacing=spacing, extent=extent, cphd=cphd, channel=channel, nesz_db=nesz_db,
                scene_sigma0_db=scene_sigma0_db))
        if product == 'sicd' and pl == 'ground' and wt == 'sva':
            raise ValueError("weighting='sva' is applied on the slant plane: for a SICD give plane='slant'")
        if product == 'sidd' and pl == 'slant':
            raise ValueError("a SIDD is written on the ground plane: plane='ground'")
        K = g['S'].shape[1]
        band = (g['fmin'], g['fmin'] + (K - 1) * g['df'])
        c, e1, e2 = g['center'], g['e1'], g['e2']
        fr = planes.slant_frame(g['meta'], c, g['pulses'])
        if extent is not None:
            b = np.array([-extent[0], extent[0], -extent[1], extent[1]], np.float64) / 2
        elif g['corners'] is None:
            raise ValueError('the collection gives no image area: give an extent or the vendor sicd')
        else:
            d = g['corners'] - c
            b = np.array([(d @ e1).min(), (d @ e1).max(), (d @ e2).min(), (d @ e2).max()])
        kept = None
        if wt == 'sva':
            lost, kept = _weighting.keystone(g['S'], g['ant'], g['fmin'], g['df'], c, fr['e1'], symmetric=False)
            g['notes'].append(f'keystone trim: {100 * lost:.1f}% of the samples dropped for a rectangular spectrum')
        valid = None
        if area != 'tasked':
            from . import rgiqe
            sup = planes.support(fr, g['ant'], band, *planes._slant_axes(fr, c[2]))
            bw_area = (sup['k1_mid'][1] - sup['k1_mid'][0]) * (sup['k2'][1] - sup['k2'][0]) * abs(fr['normal'][2])
            rad = rgiqe.Radiometry(cphd, g['meta'], channel=channel, pulses=g['pulses'], nesz_db=nesz_db,
                                   sicd=g['sicd'], scene_sigma0_db=scene_sigma0_db)
            if not rad.has_antenna:
                raise ValueError(f'area={area!r}: the file describes no antenna pattern, so the signal has no edge '
                                 "to follow; only area='tasked' can be processed")
            tasked = c + np.stack([b[[0, 1, 1, 0]], b[[2, 2, 3, 3]]], 1) @ np.stack([e1, e2])
            n = g['S'].shape[0] * K
            b = np.array(rgiqe.area_bounds(rad, area, c, e1, e2, bw_area, n, tasked=tasked))
            valid = (rad, bw_area, n, tasked)
            g['notes'].extend(rad.notes)
        t1, t2 = np.linspace(b[0], b[1], 5), np.linspace(b[2], b[3], 5)
        s = planes.sampling(fr, g['ant'], band, c + (t1[:, None, None] * e1 + t2[None, :, None] * e2).reshape(-1, 3),
                            e1, e2)
        via_slant = pl == 'slant' or product == 'sidd'
        sp = (0.96 / s['bx'], 0.96 / s['by'])
        if not via_slant:                                            # a ground-plane SICD is the image as formed: its
            need = (1 / s['bx'], 1 / s['by_full'])                   # spectrum must not wrap along the line of sight
            sp = (0.96 * need[0], 0.96 * need[1])
            if spacing is not None:
                sp = tuple(float(v) for v in np.broadcast_to(np.asarray(spacing, np.float64), (2,)))
                if sp[0] > need[0] * 1.0001 or sp[1] > need[1] * 1.0001:
                    g['notes'].append(f'spacing {sp[0]:.3f} x {sp[1]:.3f} m is coarser than the band of the data allows '
                                      f'({need[0]:.3f} x {need[1]:.3f} m): the image is aliased')
        m = _MARGIN if via_slant else 0
        st.update(spot=True, plane=pl, weighting=wt, autofocus=af, sampling=s, bounds=b, kept=kept, valid=valid,
                  via_slant=via_slant, margin=m)
        return dict(spacing=sp, bounds=(b[0] - m * sp[0], b[1] + m * sp[0], b[2] - m * sp[1], b[3] + m * sp[1]),
                    window=wt == 'taylor')

    out = form_cphd(cphd, sicd, mode=mode, backend=backend, channel=channel, height=height, precision=precision,
                    target_db=target_db, troposphere=troposphere, reader_options=reader_options, axes='auto',
                    extent=extent,
                    aperture='auto' if spacing is None else 'full', _plan=plan)
    lap('form')
    timing['read'] = st['read'] - st['start']                        # the file, to the phase history in memory
    timing['form'] -= timing['read']
    if not cuda and not jaxdev:                                      # the stages that follow run where it lives;
        out['image'] = _xp.device(out['image'], backend)             # on a device, where its plan finds room
    out.update(product=product, area=area, weighting=st['weighting'], autofocus=None, autofocus_info=None)
    tight = False
    try:
        if st['spot']:
            out, tight = _spotlight(out, st, product, area, spacing, lap, cuda, jaxdev)
        else:
            from . import moving
            out, tight = moving.stages(out, st, product, lap, backend)
        out['timing'] = timing
        if product == 'sidd':
            out.update(remap=remap, remap_args=dict(remap_args or {}), shadows_down=bool(shadows_down))
        if output is not None and product == 'sidd':                 # remapped where the image lives
            from .sidd import write_sidd
            from .remap import _via_jax
            # an image the JAX device could not hold is on the host: its remap crosses the device in blocks
            with _via_jax() if jaxdev and _xp.kind(out['image']) == 'numpy' else contextlib.nullcontext():
                out['metadata'] = write_sidd(output, out, remap=remap, remap_args=remap_args, shadows_down=shadows_down)
        out['image'] = _xp.host(out['image'])
    except Exception as e:
        if not cuda or type(e).__name__ != 'OutOfMemoryError':
            raise
        raise MemoryError(f'the GPU ran out of memory after image formation ({memory._gb(memory.cuda_free())} free '
                          f'now): {e}. FASTSAR_CUDA_BLOCK_MB sets the device memory these stages count on') from None
    if tight:
        memory.cuda_release()
    if tight == 'host':                                              # the page-locked blocks the images crossed through
        import cupy
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    if output is not None and product == 'sicd':
        from .products import write_sicd
        out['metadata'] = write_sicd(output, out)
    if output is not None:
        out['path'] = output
        lap('write')
    return out


def _mask(out, st, area):
    """The ground pixels [nx, ny] inside a quality area: the model on a coarse lattice, bilinear between its nodes."""
    from . import rgiqe
    from .patches import _rows
    rad, bw_area, n, tasked = st['valid']
    nx, ny, o, e1, e2, spx, spy = planes._grid(out)
    # nodes every 50 m, and at least 48 across the image
    ki = planes._knots(nx, max(4, int(min(50.0 / spx, nx / 48))))
    kj = planes._knots(ny, max(4, int(min(50.0 / spy, ny / 48))))
    pts = o + (ki * spx)[:, None, None] * e1 + (kj * spy)[None, :, None] * e2
    m, info = rgiqe.area_mask(rad, pts, area, bw_area, n, tasked=tasked)
    # the margin over the threshold where the model gives it (a smooth boundary between the nodes), else the mask
    f = np.asarray(info['level'], np.float64) if info.get('level') is not None else np.asarray(m, np.float64) - 0.5
    from .resample import KnotMap
    km = KnotMap(ki, kj, f, ny)
    valid = np.empty((nx, ny), bool)

    def work(sl):
        valid[sl] = km(sl) >= 0
    _rows(work, nx, 256)
    return valid


def _cuda_place(out, sizes, **stages):
    """Where the images of the product live on a CUDA device (memory.cuda_plan on the memory free now) -> (place,
    tight: False, True when the plan takes more than half of the device, so that the stages return their cached
    blocks as they go, or 'host' when some image stays on the host). A note says so when the device cannot hold
    them all."""
    plan = memory.cuda_plan(sizes, _xp.cuda_budget(), **stages)
    tight = 'host' if not plan['fits'] else plan['need'] > 0.5 * plan['budget']
    if tight:
        memory.cuda_release()                                        # the former's cached blocks and FFT plans
    if not plan['fits']:
        names = dict(ground='the image as formed', mid='the intermediate image of the slant resampling',
                     slant='the slant image', detected='the detected slant image',
                     ground_mid='the intermediate image of the ground projection', product='the product image')
        held = [names[k] for k, v in plan['place'].items() if v == 'host']
        af = stages.get('autofocus') and plan['place']['ground'] == 'host'
        out['notes'].append(
            f"the GPU has {plan['budget'] / 1e9:.1f} GB for the stages after formation, which take "
            f"{plan['need'] / 1e9:.1f} GB with every image of this product on it: kept on the host and passed "
            "through the GPU in blocks: "
            + ', '.join(held) + ('; autofocus ran on the host' if af else ''))
    return plan['place'], tight


def _jax_place(out, sizes, **stages):
    """Where the images of the product live while the stages after formation run on JAX's device (memory.cuda_plan
    with the JAX path's accounting, on _xp.jax_budget()) -> place. A device that does not report its memory (JAX on
    a CPU) holds them all. A note says so when the device cannot."""
    budget = _xp.jax_budget()
    if budget is None:
        return dict.fromkeys(('ground', 'mid', 'slant', 'detected', 'ground_mid', 'product'), 'device')
    plan = memory.cuda_plan(sizes, budget, joined=True, **stages)
    if not plan['fits']:
        names = dict(ground='the image as formed', mid='the intermediate image of the slant resampling',
                     slant='the slant image', detected='the detected slant image',
                     ground_mid='the intermediate image of the ground projection', product='the product image')
        out['notes'].append(
            f"the JAX device has {plan['budget'] / 1e9:.1f} GB for the stages after formation, which take "
            f"{plan['need'] / 1e9:.1f} GB with every image of this product on it: kept on the host and passed "
            'through the device in blocks: ' + ', '.join(names[k] for k, v in plan['place'].items() if v == 'host'))
    return plan['place']


def _detect_jax(img, where):
    """The amplitude (float32) of a complex image on the host or on JAX's device, as an array on the device
    (where='device') or the host, through the device in blocks of rows unless both are there."""
    import jax
    import jax.numpy as jnp
    from . import resample_jax as rj
    if _xp.kind(img) == 'jax' and where == 'device':
        return jnp.abs(img)
    n0, n1 = img.shape
    nb = _xp.jax_rows(n0, 24 * n1, 'detection on the JAX device')
    if _detect_jax not in rj._JIT:
        rj._JIT[_detect_jax] = jax.jit(jnp.abs)
    lines, out = rj._source(img), rj._Sink(where, n0, n1, nb, 1, np.float32)
    for a in rj._starts(n0, nb):
        out.put(a, rj._JIT[_detect_jax](lines(a, nb)))
    return out.done()


def _detect(img, where):
    """The amplitude (float32) of a complex image on the host or a CUDA device, as an array on the device
    (where='device') or the host, through the device in blocks of rows unless both are there."""
    import cupy as cp
    if isinstance(img, cp.ndarray) and where == 'device':
        return cp.abs(img)
    n0, n1 = img.shape
    from .resample_cuda import host_empty, upload
    out = cp.empty((n0, n1), np.float32) if where == 'device' else host_empty((n0, n1), np.float32)
    nb = _xp.cuda_rows(n0, 12 * n1, 'detection on the GPU')
    for a in range(0, n0, nb):
        blk = img[a:a + nb]
        r = cp.abs(blk if isinstance(blk, cp.ndarray) else upload(np.ascontiguousarray(blk)))
        if where == 'device':
            out[a:a + nb] = r
        else:
            r.get(out=out[a:a + nb])
    return out


def _spotlight(out, st, product, area, spacing, lap, cuda, jaxdev=False):
    s, wt, mg = st['sampling'], st['weighting'], st['margin']
    out['window'] = (wt, wt)
    nx, ny = out['image'].shape
    # the grids of the product, before any of its images is made
    if st['via_slant']:
        if wt == 'sva':                                              # twice Nyquist of the rectangle keystone left
            sup = planes.support(planes._frame(out), planes._used(out), out['band'],
                                 *planes._slant_axes(planes._frame(out), out['height']))
            sp = (0.5 / (st['kept'][1] - st['kept'][0]), 0.5 / (sup['k2'][1] - sup['k2'][0]))
        elif product == 'sidd':                                      # detected: the band doubles
            sp = (0.5 / s['ba'], 0.5 / s['br'])
        else:
            sp = (0.96 / s['ba'], 0.96 / s['br'])
        if product == 'sicd' and spacing is not None:
            sp = tuple(float(v) for v in np.broadcast_to(np.asarray(spacing, np.float64), (2,)))
        res = 0.886 / max(s['bx'], s['by'])
        px = 0.7 * res if spacing is None or product == 'sicd' else float(np.ravel(spacing)[0])
        b = st['bounds']
        shape = (int(np.ceil((b[1] - b[0]) / px)) + 1, int(np.ceil((b[3] - b[2]) / px)) + 1)
    place, tight, release = None, False, lambda: None
    if cuda or jaxdev:
        sizes = dict(ground=8 * nx * ny, valid=0 if st['valid'] is None else nx * ny)
        if st['via_slant']:
            na, nr = planes._slant_shape(sp, planes._slant_bounds(out, mg))[1]
            sizes.update(mid=8 * na * ny, slant=8 * na * nr)
            if product == 'sidd':
                sizes.update(detected=4 * na * nr, ground_mid=4 * na * shape[1], product=4 * shape[0] * shape[1],
                             display=shape[0] * shape[1])
    if jaxdev:
        place = _jax_place(out, sizes, autofocus=bool(st['autofocus']), sva=wt == 'sva')
        if place['ground'] == 'device':
            out['image'] = _xp.device(out['image'], 'jax')
    elif cuda:
        place, tight = _cuda_place(out, sizes, autofocus=bool(st['autofocus']), sva=wt == 'sva')
        if tight:
            release = memory.cuda_release
        if place['ground'] == 'device':
            out['image'] = _xp.device(out['image'], 'cuda')
    planes.deramp(out, _cuda=cuda, _jax=jaxdev)
    lap('baseband', out['image'])
    if st['autofocus']:
        from .autofocus import image_autofocus
        try:
            try:
                from .autofocus import _via_jax
                # an image on the host with a JAX backend: its global estimate and correction cross the device
                with _via_jax() if jaxdev and _xp.kind(out['image']) == 'numpy' else contextlib.nullcontext():
                    out['image'], info = image_autofocus(out['image'], st['autofocus'])
            except Exception as e:                                   # the device held the image, not the correction
                if _xp.kind(out['image']) != 'cupy' or type(e).__name__ != 'OutOfMemoryError':
                    raise
                img = _xp.host(out['image'])
                out['image'] = None
                memory.cuda_release()
                img, info = image_autofocus(img, st['autofocus'])
                out['image'] = _xp.device(img, 'cuda')
                del img
                out['notes'].append('autofocus ran on the host: the GPU had no memory for its correction')
        except ValueError as e:                                      # an image too small for the estimate
            info = dict(mode=st['autofocus'], applied=False, reason=str(e))
            out['notes'].append(f'autofocus skipped: {e}')
        out.update(autofocus=st['autofocus'] if info.get('applied', True) else None, autofocus_info=info)
        release()
        lap('autofocus', out['image'])
    valid = None if st['valid'] is None else _mask(out, st, area)
    if not st['via_slant']:                                          # a ground-plane SICD: the image as formed
        if valid is not None and _xp.kind(out['image']) == 'cupy':
            import cupy
            cupy.putmask(out['image'], cupy.asarray(~valid), 0)
        elif valid is not None:
            xp = _xp.module(out['image'])
            out['image'] = xp.where(xp.asarray(valid), out['image'], 0)
        out['bandwidth'] = (s['bx'], s['by_full'])
        lap('mask', out['image'])
        return out, tight
    if product == 'sicd' and spacing is not None and (sp[0] > 1.0001 / s['ba'] or sp[1] > 1.0001 / s['br']):
        out['notes'].append(f"spacing {sp[0]:.3f} x {sp[1]:.3f} m is coarser than the band of the data allows "
                            f"({1 / s['ba']:.3f} x {1 / s['br']:.3f} m): the image is aliased")
    ground = {k: out[k] for k in ('origin', 'e1', 'e2', 'spx', 'spy')}
    sl = planes.to_slant(out, sp, margin=mg, valid=valid,
                         **({'_cuda' if cuda else '_jax': dict(mid=place['mid'], out=place['slant'])} if place else {}))
    out['image'] = None                                              # the caller's dict too: the ground image is freed
    del out, valid
    release()
    lap('to_slant', sl['image'])
    if wt == 'sva':
        # apodization takes the band centered: where the aperture is not symmetric about the center of aperture,
        # the azimuth band is moved to zero for it and back
        k0, ramp = 0.5 * (st['kept'][0] + st['kept'][1]), None
        if abs(k0) > 0.002 * (st['kept'][1] - st['kept'][0]):
            a = (np.arange(sl['image'].shape[0]) - sl['scp_pixel'][0]) * sl['spx']
            ramp = np.exp(-2j * np.pi * k0 * a).astype(np.complex64)[:, None]
        if jaxdev:                                                   # the ramps with the image's blocks, as below
            img, sl['image'] = [sl['image']], None                   # given up: a device image is freed as its slabs go
            sl['image'] = _weighting.sva(img, 2, _jax=(
                ramp, ramp.conj() if ramp is not None and product == 'sicd' else None, place['slant']))
        elif cuda:                                                   # the ramps on the device, with the image's blocks
            import cupy
            if ramp is not None:
                ramp = (cupy.asarray(ramp), cupy.asarray(ramp.conj()) if product == 'sicd' else None)
            sl['image'] = _weighting.sva(sl['image'], 2, inplace=True, _cuda=(ramp,))
        else:
            if ramp is not None:
                ramp = _xp.module(sl['image']).asarray(ramp)
                sl['image'] = sl['image'] * ramp
            sl['image'] = _weighting.sva(sl['image'], 2, inplace=True)
            if ramp is not None and product == 'sicd':
                sl['image'] = sl['image'] * ramp.conj()
        release()
        lap('sva', sl['image'])
    if product == 'sicd':
        return sl, tight
    # a SIDD: detected on the slant plane, projected to square ground pixels on axes along the line of sight
    if jaxdev:
        sl['image'] = _detect_jax(sl['image'], place['detected'])
    else:
        sl['image'] = _detect(sl['image'], place['detected']) if cuda else _xp.module(sl['image']).abs(sl['image'])
    release()
    lap('detect', sl['image'])
    if px > res:
        sl['notes'].append(f'pixel {px:.3f} m is coarser than the resolution ({res:.3f} m): the detected image is not '
                           'filtered before it is resampled, and speckle aliases')
    o = ground['origin'] + mg * ground['spx'] * ground['e1'] + mg * ground['spy'] * ground['e2']
    g = planes.to_ground(sl, o, ground['e1'], ground['e2'], px, shape, **(
        {'_cuda' if cuda else '_jax': dict(mid=place['ground_mid'], out=place['product'])} if place else {}))
    del sl
    if _xp.kind(g['image']) == 'jax':
        g['image'] = _xp.module(g['image']).maximum(g['image'], 0)
    else:
        _xp.module(g['image']).maximum(g['image'], 0, out=g['image'])
    g['bandwidth'] = (s['bx'], s['by'])
    release()
    lap('to_ground', g['image'])
    return g, tight
