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
(a phase error that varies over the scene), or False (fastsar.autofocus.image_autofocus).
Weighting. 'taylor' (nbar 4, 35 dB, on the phase history), 'uniform', or 'sva' (spatially variant apodization,
fastsar.weighting, on the slant plane at twice Nyquist: the unweighted resolution without the sidelobes).
Remap (SIDD). A name of fastsar.remap.REMAPS (the NGA MATLAB SAR Toolbox's: 'nrl', 'density', 'brighter', 'darker',
'highcontrast', 'pedf', 'log') or a function.

Defaults: the tasked area; a SICD on the slant plane with Taylor weighting and quadratic autofocus; for a SIDD the
ground plane, spatially variant apodization and the NRL remap. A moving-beam collection (stripmap, sliding
spotlight) is formed on the ground plane on axes along the track, without autofocus or apodization.
"""
import time

import numpy as np

from . import _xp, planes, weighting as _weighting

__all__ = ['process_cphd']

_MARGIN = 24                  # ground pixels formed beyond the area on each side: the interpolation's edge


def _choice(name, value, allowed):
    if value not in allowed:
        raise ValueError(f'{name} must be one of {", ".join(repr(a) for a in allowed)}, got {value!r}')
    return value


def process_cphd(cphd, output=None, *, product='sicd', area='tasked', plane=None, autofocus=None, weighting=None,
                 remap='nrl', remap_args=None, spacing=None, sicd=None, mode='auto', backend='auto',
                 precision='float32', shadows_down=True, nesz_db=None, scene_sigma0_db=-10.0, height=None, channel=0,
                 troposphere=None, reader_options=None, target_db=-40.0):
    """Form the image of a CPHD collection and write it as a SICD or a SIDD (see the module docstring).

    cphd: path. output: the file to write (NITF), or None to return the product without writing it.
    product: 'sicd' or 'sidd'. area: 'tasked', 'tasked_quality', an RNIIRS (number) or 'black'.
    plane: 'slant' or 'ground'; None: slant for a SICD, ground for a SIDD (and for any moving-beam collection).
    autofocus: 'quadratic', 'pga', 'spatially_variant', 'spatially_variant_pga' or False; None: 'quadratic' for a
        spotlight, none for a moving beam.
    weighting: 'taylor', 'uniform' or 'sva'; None: 'taylor' for a SICD, 'sva' for a SIDD ('taylor' for a moving beam).
    remap, remap_args: the SIDD's remap and its parameters (fastsar.sidd.to_display). shadows_down: lay a SIDD out
        with range increasing down the rows (False: the grid's own layout).
    spacing: the product's pixel spacing in meters: (azimuth, range) on the slant plane, (across, along the line of
        sight) for a ground-plane SICD, one number (square pixels) for a ground-plane SIDD. None: 0.96 of the
        reciprocal of the bandwidth for a complex product (half of it with 'sva'), 0.7 of the finer ground
        resolution for a SIDD. Without a spacing or a vendor sicd, a spotlight uses the central part of the
        aperture that makes the azimuth resolution the ground range resolution (form_cphd's aperture='auto').
    sicd: the vendor's SICD (path, .xml or sarpy SICDType), whose footprint is then the tasked area.
    nesz_db, scene_sigma0_db: the noise level of the quality areas when the files give none (fastsar.rgiqe).
    mode, backend, precision, height, channel, troposphere, reader_options, target_db: form_cphd's.

    -> the product dict (form_cphd's entries on the product's grid: image, origin, e1, e2, spx, spy, plane, meta,
    notes, ...), with product, area, weighting, autofocus (the mode run), autofocus_info, timing (seconds of each
    stage), and, when written, path and metadata (the sarpy SICDType or SIDDType)."""
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
    if spacing is not None and (np.size(spacing) not in (1, 2) or np.min(spacing) <= 0):
        raise ValueError(f'spacing must be one or two positive lengths in meters, got {spacing!r}')
    timing, st = {}, {}
    t0 = time.perf_counter()
    st['start'] = t0

    def lap(name):
        nonlocal t0
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
            for name, v, ok in (('plane', pl, pl == 'ground'), ('weighting', wt, wt != 'sva'), ('autofocus', af, not af),
                                ('area', area, area == 'tasked')):
                if not ok:
                    raise ValueError(f'{name}={v!r} is for spotlight collections; this one has a moving beam')
            st.update(spot=False, plane=pl, weighting=wt, autofocus=af)
            return dict(window=wt == 'taylor')
        if product == 'sicd' and pl == 'ground' and wt == 'sva':
            raise ValueError("weighting='sva' is applied on the slant plane: for a SICD give plane='slant'")
        if product == 'sidd' and pl == 'slant':
            raise ValueError("a SIDD is written on the ground plane: plane='ground'")
        K = g['S'].shape[1]
        band = (g['fmin'], g['fmin'] + (K - 1) * g['df'])
        c, e1, e2 = g['center'], g['e1'], g['e2']
        fr = planes.slant_frame(g['meta'], c, g['pulses'])
        if g['corners'] is None:
            raise ValueError('the collection gives no image area: give the vendor sicd')
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
        if not via_slant and spacing is not None:                    # a ground-plane SICD is the image as formed
            sp = tuple(float(v) for v in np.broadcast_to(np.asarray(spacing, np.float64), (2,)))
        m = _MARGIN if via_slant else 0
        st.update(spot=True, plane=pl, weighting=wt, autofocus=af, sampling=s, bounds=b, kept=kept, valid=valid,
                  via_slant=via_slant, margin=m)
        return dict(spacing=sp, bounds=(b[0] - m * sp[0], b[1] + m * sp[0], b[2] - m * sp[1], b[3] + m * sp[1]),
                    window=wt == 'taylor')

    out = form_cphd(cphd, sicd, mode=mode, backend=backend, channel=channel, height=height, precision=precision,
                    target_db=target_db, troposphere=troposphere, reader_options=reader_options, axes='auto',
                    aperture='auto' if spacing is None else 'full', _plan=plan)
    lap('form')
    timing['read'] = st['read'] - st['start']                        # the file, to the phase history in memory
    timing['form'] -= timing['read']
    out['image'] = _xp.device(out['image'], backend)                 # the stages that follow run where it lives
    out.update(product=product, area=area, weighting=st['weighting'], autofocus=None, autofocus_info=None)
    if st['spot']:
        out = _spotlight(out, st, product, area, spacing, lap)
    out['timing'] = timing
    if output is not None and product == 'sidd':                     # remapped where the image lives
        from .sidd import write_sidd
        out['metadata'] = write_sidd(output, out, remap=remap, remap_args=remap_args, shadows_down=shadows_down)
    out['image'] = _xp.host(out['image'])
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
    ki, kj = planes._knots(nx, max(8, int(50.0 / spx))), planes._knots(ny, max(8, int(50.0 / spy)))
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


def _spotlight(out, st, product, area, spacing, lap):
    s, wt, mg = st['sampling'], st['weighting'], st['margin']
    out['window'] = (wt, wt)
    planes.deramp(out)
    lap('baseband')
    if st['autofocus']:
        from .autofocus import image_autofocus
        out['image'], info = image_autofocus(out['image'], st['autofocus'])
        out.update(autofocus=st['autofocus'] if info.get('applied', True) else None, autofocus_info=info)
        lap('autofocus')
    valid = None if st['valid'] is None else _mask(out, st, area)
    if not st['via_slant']:                                          # a ground-plane SICD: the image as formed
        if valid is not None:
            xp = _xp.module(out['image'])
            out['image'] = xp.where(xp.asarray(valid), out['image'], 0)
        out['bandwidth'] = (s['bx'], s['by'])
        lap('mask')
        return out
    if wt == 'sva':                                                  # twice Nyquist of the rectangle keystone left
        sup = planes.support(planes._frame(out), planes._used(out), out['band'],
                             *planes._slant_axes(planes._frame(out), out['height']))
        sp = (0.5 / (st['kept'][1] - st['kept'][0]), 0.5 / (sup['k2'][1] - sup['k2'][0]))
    else:
        sp = (0.96 / s['ba'], 0.96 / s['br'])
    if product == 'sicd' and spacing is not None:
        if wt == 'sva':
            raise ValueError("weighting='sva' sets the slant spacing itself (twice Nyquist): leave spacing=None")
        sp = tuple(float(v) for v in np.broadcast_to(np.asarray(spacing, np.float64), (2,)))
    ground = {k: out[k] for k in ('origin', 'e1', 'e2', 'spx', 'spy')}
    sl = planes.to_slant(out, sp, margin=mg, valid=valid)
    del out, valid
    lap('to_slant')
    if wt == 'sva':
        # apodization takes the band centered: where the aperture is not symmetric about the center of aperture,
        # the azimuth band is moved to zero for it and back
        k0, ramp = 0.5 * (st['kept'][0] + st['kept'][1]), None
        if abs(k0) > 0.002 * (st['kept'][1] - st['kept'][0]):
            a = (np.arange(sl['image'].shape[0]) - sl['scp_pixel'][0]) * sl['spx']
            ramp = _xp.module(sl['image']).asarray(np.exp(-2j * np.pi * k0 * a).astype(np.complex64)[:, None])
            sl['image'] = sl['image'] * ramp
        sl['image'] = _weighting.sva(sl['image'], 2, inplace=True)
        if ramp is not None and product == 'sicd':
            sl['image'] = sl['image'] * ramp.conj()
        lap('sva')
    if product == 'sicd':
        return sl
    # a SIDD: detected on the slant plane, projected to square ground pixels on axes along the line of sight
    sl['image'] = _xp.module(sl['image']).abs(sl['image'])
    lap('detect')
    res = 0.886 / max(s['bx'], s['by'])
    px = float(spacing) if spacing is not None and np.size(spacing) == 1 else (0.7 * res if spacing is None else None)
    if px is None:
        raise ValueError('a ground-plane SIDD has square pixels: give spacing as one number')
    b = st['bounds']
    shape = (int(np.ceil((b[1] - b[0]) / px)) + 1, int(np.ceil((b[3] - b[2]) / px)) + 1)
    o = ground['origin'] + mg * ground['spx'] * ground['e1'] + mg * ground['spy'] * ground['e2']
    g = planes.to_ground(sl, o, ground['e1'], ground['e2'], px, shape)
    g['image'] = _xp.module(g['image']).maximum(g['image'], 0)
    g['bandwidth'] = (s['bx'], s['by'])
    lap('to_ground')
    return g
