"""The run-time independence of fastsar from sarpy and sarkit: in a fresh interpreter where neither can be imported,
a collection is simulated, written as a CPHD, read, processed end to end to a SICD and to a SIDD, and the files are
read back with fastsar's own readers. The files' validity against the two libraries is the business of the other
tests; this one proves that nothing at run time reaches for them."""
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent('''
    import importlib.abc
    import sys


    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] in ('sarpy', 'sarkit'):
                raise ImportError(f'{name} is blocked in this test')
            return None


    sys.meta_path.insert(0, Block())
    for name in ('sarpy', 'sarkit'):
        try:
            __import__(name)
        except ImportError:
            pass
        else:
            raise SystemExit(f'{name} could still be imported')

    import numpy as np
    import fastsar
    from fastsar import _nitf, _sicdxml, io, products, rgiqe, sidd, sim

    d = sys.argv[1]
    tg = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 15.0, 0.0], [-18.0, -22.0, 0.0]])
    col = sim.make_collect(res=0.5, scene=120.0, r0=10e3, graze_deg=35.0)
    cphd = f'{d}/c.cphd'
    sim.write_cphd(cphd, col, sim.simulate(col, tg, np.ones(len(tg))), 40.79, -77.86, 351.0, heading=30.0, scene=100.0)
    ecf = sim.to_ecf(tg, 40.79, -77.86, 351.0, 30.0)

    # CPHD to SICD, and the SICD read back: pixels, metadata, and the targets at its projection of them
    o = fastsar.process_cphd(cphd, f'{d}/a.nitf', product='sicd', weighting='taylor', backend='cpu')
    sm = o['metadata']
    f = _nitf.read(f'{d}/a.nitf')
    pix = f.pixels(native=True)
    back = _sicdxml.read(f'{d}/a.nitf')
    assert pix.shape == (back.ImageData.NumRows, back.ImageData.NumCols) and pix.dtype == np.complex64
    assert back.to_xml_bytes() == sm.to_xml_bytes()
    rc = back.project_ground_to_image(ecf)[0]
    for r, c in np.rint(rc).astype(int):
        w = np.abs(pix[r - 4:r + 5, c - 4:c + 5])
        assert np.unravel_index(np.argmax(w), w.shape) == (4, 4), (r, c)
    assert np.abs(back.project_image_to_ground(rc) - ecf).max() < 1e-3

    # the vendor-SICD inputs: the grid of read_cphd and form_cphd, sicd_points, write_sicd on a template
    c, meta = io.read_cphd(cphd, sicd=f'{d}/a.nitf', meta=True)
    assert c['nx'] * c['ny'] == pix.size
    pts = io.sicd_points(f'{d}/a.nitf', np.array([0, 5]), np.array([0, 7]), meta)
    assert pts.shape == (2, 3) and np.isfinite(pts).all()
    products.write_sicd(f'{d}/b.nitf', pix, f'{d}/a.nitf', transpose=False)
    assert np.array_equal(_nitf.read(f'{d}/b.nitf').pixels(native=True), pix)

    # CPHD to SIDD, and the radiometry model on the file
    o = fastsar.process_cphd(cphd, f'{d}/s.nitf', product='sidd', backend='cpu')
    g = _nitf.read(f'{d}/s.nitf')
    disp = g.pixels()
    assert disp.dtype == np.uint8 and disp.shape == o['image'].shape and len(g.xml('SIDD')) == 1
    assert o['metadata'].Display.PixelType == 'MONO8I'
    sidd.write_sidd(f'{d}/t.nitf', o, sicd=products.sicd_meta(o)[0], version=2)
    assert len(_nitf.read(f'{d}/t.nitf').xml('SICD')) == 1
    rad = rgiqe.Radiometry(cphd, meta, nesz_db=-20.0, sicd=f'{d}/a.nitf')
    assert rad.pulses_used > 2
    loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('sarpy', 'sarkit'))
    assert not loaded, loaded
    print('independent')
''')


def test_process_without_sarpy(tmp_path):
    script = tmp_path / 'run.py'
    script.write_text(SCRIPT)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''), CUDA_VISIBLE_DEVICES='',
               JAX_PLATFORMS='cpu')
    out = tmp_path / 'out'
    out.mkdir()
    r = subprocess.run([sys.executable, str(script), str(out)], capture_output=True, text=True, env=env, timeout=900,
                       cwd=str(tmp_path))
    assert r.returncode == 0 and 'independent' in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
