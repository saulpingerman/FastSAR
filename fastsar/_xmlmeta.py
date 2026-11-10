"""A light attribute-style view of an XML metadata tree (SICD, SIDD, CPHD), in place of one class per element.

    root = parse(xml_bytes, spec)            # Node over the root element
    root.Grid.Row.SS                         # a leaf: float, int, bool or str
    root.GeoData.SCP.ECF.get_array()         # [X, Y, Z] as float64
    root.Grid.TimeCOAPoly.get_array()        # polynomial coefficients [order1 + 1, order2 + 1]
    root.Grid.TimeCOAPoly[0, 0]
    root.Nope                                # None: a missing element reads as None
    root.Grid.Type = 'PLANE'; root.PFA = None    # set a leaf's text; None removes the element
    root.to_xml_bytes()

Conventions. Element names are the XML's own local names (namespaces are dropped on parsing and the root's is
restored on writing). A Spec (one per format) names what the tree alone cannot tell: which leaves are text whatever
they look like, which are integers, which children repeat (and so read as lists), and aliases, the plural names
under which a list is also reached (Data.Channels for the Channel elements of a CPHD's Data block). Leaves not named
in the Spec read as bool ('true', 'false'), else float where the text parses as one, else str."""
import copy as _copy
import xml.etree.ElementTree as ET

import numpy as np


class Spec:
    """text, ints: sets of element names read as str and int. lists: {(parent, name)} or {name} of children that
    repeat (read as a list of all of them, possibly empty -> None). aliases: {(parent, alias): (path, tag, kind)}:
    parent.alias reads the `tag` children found under the child path `path` ('' for parent itself) as kind 'list'
    (list of Nodes or leaf values), 'dict' (name attribute -> text) or 'points' (float64 array of the children's
    get_array, ordered by their index attribute)."""

    def __init__(self, text=(), ints=(), lists=(), aliases=None):
        self.text, self.ints, self.lists, self.aliases = frozenset(text), frozenset(ints), frozenset(lists), aliases or {}


_PLAIN = Spec()
_VECTORS = (('X', 'Y', 'Z'), ('Lat', 'Lon', 'HAE'), ('Lat', 'Lon'), ('Row', 'Col'), ('Line', 'Sample'), ('X', 'Y'),
            ('DCX', 'DCY'), ('Range', 'Azimuth'))


def _local(tag):
    return tag.rsplit('}', 1)[-1]


def parse(xml, spec=None):
    """xml: bytes, str or an ElementTree element -> Node over a namespace-free copy of the tree (the root's namespace
    is kept in root.namespace)."""
    e = _copy.deepcopy(xml) if isinstance(xml, ET.Element) else ET.fromstring(xml)
    ns = e.tag[1:].split('}')[0] if e.tag.startswith('{') else None
    for x in e.iter():
        x.tag = _local(x.tag)
        if any(k.startswith('{') for k in x.attrib):
            for k in [k for k in x.attrib if k.startswith('{')]:
                del x.attrib[k]                 # schema-location hints and the like
    n = Node(e, spec or _PLAIN)
    object.__setattr__(n, '_ns', ns)
    return n


class Node:
    """One element of a parsed metadata tree; see the module docstring. node.Name: the child element (a Node, or a
    value for a leaf), a list for children the Spec says repeat, None when absent. node.attrib: the XML attributes.
    node.element: the ElementTree element. find('A/B'): the same by path."""
    __slots__ = ('_e', '_spec', '_ns')

    def __init__(self, element, spec=_PLAIN):
        object.__setattr__(self, '_e', element)
        object.__setattr__(self, '_spec', spec)
        object.__setattr__(self, '_ns', None)

    # ------------------------------------------------------------------ reading
    @property
    def element(self):
        return self._e

    @property
    def attrib(self):
        return self._e.attrib

    @property
    def tag(self):
        return self._e.tag

    @property
    def namespace(self):
        return self._ns

    def _value(self, c):
        if len(c):
            return Node(c, self._spec)
        t = (c.text or '').strip()
        sp = self._spec
        if c.tag in sp.text:
            return t
        if c.tag in sp.ints:
            try:
                return int(t)
            except ValueError:
                return t
        if t in ('true', 'false'):
            return t == 'true'
        try:
            return float(t)
        except ValueError:
            return t

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        e, sp = self._e, self._spec
        al = sp.aliases.get((e.tag, name))
        if al is not None:
            path, tag, kind = al
            base = e if not path else e.find(path)
            kids = [] if base is None else base.findall(tag)
            if kind == 'dict':
                return {k.get('name'): (k.text or '').strip() for k in kids} if base is not None else None
            if not kids:
                return None
            if kind == 'points':
                kids = sorted(kids, key=lambda k: int(str(k.get('index', '0')).split(':')[0] or 0))
                return np.array([Node(k, sp).get_array() for k in kids])
            return [self._value(k) for k in kids]
        kids = e.findall(name)
        if (e.tag, name) in sp.lists or name in sp.lists:
            return [self._value(k) for k in kids] or None
        return self._value(kids[0]) if kids else None

    def find(self, path):
        """The element at a '/'-separated path below this one, as attribute access gives it, or None."""
        n = self
        for k in path.split('/'):
            n = getattr(n, k) if isinstance(n, Node) else None
            if n is None:
                return None
        return n

    def children(self):
        """All child elements, in document order: [(name, Node or leaf value)]."""
        return [(c.tag, self._value(c)) for c in self._e]

    def get_array(self, dtype=np.float64):
        """A vector element (X, Y, Z; Lat, Lon[, HAE]; Row, Col; ...) as an array of its components, or a
        polynomial (Coef children with exponent1[, exponent2] attributes) as its coefficient array."""
        e = self._e
        coefs = e.findall('Coef')
        if coefs:
            two = any('exponent2' in c.attrib for c in coefs) or 'order2' in e.attrib
            n1 = max(int(e.get('order1', 0)), max(int(c.get('exponent1', 0)) for c in coefs)) + 1
            n2 = max(int(e.get('order2', 0)), max(int(c.get('exponent2', 0)) for c in coefs)) + 1
            a = np.zeros((n1, n2), dtype)
            for c in coefs:
                a[int(c.get('exponent1', 0)), int(c.get('exponent2', 0))] = float(c.text)
            return a if two else a[:, 0]
        tags = [c.tag for c in e]
        for v in _VECTORS:
            if tuple(tags[:len(v)]) == v and len(tags) == len(v):
                return np.array([float(c.text) for c in e], dtype)
        if tags and all(len(c) == 0 for c in e):
            return np.array([float(c.text) for c in e], dtype)
        raise TypeError(f'{e.tag} is neither a vector nor a polynomial element')

    def __getitem__(self, i):
        return self.get_array()[i]

    def __call__(self, *x):
        """Evaluate a polynomial element at x (one variable) or (x, y) (two); arrays broadcast."""
        a = self.get_array()
        if a.ndim == 1:
            return np.polynomial.polynomial.polyval(np.asarray(x[0], np.float64), a)
        return np.polynomial.polynomial.polyval2d(np.asarray(x[0], np.float64), np.asarray(x[1], np.float64), a)

    def __contains__(self, name):
        return self._e.find(name) is not None

    def __bool__(self):
        return True

    def __repr__(self):
        return f'<{self._e.tag}: {", ".join(dict.fromkeys(c.tag for c in self._e))}>'

    # ------------------------------------------------------------------ editing
    def __setattr__(self, name, value):
        e = self._e
        kids = e.findall(name)
        if value is None:
            for k in kids:
                e.remove(k)
            return
        if isinstance(value, Node):
            for k in kids:
                e.remove(k)
            new = _copy.deepcopy(value._e)
            new.tag = name
            e.append(new)
            return
        k = kids[0] if kids else ET.SubElement(e, name)
        if len(k):
            raise TypeError(f'{e.tag}.{name} is not a leaf element; assign a Node or None')
        k.text = text(value)

    def copy(self):
        n = Node(_copy.deepcopy(self._e), self._spec)
        object.__setattr__(n, '_ns', self._ns)
        return n

    def __getstate__(self):
        return (ET.tostring(self._e), self._spec, self._ns)

    def __setstate__(self, s):
        object.__setattr__(self, '_e', ET.fromstring(s[0]))
        object.__setattr__(self, '_spec', s[1])
        object.__setattr__(self, '_ns', s[2])

    # ------------------------------------------------------------------ writing
    def to_xml_bytes(self, declaration=False):
        """The tree as UTF-8 XML with the root's namespace as the default one."""
        e = self._e
        if self._ns:
            e = _copy.copy(e)                    # shallow: only the root gains the xmlns attribute
            e.attrib = dict(e.attrib, xmlns=self._ns)
        b = ET.tostring(e, encoding='utf-8')
        if b.startswith(b'<?xml'):
            b = b.split(b'?>', 1)[1].lstrip()
        return (b'<?xml version="1.0" encoding="utf-8"?>\n' if declaration else b'') + b

    def to_xml_string(self, **kw):
        return self.to_xml_bytes().decode()


def text(v):
    """A value as XML text: bool as true/false, float in the shortest form that reads back exactly, others str()."""
    if isinstance(v, (bool, np.bool_)):
        return 'true' if v else 'false'
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)
