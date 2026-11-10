"""A light attribute-style view of an XML metadata tree (SICD, SIDD, CPHD), in place of one class per element.

    root = parse(xml_bytes, spec)            # Node over the root element
    root.Grid.Row.SS                         # a leaf: float, int, bool or str
    root.GeoData.SCP.ECF.get_array()         # [X, Y, Z] as float64
    root.Grid.TimeCOAPoly.get_array()        # polynomial coefficients [order1 + 1, order2 + 1]
    root.Grid.TimeCOAPoly[0, 0]
    root.Nope                                # None: a missing element reads as None (an all-lower-case name, which
                                             # no element has, raises AttributeError)
    root.Grid.Type = 'PLANE'; root.PFA = None    # set a leaf's text; None removes the element
    root.to_xml_bytes()

Conventions. Element names are the XML's own local names (namespaces are dropped on parsing and the root's is
restored on writing). A Spec (one per format) names what the tree alone cannot tell: which leaves are text whatever
they look like, which are integers (by name, or as 'Parent/Name'), which children repeat (and so read as lists),
and aliases, the plural names
under which a list is also reached (Data.Channels for the Channel elements of a CPHD's Data block). Leaves not named
in the Spec read as bool ('true', 'false'), else float where the text parses as one, else str."""
import copy as _copy
import re
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


class reading:
    """Context manager around the reading of a file: whatever its bytes make the parsing code raise (a missing key,
    a field that is not a number, an offset past the end, XML that does not parse) leaves as a ValueError whose
    message starts with the file's path and says what is wrong. OSError passes as it is. wrap=False only puts the
    path in front of ValueErrors (for code that does more than parse, whose other exceptions are not the file's)."""

    def __init__(self, path, what='file', wrap=True):
        self.path, self.what, self.wrap = str(path), what, wrap

    def __enter__(self):
        return self

    def __exit__(self, kind, e, tb):
        if e is None or isinstance(e, OSError) or not isinstance(e, Exception):
            return False
        msg = str(e)
        if isinstance(e, ValueError) and msg.startswith(self.path):
            return False
        if not self.wrap and not isinstance(e, ValueError):
            return False
        if isinstance(e, KeyError):
            msg = f'{msg} is missing' if e.args and isinstance(e.args[0], str) and ' ' not in e.args[0] else str(e.args[0])
        elif isinstance(e, (MemoryError, RecursionError)):
            msg = f'the {self.what} asks for more memory or nesting than can be right ({type(e).__name__})'
        elif not isinstance(e, ValueError):
            msg = f'malformed {self.what} ({type(e).__name__}: {msg})'
        raise ValueError(f'{self.path}: {msg}') from e


_DECLARATION = re.compile(rb'<!\s*(DOCTYPE|ENTITY)', re.I)


def fromstring(xml, what='XML'):
    """bytes or str -> ElementTree root element. The document may start after padding (spaces, NULs) and may be
    UTF-16 with a byte order mark. A document type or entity declaration is refused outright: SICD, SIDD and CPHD
    metadata have none, and they are the way to entity expansion and to references outside the document.
    ValueError for XML that does not parse."""
    b = xml.encode() if isinstance(xml, str) else bytes(xml)
    if b[:2] in (b'\xff\xfe', b'\xfe\xff'):
        try:
            b = re.sub(r'^\s*<\?xml[^>]*\?>', '', b.decode('utf-16')).encode()
        except UnicodeError:
            raise ValueError(f'the {what} is not valid UTF-16') from None
    b = b.strip(b' \t\r\n\0')
    if b.startswith(b'\xef\xbb\xbf'):
        b = b[3:]
    if not b.startswith(b'<'):
        raise ValueError(f'the {what} does not start with an element (it starts {b[:16]!r})')
    if _DECLARATION.search(b):
        raise ValueError(f'the {what} contains a document type or entity declaration, which is not accepted')
    try:
        return ET.fromstring(b)
    except ET.ParseError as e:
        raise ValueError(f'the {what} is not well-formed XML ({e})') from None
    except RecursionError:
        raise ValueError(f'the {what} is nested too deeply') from None


def root_name(xml):
    """The local name of the root element of an XML document (bytes), or None; found by scanning past the
    declaration, comments and processing instructions however long they are, without parsing the document."""
    b = bytes(xml)
    if b[:2] in (b'\xff\xfe', b'\xfe\xff'):
        try:
            b = b.decode('utf-16').encode()
        except UnicodeError:
            return None
    i = 0
    while True:
        i = b.find(b'<', i)
        if i < 0:
            return None
        for start, end in ((b'<!--', b'-->'), (b'<?', b'?>'), (b'<!', b'>')):
            if b.startswith(start, i):
                j = b.find(end, i + len(start))
                if j < 0:
                    return None
                i = j + len(end)
                break
        else:
            m = re.match(rb'<(?:[\w.-]+:)?([\w.-]+)', b[i:i + 256])
            return m.group(1).decode() if m else None


def parse(xml, spec=None, what='XML'):
    """xml: bytes, str or an ElementTree element -> Node over a namespace-free copy of the tree (the root's namespace
    is kept in root.namespace). ValueError for text that is not well-formed XML or that declares a document type
    or entities (see fromstring)."""
    e = _copy.deepcopy(xml) if isinstance(xml, ET.Element) else fromstring(xml, what)
    ns = e.tag[1:].split('}')[0] if e.tag.startswith('{') else None
    for x in e.iter():
        x.tag = _local(x.tag)
        for k in [k for k in x.attrib if k.startswith('{')]:
            v = x.attrib.pop(k)
            if 'XMLSchema-instance' not in k:   # schema-location hints are dropped; others read by local name
                x.attrib.setdefault(_local(k), v)
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
        if c.tag in sp.ints or f'{self._e.tag}/{c.tag}' in sp.ints:
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
        if name.startswith('_') or name.islower():      # no element name is all lower case: not a metadata field
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
        """A vector element (X, Y, Z; Lat, Lon[, HAE]; Row, Col; ...) as an array of its components, a polynomial
        (Coef children with exponent1[, exponent2] attributes) as its coefficient array, a position polynomial (X, Y
        and Z polynomials) as [3, order + 1], or a list of indexed points as [n, 2]."""
        e = self._e
        coefs = e.findall('Coef')
        if coefs:
            two = any('exponent2' in c.attrib for c in coefs) or 'order2' in e.attrib
            n1 = max(int(e.get('order1', 0)), max(int(c.get('exponent1', 0)) for c in coefs)) + 1
            n2 = max(int(e.get('order2', 0)), max(int(c.get('exponent2', 0)) for c in coefs)) + 1
            a = np.zeros((n1, n2), dtype)
            try:
                for c in coefs:
                    a[int(c.get('exponent1', 0)), int(c.get('exponent2', 0))] = float(c.text)
            except (TypeError, ValueError):
                raise ValueError(f'{e.tag}: a polynomial coefficient is not a number') from None
            return a if two else a[:, 0]
        tags = [c.tag for c in e]
        if tags == ['X', 'Y', 'Z'] and all(c.find('Coef') is not None for c in e):
            # a position polynomial: the coefficients of X, Y and Z as rows [3, order + 1]
            rows = [Node(c, self._spec).get_array(dtype) for c in e]
            out = np.zeros((3, max(len(r) for r in rows)), dtype)
            for k, r in enumerate(rows):
                out[k, :len(r)] = r
            return out
        if tags and all(len(c) == 0 for c in e):
            try:
                return np.array([float(c.text) for c in e], dtype)
            except (TypeError, ValueError):
                raise ValueError(f'{e.tag}: its components {tags} are not all numbers') from None
        if tags and len(set(tags)) == 1 and all('index' in c.attrib for c in e):      # ICP, Vertex, IACP: [n, 2]
            kids = sorted(e, key=lambda c: int(str(c.get('index')).split(':')[0]))
            return np.array([Node(c, self._spec).get_array(dtype) for c in kids])
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
