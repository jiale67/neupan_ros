"""Parse Gazebo .world files into obstacle specs."""
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

PLUGIN = 'libwaypointObstaclePlugin.so'


@dataclass
class ObstacleSpec:
    name: str
    kind: str          # 'box' or 'cylinder'
    size: tuple        # box: (lx, ly); cylinder: (r,)
    x: float
    y: float
    yaw: float
    dynamic: bool
    path: tuple = ()   # world-frame ((x, y), ...) plugin waypoints; () if static


def _pose(elem):
    """(x, y, yaw) from a <pose> child of elem; zeros if absent. Ignores z/roll/pitch."""
    p = elem.find('pose') if elem is not None else None
    if p is None or not (p.text or '').strip():
        return 0.0, 0.0, 0.0
    v = [float(t) for t in p.text.split()]
    return v[0], v[1], v[5]


def _compose(a, b):
    ax, ay, ayaw = a
    bx, by, byaw = b
    c, s = math.cos(ayaw), math.sin(ayaw)
    return ax + c * bx - s * by, ay + s * bx + c * by, ayaw + byaw


def _geometry(col):
    g = col.find('geometry')
    box = g.find('box')
    if box is not None:
        s = [float(t) for t in box.find('size').text.split()]
        return 'box', (s[0], s[1])
    cyl = g.find('cylinder')
    if cyl is not None:
        return 'cylinder', (float(cyl.find('radius').text),)
    return None, None


def _path(model, model_pose):
    plugin = model.find('plugin')
    wps = [[float(t) for t in w.text.split()] for w in plugin.findall('.//path/waypoint')]
    if not wps:
        return ()
    ox, oy = 0.0, 0.0
    anchor = plugin.find('anchor_to_model_pose')
    if anchor is not None and anchor.text.strip() not in ('0', 'false'):
        ox, oy = model_pose[0] - wps[0][0], model_pose[1] - wps[0][1]
    return tuple((w[0] + ox, w[1] + oy) for w in wps)


def parse_world(path):
    world = ET.parse(path).getroot().find('world')
    state = {}
    st = world.find('state')
    if st is not None:
        for sm in st.findall('model'):
            links = {l.get('name'): _pose(l) for l in sm.findall('link')}
            state[sm.get('name')] = (_pose(sm), links)

    specs = []
    for m in world.findall('model'):
        name = m.get('name')
        if name == 'ground_plane':
            continue
        dynamic = any(p.get('filename') == PLUGIN for p in m.findall('plugin'))
        model_pose = _pose(m)
        if name in state:
            model_pose = state[name][0]
        cols = []
        for l in m.findall('link'):
            if name in state and l.get('name') in state[name][1]:
                link_world = state[name][1][l.get('name')]   # state link poses are world poses
            else:
                link_world = _compose(model_pose, _pose(l))
            for c in l.findall('collision'):
                kind, size = _geometry(c)
                if kind is None:
                    continue
                cols.append((kind, size, _compose(link_world, _pose(c))))
        path = _path(m, model_pose) if dynamic else ()
        for i, (kind, size, (x, y, yaw)) in enumerate(cols):
            nm = name if len(cols) == 1 else '%s/%d' % (name, i)
            specs.append(ObstacleSpec(nm, kind, size, x, y, yaw, dynamic, path))
    return specs
