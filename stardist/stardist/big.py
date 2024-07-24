class Polygon:

    def __init__(self, coord, bbox=None, shape_max=None):
        self.bbox = self.coords_bbox(coord, shape_max=shape_max) if bbox is None else bbox
        self.coord = coord - np.array([r[0] for r in self.bbox]).reshape(2,1)
        self.slice = tuple(slice(*r) for r in self.bbox)
        self.shape = tuple(r[1]-r[0] for r in self.bbox)
        rr,cc = polygon(*self.coord, self.shape)
        self.mask = np.zeros(self.shape, bool)
        self.mask[rr,cc] = True

    @staticmethod
    def coords_bbox(*coords, shape_max=None):
        assert all(isinstance(c, np.ndarray) and c.ndim==2 and c.shape[0]==2 for c in coords)
        if shape_max is None:
            shape_max = (np.inf, np.inf)
        coord = np.concatenate(coords, axis=1)
        mins = np.maximum(0,         np.floor(np.min(coord,axis=1))).astype(int)
        maxs = np.minimum(shape_max, np.ceil (np.max(coord,axis=1))).astype(int)
        return tuple(zip(tuple(mins),tuple(maxs)))

class Polyhedron:

    def __init__(self, dist, origin, rays, bbox=None, shape_max=None):
        self.bbox = self.coords_bbox((dist, origin), rays=rays, shape_max=shape_max) if bbox is None else bbox
        self.slice = tuple(slice(*r) for r in self.bbox)
        self.shape = tuple(r[1]-r[0] for r in self.bbox)
        _origin = origin.reshape(1,3) - np.array([r[0] for r in self.bbox]).reshape(1,3)
        self.mask = polyhedron_to_label(dist[np.newaxis], _origin, rays, shape=self.shape, verbose=False).astype(bool)

    @staticmethod
    def coords_bbox(*dist_origin, rays, shape_max=None):
        dists, points = zip(*dist_origin)
        assert all(isinstance(d, np.ndarray) and d.ndim==1 and len(d)==len(rays) for d in dists)
        assert all(isinstance(p, np.ndarray) and p.ndim==1 and len(p)==3 for p in points)
        dists, points, verts = np.stack(dists)[...,np.newaxis], np.stack(points)[:,np.newaxis], rays.vertices[np.newaxis]
        coord = dists * verts + points
        coord = np.concatenate(coord, axis=0)
        if shape_max is None:
            shape_max = (np.inf, np.inf, np.inf)
        mins = np.maximum(0,         np.floor(np.min(coord,axis=0))).astype(int)
        maxs = np.minimum(shape_max, np.ceil (np.max(coord,axis=0))).astype(int)
        return tuple(zip(tuple(mins),tuple(maxs)))

