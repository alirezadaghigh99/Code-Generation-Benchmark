class Rays_GoldenSpiral(Rays_Base):
    def __init__(self, n=70, anisotropy = None):
        if n<4:
            raise ValueError("At least 4 points have to be given!")
        super().__init__(n=n, anisotropy = anisotropy if anisotropy is None else tuple(anisotropy))

    def setup_vertices_faces(self):
        n = self.kwargs["n"]
        anisotropy = self.kwargs["anisotropy"]
        if anisotropy is None:
            anisotropy = np.ones(3)
        else:
            anisotropy = np.array(anisotropy)

        # the smaller golden angle = 2pi * 0.3819...
        g = (3. - np.sqrt(5.)) * np.pi
        phi = g * np.arange(n)
        # z = np.linspace(-1, 1, n + 2)[1:-1]
        # rho = np.sqrt(1. - z ** 2)
        # verts = np.stack([rho*np.cos(phi), rho*np.sin(phi),z]).T
        #
        z = np.linspace(-1, 1, n)
        rho = np.sqrt(1. - z ** 2)
        verts = np.stack([z, rho * np.sin(phi), rho * np.cos(phi)]).T

        # warnings.warn("ray definition has changed! Old results are invalid!")

        # correct for anisotropy
        verts = verts/anisotropy
        #verts /= np.linalg.norm(verts, axis=-1, keepdims=True)

        hull = ConvexHull(verts)
        faces = reorder_faces(verts,hull.simplices)

        verts /= np.linalg.norm(verts, axis=-1, keepdims=True)

        return verts, faces

