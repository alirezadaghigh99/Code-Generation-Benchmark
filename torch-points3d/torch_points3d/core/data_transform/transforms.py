class SphereSampling:
    """ Samples points within a sphere

    Parameters
    ----------
    radius : float
        Radius of the sphere
    sphere_centre : torch.Tensor or np.array
        Centre of the sphere (1D array that contains (x,y,z))
    align_origin : bool, optional
        move resulting point cloud to origin
    """

    KDTREE_KEY = KDTREE_KEY

    def __init__(self, radius, sphere_centre, align_origin=True):
        self._radius = radius
        self._centre = np.asarray(sphere_centre)
        if len(self._centre.shape) == 1:
            self._centre = np.expand_dims(self._centre, 0)
        self._align_origin = align_origin

    def __call__(self, data):
        num_points = data.pos.shape[0]
        if not hasattr(data, self.KDTREE_KEY):
            tree = KDTree(np.asarray(data.pos), leaf_size=50)
            setattr(data, self.KDTREE_KEY, tree)
        else:
            tree = getattr(data, self.KDTREE_KEY)

        t_center = torch.FloatTensor(self._centre)
        ind = torch.LongTensor(tree.query_radius(self._centre, r=self._radius)[0])
        new_data = Data()
        for key in set(data.keys):
            if key == self.KDTREE_KEY:
                continue
            item = data[key]
            if torch.is_tensor(item) and num_points == item.shape[0]:
                item = item[ind]
                if self._align_origin and key == "pos":  # Center the sphere.
                    item -= t_center
            elif torch.is_tensor(item):
                item = item.clone()
            setattr(new_data, key, item)
        return new_data

    def __repr__(self):
        return "{}(radius={}, center={}, align_origin={})".format(
            self.__class__.__name__, self._radius, self._centre, self._align_origin
        )

class MeshToNormal(object):
    """ Computes mesh normals (IN PROGRESS)
    """

    def __init__(self):
        pass

    def __call__(self, data):
        if hasattr(data, "face"):
            pos = data.pos
            face = data.face
            vertices = [pos[f] for f in face]
            normals = torch.cross(vertices[0] - vertices[1], vertices[0] - vertices[2], dim=1)
            normals = F.normalize(normals)
            data.normals = normals
        return data

    def __repr__(self):
        return "{}".format(self.__class__.__name__)

class RemoveAttributes(object):
    """This transform allows to remove unnecessary attributes from data for optimization purposes

    Parameters
    ----------
    attr_names: list
        Remove the attributes from data using the provided `attr_name` within attr_names
    strict: bool=False
        Wether True, it will raise an execption if the provided attr_name isn t within data keys.
    """

    def __init__(self, attr_names=[], strict=False):
        self._attr_names = attr_names
        self._strict = strict

    def __call__(self, data):
        keys = set(data.keys)
        for attr_name in self._attr_names:
            if attr_name not in keys and self._strict:
                raise Exception("attr_name: {} isn t within keys: {}".format(attr_name, keys))
        for attr_name in self._attr_names:
            delattr(data, attr_name)
        return data

    def __repr__(self):
        return "{}(attr_names={}, strict={})".format(self.__class__.__name__, self._attr_names, self._strict)

class RandomDropout:
    """ Randomly drop points from the input data

    Parameters
    ----------
    dropout_ratio : float, optional
        Ratio that gets dropped
    dropout_application_ratio   : float, optional
        chances of the dropout to be applied
    """

    def __init__(self, dropout_ratio: float = 0.2, dropout_application_ratio: float = 0.5):
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data):
        if random.random() < self.dropout_application_ratio:
            N = len(data.pos)
            data = FP(int(N * (1 - self.dropout_ratio)))(data)
        return data

    def __repr__(self):
        return "{}(dropout_ratio={}, dropout_application_ratio={})".format(
            self.__class__.__name__, self.dropout_ratio, self.dropout_application_ratio
        )

class RandomWalkDropout(object):
    """
    randomly drop points from input data using random walk

    Parameters
    ----------
    dropout_ratio: float, optional
        Ratio that gets dropped
    num_iter: int, optional
        number of iterations
    radius: float, optional
        radius of the neighborhood search to create the graph
    max_num: int optional
       max number of neighbors
    skip_keys: List optional
        skip_keys where we don't apply the mask
    """

    def __init__(
        self,
        dropout_ratio: float = 0.05,
        num_iter: int = 5000,
        radius: float = 0.5,
        max_num: int = -1,
        skip_keys: List = [],
    ):
        self.dropout_ratio = dropout_ratio
        self.num_iter = num_iter
        self.radius = radius
        self.max_num = max_num
        self.skip_keys = skip_keys

    def __call__(self, data):

        pos = data.pos.detach().cpu().numpy()
        ind, dist = ball_query(data.pos, data.pos, radius=self.radius, max_num=self.max_num, mode=0)
        mask = np.ones(len(pos), dtype=bool)
        mask = rw_mask(
            pos=pos,
            ind=ind.detach().cpu().numpy(),
            dist=dist.detach().cpu().numpy(),
            mask_vertices=mask,
            num_iter=self.num_iter,
            random_ratio=self.dropout_ratio,
        )

        data = apply_mask(data, mask, self.skip_keys)

        return data

    def __repr__(self):
        return "{}(dropout_ratio={}, num_iter={}, radius={}, max_num={}, skip_keys={})".format(
            self.__class__.__name__, self.dropout_ratio, self.num_iter, self.radius, self.max_num, self.skip_keys
        )

class ShiftVoxels:
    """ Trick to make Sparse conv invariant to even and odds coordinates
    https://github.com/chrischoy/SpatioTemporalSegmentation/blob/master/lib/train.py#L78

    Parameters
    -----------
    apply_shift: bool:
        Whether to apply the shift on indices
    """

    def __init__(self, apply_shift=True):
        self._apply_shift = apply_shift

    def __call__(self, data):
        if self._apply_shift:
            if not hasattr(data, "coords"):
                raise Exception("should quantize first using GridSampling3D")

            if not isinstance(data.coords, torch.IntTensor):
                raise Exception("The pos are expected to be coordinates, so torch.IntTensor")
            data.coords[:, :3] += (torch.rand(3) * 100).type_as(data.coords)
        return data

    def __repr__(self):
        return "{}(apply_shift={})".format(self.__class__.__name__, self._apply_shift)

class ScalePos:
    def __init__(self, scale=None):
        self.scale = scale

    def __call__(self, data):
        data.pos *= self.scale
        return data

    def __repr__(self):
        return "{}(scale={})".format(self.__class__.__name__, self.scale)

class CubeCrop(object):
    """
    Crop cubically the point cloud. This function take a cube of size c
    centered on a random point, then points outside the cube are rejected.

    Parameters
    ----------
    c: float, optional
        half size of the cube
    rot_x: float_otional
        rotation of the cube around x axis
    rot_y: float_otional
        rotation of the cube around x axis
    rot_z: float_otional
        rotation of the cube around x axis
    """

    def __init__(
        self, c: float = 1, rot_x: float = 180, rot_y: float = 180, rot_z: float = 180, grid_size_center: float = 0.01
    ):
        self.c = c
        self.random_rotation = Random3AxisRotation(rot_x=rot_x, rot_y=rot_y, rot_z=rot_z)
        self.grid_sampling = GridSampling3D(grid_size_center, mode="last")

    def __call__(self, data):
        data_c = self.grid_sampling(data.clone())
        data_temp = data.clone()
        i = torch.randint(0, len(data_c.pos), (1,))
        center = data_c.pos[i]
        min_square = center - self.c
        max_square = center + self.c
        data_temp.pos = data_temp.pos - center
        data_temp = self.random_rotation(data_temp)
        data_temp.pos = data_temp.pos + center
        mask = torch.prod((data_temp.pos - min_square) > 0, dim=1) * torch.prod((max_square - data_temp.pos) > 0, dim=1)
        mask = mask.to(torch.bool)
        data = apply_mask(data, mask)
        return data

    def __repr__(self):
        return "{}(c={}, rotation={})".format(self.__class__.__name__, self.c, self.random_rotation)

class SphereCrop(object):
    """
    crop the point cloud on a sphere. this function.
    takes a ball of radius radius centered on a random point and points
    outside the ball are rejected.

    Parameters
    ----------
    radius: float, optional
        radius of the sphere
    """

    def __init__(self, radius: float = 50):
        self.radius = radius

    def __call__(self, data):
        i = torch.randint(0, len(data.pos), (1,))
        ind, dist = ball_query(data.pos, data.pos[i].view(1, 3), radius=self.radius, max_num=-1, mode=1)
        ind = ind[dist[:, 0] > 0]
        size_pos = len(data.pos)
        for k in data.keys:
            if size_pos == len(data[k]):
                data[k] = data[k][ind[:, 0]]
        return data

    def __repr__(self):
        return "{}(radius={})".format(self.__class__.__name__, self.radius)

class RandomSphereDropout(object):
    """
    drop out of points on random spheres of fixed radius.
    This function takes n random balls of fixed radius r and drop
    out points inside these balls.

    Parameters
    ----------
    num_sphere: int, optional
        number of random spheres
    radius: float, optional
        radius of the spheres
    """

    def __init__(self, num_sphere: int = 10, radius: float = 5, grid_size_center: float = 0.01):
        self.num_sphere = num_sphere
        self.radius = radius
        self.grid_sampling = GridSampling3D(grid_size_center, mode="last")

    def __call__(self, data):

        data_c = self.grid_sampling(data.clone())
        list_ind = torch.randint(0, len(data_c.pos), (self.num_sphere,))
        center = data_c.pos[list_ind]
        pos = data.pos
        # list_ind = torch.randint(0, len(pos), (self.num_sphere,))

        ind, dist = ball_query(data.pos, center, radius=self.radius, max_num=-1, mode=1)
        ind = ind[dist[:, 0] >= 0]
        mask = torch.ones(len(pos), dtype=torch.bool)
        mask[ind[:, 0]] = False
        data = apply_mask(data, mask)

        return data

    def __repr__(self):
        return "{}(num_sphere={}, radius={})".format(self.__class__.__name__, self.num_sphere, self.radius)

class DensityFilter(object):
    """
    Remove points with a low density(compute the density with a radius search and remove points with)
    a low number of neighbors

    Parameters
    ----------
    radius_nn: float, optional
        radius for the neighbors search
    min_num: int, otional
        minimum number of neighbors to be dense
    skip_keys: int, otional
        list of attributes of data to skip when we apply the mask
    """

    def __init__(self, radius_nn: float = 0.04, min_num: int = 6, skip_keys: List = []):
        self.radius_nn = radius_nn
        self.min_num = min_num
        self.skip_keys = skip_keys

    def __call__(self, data):

        ind, dist = ball_query(data.pos, data.pos, radius=self.radius_nn, max_num=-1, mode=0)

        mask = (dist > 0).sum(1) > self.min_num
        data = apply_mask(data, mask, self.skip_keys)
        return data

    def __repr__(self):
        return "{}(radius_nn={}, min_num={}, skip_keys={})".format(
            self.__class__.__name__, self.radius_nn, self.min_num, self.skip_keys
        )

class Select:
    """ Selects given points from a data object

    Parameters
    ----------
    indices : torch.Tensor
        indeices of the points to keep. Can also be a boolean mask
    """

    def __init__(self, indices=None):
        self._indices = indices

    def __call__(self, data):
        num_points = data.pos.shape[0]
        new_data = Data()
        for key in data.keys:
            if key == KDTREE_KEY:
                continue
            item = data[key]
            if torch.is_tensor(item) and num_points == item.shape[0]:
                item = item[self._indices].clone()
            elif torch.is_tensor(item):
                item = item.clone()
            setattr(new_data, key, item)
        return new_data

class RandomSphere(object):
    """Select points within a sphere of a given radius. The centre is chosen randomly within the point cloud.

    Parameters
    ----------
    radius: float
        Radius of the sphere to be sampled.
    strategy: str
        choose between `random` and `freq_class_based`. The `freq_class_based` \
        favors points with low frequency class. This can be used to balance unbalanced datasets
    center: bool
        if True then the sphere will be moved to the origin
    """

    def __init__(self, radius, strategy="random", class_weight_method="sqrt", center=True):
        self._radius = eval(radius) if isinstance(radius, str) else float(radius)
        self._sampling_strategy = SamplingStrategy(strategy=strategy, class_weight_method=class_weight_method)
        self._center = center

    def _process(self, data):
        # apply sampling strategy
        random_center = self._sampling_strategy(data)
        random_center = np.asarray(data.pos[random_center])[np.newaxis]
        sphere_sampling = SphereSampling(self._radius, random_center, align_origin=self._center)
        return sphere_sampling(data)

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(radius={}, center={}, sampling_strategy={})".format(
            self.__class__.__name__, self._radius, self._center, self._sampling_strategy
        )

class CylinderSampling:
    """ Samples points within a cylinder

    Parameters
    ----------
    radius : float
        Radius of the cylinder
    cylinder_centre : torch.Tensor or np.array
        Centre of the cylinder (1D array that contains (x,y,z) or (x,y))
    align_origin : bool, optional
        move resulting point cloud to origin
    """

    KDTREE_KEY = KDTREE_KEY

    def __init__(self, radius, cylinder_centre, align_origin=True):
        self._radius = radius
        if cylinder_centre.shape[0] == 3:
            cylinder_centre = cylinder_centre[:-1]
        self._centre = np.asarray(cylinder_centre)
        if len(self._centre.shape) == 1:
            self._centre = np.expand_dims(self._centre, 0)
        self._align_origin = align_origin

    def __call__(self, data):
        num_points = data.pos.shape[0]
        if not hasattr(data, self.KDTREE_KEY):
            tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=50)
            setattr(data, self.KDTREE_KEY, tree)
        else:
            tree = getattr(data, self.KDTREE_KEY)

        t_center = torch.FloatTensor(self._centre)
        ind = torch.LongTensor(tree.query_radius(self._centre, r=self._radius)[0])

        new_data = Data()
        for key in set(data.keys):
            if key == self.KDTREE_KEY:
                continue
            item = data[key]
            if torch.is_tensor(item) and num_points == item.shape[0]:
                item = item[ind]
                if self._align_origin and key == "pos":  # Center the cylinder.
                    item[:, :-1] -= t_center
            elif torch.is_tensor(item):
                item = item.clone()
            setattr(new_data, key, item)
        return new_data

    def __repr__(self):
        return "{}(radius={}, center={}, align_origin={})".format(
            self.__class__.__name__, self._radius, self._centre, self._align_origin
        )

