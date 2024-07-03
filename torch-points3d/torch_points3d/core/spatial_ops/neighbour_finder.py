class MultiscaleRadiusNeighbourFinder(BaseMSNeighbourFinder):
    """Radius search with support for multiscale for sparse graphs

    Arguments:
        radius {Union[float, List[float]]}

    Keyword Arguments:
        max_num_neighbors {Union[int, List[int]]}  (default: {64})

    Raises:
        ValueError: [description]
    """

    def __init__(
        self,
        radius: Union[float, List[float]],
        max_num_neighbors: Union[int, List[int]] = 64,
    ):
        if DEBUGGING_VARS["FIND_NEIGHBOUR_DIST"]:
            if not isinstance(radius, list):
                radius = [radius]
            self._dist_meters = [DistributionNeighbour(r) for r in radius]
            if not isinstance(max_num_neighbors, list):
                max_num_neighbors = [max_num_neighbors]
            max_num_neighbors = [256 for _ in max_num_neighbors]

        if not is_list(max_num_neighbors) and is_list(radius):
            self._radius = cast(list, radius)
            max_num_neighbors = cast(int, max_num_neighbors)
            self._max_num_neighbors = [max_num_neighbors for i in range(len(self._radius))]
            return

        if not is_list(radius) and is_list(max_num_neighbors):
            self._max_num_neighbors = cast(list, max_num_neighbors)
            radius = cast(int, radius)
            self._radius = [radius for i in range(len(self._max_num_neighbors))]
            return

        if is_list(max_num_neighbors):
            max_num_neighbors = cast(list, max_num_neighbors)
            radius = cast(list, radius)
            if len(max_num_neighbors) != len(radius):
                raise ValueError("Both lists max_num_neighbors and radius should be of the same length")
            self._max_num_neighbors = max_num_neighbors
            self._radius = radius
            return

        self._max_num_neighbors = [cast(int, max_num_neighbors)]
        self._radius = [cast(int, radius)]

    def find_neighbours(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        if scale_idx >= self.num_scales:
            raise ValueError("Scale %i is out of bounds %i" % (scale_idx, self.num_scales))

        radius_idx = radius(
            x, y, self._radius[scale_idx], batch_x, batch_y, max_num_neighbors=self._max_num_neighbors[scale_idx]
        )
        return radius_idx

    @property
    def num_scales(self):
        return len(self._radius)

    def __call__(self, x, y, batch_x=None, batch_y=None, scale_idx=0):
        """Sparse interface of the neighboorhood finder"""
        return self.find_neighbours(x, y, batch_x, batch_y, scale_idx)