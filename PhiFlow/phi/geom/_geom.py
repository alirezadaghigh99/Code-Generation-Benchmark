class Point(Geometry):
    """
    Points have zero volume and are determined by a single location.
    An instance of `Point` represents a single n-dimensional point or a batch of points.
    """

    def __init__(self, location: math.Tensor):
        assert 'vector' in location.shape, "location must have a vector dimension"
        assert location.shape.get_item_names('vector') is not None, "Vector dimension needs to list spatial dimension as item names."
        self._location = location

    @property
    def center(self) -> Tensor:
        return self._location

    @property
    def shape(self) -> Shape:
        return self._location.shape

    def unstack(self, dimension: str) -> tuple:
        return tuple(Point(loc) for loc in self._location.unstack(dimension))

    def lies_inside(self, location: Tensor) -> Tensor:
        return expand(math.wrap(False), shape(location).without('vector'))

    def approximate_signed_distance(self, location: Union[Tensor, tuple]) -> Tensor:
        return math.vec_abs(location - self._location)

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        return positions

    def bounding_radius(self) -> Tensor:
        return math.zeros()

    def bounding_half_extent(self) -> Tensor:
        return math.zeros()

    def at(self, center: Tensor) -> 'Geometry':
        return Point(center)

    def rotated(self, angle) -> 'Geometry':
        return self

    def __hash__(self):
        return hash(self._location)

    def __variable_attrs__(self):
        return '_location',

    @property
    def volume(self) -> Tensor:
        return math.wrap(0)

    @property
    def shape_type(self) -> Tensor:
        return math.tensor('P')

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return self

    def __getitem__(self, item):
        return Point(self._location[_keep_vector(slicing_dict(self, item))])

