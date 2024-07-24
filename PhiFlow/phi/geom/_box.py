class Box(BaseBox, metaclass=BoxType):
    """
    Simple cuboid defined by location of lower and upper corner in physical space.

    Boxes can be constructed either from two positional vector arguments `(lower, upper)` or by specifying the limits by dimension name as `kwargs`.

    Examples:
        >>> Box(x=1, y=1)  # creates a two-dimensional unit box with `lower=(0, 0)` and `upper=(1, 1)`.
        >>> Box(x=(None, 1), y=(0, None)  # creates a Box with `lower=(-inf, 0)` and `upper=(1, inf)`.

        The slicing constructor was updated in version 2.2 and now requires the dimension order as the first argument.

        >>> Box['x,y', 0:1, 0:1]  # creates a two-dimensional unit box with `lower=(0, 0)` and `upper=(1, 1)`.
        >>> Box['x,y', :1, 0:]  # creates a Box with `lower=(-inf, 0)` and `upper=(1, inf)`.
    """

    def __init__(self, lower: Tensor = None, upper: Tensor = None, **size: Union[int, Tensor, tuple, list]):
        """
        Args:
          lower: physical location of lower corner
          upper: physical location of upper corner
          **size: Specify size by dimension, either as `int` or `tuple` containing (lower, upper).
        """
        if lower is not None:
            assert isinstance(lower, Tensor), f"lower must be a Tensor but got {type(lower)}"
            assert 'vector' in lower.shape, "lower must have a vector dimension"
            assert lower.vector.item_names is not None, "vector dimension of lower must list spatial dimension order"
            self._lower = lower
        if upper is not None:
            assert isinstance(upper, Tensor), f"upper must be a Tensor but got {type(upper)}"
            assert 'vector' in upper.shape, "lower must have a vector dimension"
            assert upper.vector.item_names is not None, "vector dimension of lower must list spatial dimension order"
            self._upper = upper
        else:
            lower = []
            upper = []
            for item in size.values():
                if isinstance(item, (tuple, list)):
                    assert len(item) == 2, f"Box kwargs must be either dim=upper or dim=(lower,upper) but got {item}"
                    lo, up = item
                    lower.append(lo)
                    upper.append(up)
                elif item is None:
                    lower.append(-INF)
                    upper.append(INF)
                else:
                    lower.append(0)
                    upper.append(item)
            lower = [-INF if l is None else l for l in lower]
            upper = [INF if u is None else u for u in upper]
            self._upper = math.wrap(upper, math.channel(vector=tuple(size.keys())))
            self._lower = math.wrap(lower, math.channel(vector=tuple(size.keys())))
        vector_shape = self._lower.shape & self._upper.shape
        self._lower = math.expand(self._lower, vector_shape)
        self._upper = math.expand(self._upper, vector_shape)
        if self.size.vector.item_names is None:
            warnings.warn("Creating a Box without item names prevents certain operations like project()", DeprecationWarning, stacklevel=2)

    def __getitem__(self, item):
        item = _keep_vector(slicing_dict(self, item))
        return Box(self._lower[item], self._upper[item])

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        if all(isinstance(v, Box) for v in values):
            return NotImplemented  # stack attributes
        else:
            return Geometry.__stack__(values, dim, **kwargs)

    def __eq__(self, other):
        if self._lower is None and self._upper is None:
            return isinstance(other, Box)
        return isinstance(other, BaseBox)\
               and set(self.shape) == set(other.shape)\
               and self.size.shape.get_size('vector') == other.size.shape.get_size('vector')\
               and math.close(self._lower, other.lower)\
               and math.close(self._upper, other.upper)

    def without(self, dims: Tuple[str, ...]):
        remaining = list(self.shape.get_item_names('vector'))
        for dim in dims:
            if dim in remaining:
                remaining.remove(dim)
        return self.vector[remaining]

    def largest(self, dim: DimFilter) -> 'Box':
        dim = self.shape.without('vector').only(dim)
        if not dim:
            return self
        return Box(math.min(self._lower, dim), math.max(self._upper, dim))

    def __hash__(self):
        return hash(self._upper)

    def __variable_attrs__(self):
        return '_lower', '_upper'

    @property
    def shape(self):
        if self._lower is None or self._upper is None:
            return None
        return self._lower.shape & self._upper.shape

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def size(self):
        return self.upper - self.lower

    @property
    def center(self):
        return 0.5 * (self.lower + self.upper)

    @property
    def half_size(self):
        return self.size * 0.5

    def shifted(self, delta, **delta_by_dim):
        return Box(self.lower + delta, self.upper + delta)

    def __mul__(self, other):
        if not isinstance(other, Box):
            return NotImplemented
        lower = self._lower.vector.unstack(self.spatial_rank) + other._lower.vector.unstack(other.spatial_rank)
        upper = self._upper.vector.unstack(self.spatial_rank) + other._upper.vector.unstack(other.spatial_rank)
        names = self._upper.vector.item_names + other._upper.vector.item_names
        lower = math.stack(lower, math.channel(vector=names))
        upper = math.stack(upper, math.channel(vector=names))
        return Box(lower, upper)

    def __repr__(self):
        if self.shape.non_channel.volume == 1:
            item_names = self.size.vector.item_names
            if item_names:
                return f"Box({', '.join([f'{dim}=({lo}, {up})' for dim, lo, up in zip(item_names, self._lower, self._upper)])})"
            else:  # deprecated
                return 'Box[%s at %s]' % ('x'.join([str(x) for x in self.size.numpy().flatten()]), ','.join([str(x) for x in self.lower.numpy().flatten()]))
        else:
            return f'Box[shape={self.shape}]'

class Cuboid(BaseBox):
    """
    Box specified by center position and half size.
    """

    def __init__(self,
                 center: Tensor = 0,
                 half_size: Union[float, Tensor] = None,
                 **size: Union[float, Tensor]):
        if half_size is not None:
            assert isinstance(half_size, Tensor), "half_size must be a Tensor"
            assert 'vector' in half_size.shape, f"Cuboid size must have a 'vector' dimension."
            assert half_size.shape.get_item_names('vector') is not None, f"Vector dimension must list spatial dimensions as item names. Use the syntax Cuboid(x=x, y=y) to assign names."
            self._half_size = half_size
        else:
            self._half_size = math.wrap(tuple(size.values()), math.channel(vector=tuple(size.keys()))) * 0.5
        center = wrap(center)
        if 'vector' not in center.shape or center.shape.get_item_names('vector') is None:
            center = math.expand(center, channel(self._half_size))
        self._center = center


    def __eq__(self, other):
        if self._center is None and self._half_size is None:
            return isinstance(other, Cuboid)
        return isinstance(other, BaseBox)\
               and set(self.shape) == set(other.shape)\
               and math.close(self._center, other.center)\
               and math.close(self._half_size, other.half_size)

    def __hash__(self):
        return hash(self._center)

    def __repr__(self):
        return f"Cuboid(center={self._center}, half_size={self._half_size})"

    def __getitem__(self, item):
        item = _keep_vector(slicing_dict(self, item))
        return Cuboid(self._center[item], self._half_size[item])

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        if all(isinstance(v, Cuboid) for v in values):
            return Cuboid(math.stack([v.center for v in values], dim, **kwargs), math.stack([v.half_size for v in values], dim, **kwargs))
        else:
            return Geometry.__stack__(values, dim, **kwargs)

    def __variable_attrs__(self):
        return '_center', '_half_size'

    @property
    def center(self):
        return self._center

    @property
    def half_size(self):
        return self._half_size

    @property
    def shape(self):
        if self._center is None or self._half_size is None:
            return None
        return self._center.shape & self._half_size.shape

    @property
    def size(self):
        return 2 * self.half_size

    @property
    def lower(self):
        return self.center - self.half_size

    @property
    def upper(self):
        return self.center + self.half_size

    def shifted(self, delta, **delta_by_dim) -> 'Cuboid':
        return Cuboid(self._center + delta, self._half_size)

