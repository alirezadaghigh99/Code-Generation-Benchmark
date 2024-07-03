class CoordinateBox(object):
    """A coordinate box that represents a block in space.

    Molecular complexes are typically represented with atoms as
    coordinate points. Each complex is naturally associated with a
    number of different box regions. For example, the bounding box is a
    box that contains all atoms in the molecular complex. A binding
    pocket box is a box that focuses in on a binding region of a protein
    to a ligand. A interface box is the region in which two proteins
    have a bulk interaction.

    The `CoordinateBox` class is designed to represent such regions of
    space. It consists of the coordinates of the box, and the collection
    of atoms that live in this box alongside their coordinates.
    """

    def __init__(self, x_range: Tuple[float, float],
                 y_range: Tuple[float, float], z_range: Tuple[float, float]):
        """Initialize this box.

        Parameters
        ----------
        x_range: Tuple[float, float]
            A tuple of `(x_min, x_max)` with max and min x-coordinates.
        y_range: Tuple[float, float]
            A tuple of `(y_min, y_max)` with max and min y-coordinates.
        z_range: Tuple[float, float]
            A tuple of `(z_min, z_max)` with max and min z-coordinates.

        Raises
        ------
        `ValueError` if this interval is malformed
        """
        if not isinstance(x_range, tuple) or not len(x_range) == 2:
            raise ValueError("x_range must be a tuple of length 2")
        else:
            x_min, x_max = x_range
            if not x_min <= x_max:
                raise ValueError("x minimum must be <= x maximum")
        if not isinstance(y_range, tuple) or not len(y_range) == 2:
            raise ValueError("y_range must be a tuple of length 2")
        else:
            y_min, y_max = y_range
            if not y_min <= y_max:
                raise ValueError("y minimum must be <= y maximum")
        if not isinstance(z_range, tuple) or not len(z_range) == 2:
            raise ValueError("z_range must be a tuple of length 2")
        else:
            z_min, z_max = z_range
            if not z_min <= z_max:
                raise ValueError("z minimum must be <= z maximum")
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def __repr__(self):
        """Create a string representation of this box"""
        x_str = str(self.x_range)
        y_str = str(self.y_range)
        z_str = str(self.z_range)
        return "Box[x_bounds=%s, y_bounds=%s, z_bounds=%s]" % (x_str, y_str,
                                                               z_str)

    def __str__(self):
        """Create a string representation of this box."""
        return self.__repr__()

    def __contains__(self, point: Sequence[float]) -> bool:
        """Check whether a point is in this box.

        Parameters
        ----------
        point: Sequence[float]
            3-tuple or list of length 3 or np.ndarray of shape `(3,)`.
            The `(x, y, z)` coordinates of a point in space.

        Returns
        -------
        bool
            `True` if `other` is contained in this box.
        """
        (x_min, x_max) = self.x_range
        (y_min, y_max) = self.y_range
        (z_min, z_max) = self.z_range
        x_cont = (x_min <= point[0] and point[0] <= x_max)
        y_cont = (y_min <= point[1] and point[1] <= y_max)
        z_cont = (z_min <= point[2] and point[2] <= z_max)
        return x_cont and y_cont and z_cont

    # FIXME: Argument 1 of "__eq__" is incompatible with supertype "object"
    def __eq__(self, other: "CoordinateBox") -> bool:  # type: ignore
        """Compare two boxes to see if they're equal.

        Parameters
        ----------
        other: CoordinateBox
            Compare this coordinate box to the other one.

        Returns
        -------
        bool
            That's `True` if all bounds match.

        Raises
        ------
        `ValueError` if attempting to compare to something that isn't a
        `CoordinateBox`.
        """
        if not isinstance(other, CoordinateBox):
            raise ValueError("Can only compare to another box.")
        return (self.x_range == other.x_range and
                self.y_range == other.y_range and self.z_range == other.z_range)

    def __hash__(self) -> int:
        """Implement hashing function for this box.

        Uses the default `hash` on `self.x_range, self.y_range,
        self.z_range`.

        Returns
        -------
        int
            Unique integer
        """
        return hash((self.x_range, self.y_range, self.z_range))

    def center(self) -> Tuple[float, float, float]:
        """Computes the center of this box.

        Returns
        -------
        Tuple[float, float, float]
            `(x, y, z)` the coordinates of the center of the box.

        Examples
        --------
        >>> box = CoordinateBox((0, 1), (0, 1), (0, 1))
        >>> box.center()
        (0.5, 0.5, 0.5)
        """
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range
        return (x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2,
                z_min + (z_max - z_min) / 2)

    def volume(self) -> float:
        """Computes and returns the volume of this box.

        Returns
        -------
        float
            The volume of this box. Can be 0 if box is empty

        Examples
        --------
        >>> box = CoordinateBox((0, 1), (0, 1), (0, 1))
        >>> box.volume()
        1
        """
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range
        return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    def contains(self, other: "CoordinateBox") -> bool:
        """Test whether this box contains another.

        This method checks whether `other` is contained in this box.

        Parameters
        ----------
        other: CoordinateBox
            The box to check is contained in this box.

        Returns
        -------
        bool
            `True` if `other` is contained in this box.

        Raises
        ------
        `ValueError` if `not isinstance(other, CoordinateBox)`.
        """
        if not isinstance(other, CoordinateBox):
            raise ValueError("other must be a CoordinateBox")
        other_x_min, other_x_max = other.x_range
        other_y_min, other_y_max = other.y_range
        other_z_min, other_z_max = other.z_range
        self_x_min, self_x_max = self.x_range
        self_y_min, self_y_max = self.y_range
        self_z_min, self_z_max = self.z_range
        return (self_x_min <= other_x_min and other_x_max <= self_x_max and
                self_y_min <= other_y_min and other_y_max <= self_y_max and
                self_z_min <= other_z_min and other_z_max <= self_z_max)