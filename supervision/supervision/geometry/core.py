class Point:
    x: float
    y: float

    def as_xy_int_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

class Vector:
    start: Point
    end: Point

    @property
    def magnitude(self) -> float:
        """
        Calculate the magnitude (length) of the vector.

        Returns:
            float: The magnitude of the vector.
        """
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return sqrt(dx**2 + dy**2)

    @property
    def center(self) -> Point:
        """
        Calculate the center point of the vector.

        Returns:
            Point: The center point of the vector.
        """
        return Point(
            x=(self.start.x + self.end.x) / 2,
            y=(self.start.y + self.end.y) / 2,
        )

    def cross_product(self, point: Point) -> float:
        """
        Calculate the 2D cross product (also known as the vector product or outer
        product) of the vector and a point, treated as vectors in 2D space.

        Args:
            point (Point): The point to be evaluated, treated as the endpoint of a
                vector originating from the 'start' of the main vector.

        Returns:
            float: The scalar value of the cross product. It is positive if 'point'
                lies to the left of the vector (when moving from 'start' to 'end'),
                negative if it lies to the right, and 0 if it is collinear with the
                vector.
        """
        dx_vector = self.end.x - self.start.x
        dy_vector = self.end.y - self.start.y
        dx_point = point.x - self.start.x
        dy_point = point.y - self.start.y
        return (dx_vector * dy_point) - (dy_vector * dx_point)

class Point:
    x: float
    y: float

    def as_xy_int_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

