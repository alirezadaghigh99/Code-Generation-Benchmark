class Color:
    """
    Represents a color in RGB format.

    This class provides methods to work with colors, including creating colors from hex
    codes, converting colors to hex strings, RGB tuples, and BGR tuples.

    Attributes:
        r (int): Red channel value (0-255).
        g (int): Green channel value (0-255).
        b (int): Blue channel value (0-255).

    Example:
        ```python
        import supervision as sv

        sv.Color.WHITE
        # Color(r=255, g=255, b=255)
        ```

    | Constant   | Hex Code   | RGB              |
    |------------|------------|------------------|
    | `WHITE`    | `#FFFFFF`  | `(255, 255, 255)`|
    | `BLACK`    | `#000000`  | `(0, 0, 0)`      |
    | `RED`      | `#FF0000`  | `(255, 0, 0)`    |
    | `GREEN`    | `#00FF00`  | `(0, 255, 0)`    |
    | `BLUE`     | `#0000FF`  | `(0, 0, 255)`    |
    | `YELLOW`   | `#FFFF00`  | `(255, 255, 0)`  |
    | `ROBOFLOW` | `#A351FB`  | `(163, 81, 251)` |
    """

    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, color_hex: str) -> Color:
        """
        Create a Color instance from a hex string.

        Args:
            color_hex (str): The hex string representing the color. This string can
                start with '#' followed by either 3 or 6 hexadecimal characters. In
                case of 3 characters, each character is repeated to form the full
                6-character hex code.

        Returns:
            Color: An instance representing the color.

        Example:
            ```python
            import supervision as sv

            sv.Color.from_hex('#ff00ff')
            # Color(r=255, g=0, b=255)

            sv.Color.from_hex('#f0f')
            # Color(r=255, g=0, b=255)
            ```
        """
        _validate_color_hex(color_hex)
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 3:
            color_hex = "".join(c * 2 for c in color_hex)
        r, g, b = (int(color_hex[i : i + 2], 16) for i in range(0, 6, 2))
        return cls(r, g, b)

    @classmethod
    def from_rgb_tuple(cls, color_tuple: Tuple[int, int, int]) -> Color:
        """
        Create a Color instance from an RGB tuple.

        Args:
            color_tuple (Tuple[int, int, int]): A tuple representing the color in RGB
                format, where each element is an integer in the range 0-255.

        Returns:
            Color: An instance representing the color.

        Example:
            ```python
            import supervision as sv

            sv.Color.from_rgb_tuple((255, 255, 0))
            # Color(r=255, g=255, b=0)
            ```
        """
        r, g, b = color_tuple
        return cls(r=r, g=g, b=b)

    @classmethod
    def from_bgr_tuple(cls, color_tuple: Tuple[int, int, int]) -> Color:
        """
        Create a Color instance from a BGR tuple.

        Args:
            color_tuple (Tuple[int, int, int]): A tuple representing the color in BGR
                format, where each element is an integer in the range 0-255.

        Returns:
            Color: An instance representing the color.

        Example:
            ```python
            import supervision as sv

            sv.Color.from_bgr_tuple((0, 255, 255))
            # Color(r=255, g=255, b=0)
            ```
        """
        b, g, r = color_tuple
        return cls(r=r, g=g, b=b)

    def as_hex(self) -> str:
        """
        Converts the Color instance to a hex string.

        Returns:
            str: The hexadecimal color string.

        Example:
            ```python
            import supervision as sv

            sv.Color(r=255, g=255, b=0).as_hex()
            # '#ffff00'
            ```
        """
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def as_rgb(self) -> Tuple[int, int, int]:
        """
        Returns the color as an RGB tuple.

        Returns:
            Tuple[int, int, int]: RGB tuple.

        Example:
            ```python
            import supervision as sv

            sv.Color(r=255, g=255, b=0).as_rgb()
            # (255, 255, 0)
            ```
        """
        return self.r, self.g, self.b

    def as_bgr(self) -> Tuple[int, int, int]:
        """
        Returns the color as a BGR tuple.

        Returns:
            Tuple[int, int, int]: BGR tuple.

        Example:
            ```python
            import supervision as sv

            sv.Color(r=255, g=255, b=0).as_bgr()
            # (0, 255, 255)
            ```
        """
        return self.b, self.g, self.r

    @classproperty
    def WHITE(cls) -> Color:
        return Color.from_hex("#FFFFFF")

    @classproperty
    def BLACK(cls) -> Color:
        return Color.from_hex("#000000")

    @classproperty
    def RED(cls) -> Color:
        return Color.from_hex("#FF0000")

    @classproperty
    def GREEN(cls) -> Color:
        return Color.from_hex("#00FF00")

    @classproperty
    def BLUE(cls) -> Color:
        return Color.from_hex("#0000FF")

    @classproperty
    def YELLOW(cls) -> Color:
        return Color.from_hex("#FFFF00")

    @classproperty
    def ROBOFLOW(cls) -> Color:
        return Color.from_hex("#A351FB")

    @classmethod
    @deprecated(
        "`Color.white()` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Color.WHITE` instead."
    )
    def white(cls) -> Color:
        return Color.from_hex(color_hex="#ffffff")

    @classmethod
    @deprecated(
        "`Color.black()` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Color.BLACK` instead."
    )
    def black(cls) -> Color:
        return Color.from_hex(color_hex="#000000")

    @classmethod
    @deprecated(
        "`Color.red()` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Color.RED` instead."
    )
    def red(cls) -> Color:
        return Color.from_hex(color_hex="#ff0000")

    @classmethod
    @deprecated(
        "`Color.green()` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Color.GREEN` instead."
    )
    def green(cls) -> Color:
        return Color.from_hex(color_hex="#00ff00")

    @classmethod
    @deprecated(
        "`Color.blue()` is deprecated and will be removed in "
        "`supervision-0.22.0`. Use `Color.BLUE` instead."
    )
    def blue(cls) -> Color:
        return Color.from_hex(color_hex="#0000ff")

