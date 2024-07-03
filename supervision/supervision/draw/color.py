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