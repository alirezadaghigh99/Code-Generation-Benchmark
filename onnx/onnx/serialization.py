    def get(self, fmt: str) -> ProtoSerializer:
        """Get a serializer for a format.

        Args:
            fmt: The format to get a serializer for.

        Returns:
            ProtoSerializer: The serializer for the format.

        Raises:
            ValueError: If the format is not supported.
        """
        try:
            return self._serializers[fmt]
        except KeyError:
            raise ValueError(
                f"Unsupported format: '{fmt}'. Supported formats are: {self._serializers.keys()}"
            ) from None