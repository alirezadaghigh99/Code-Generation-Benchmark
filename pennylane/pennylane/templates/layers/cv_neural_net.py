def shape(n_layers, n_wires):
        r"""Returns a list of shapes for the 11 parameter tensors.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            list[tuple[int]]: list of shapes
        """
        # n_if -> theta and phi shape for Interferometer
        n_if = n_wires * (n_wires - 1) // 2

        shapes = (
            [(n_layers, n_if)] * 2
            + [(n_layers, n_wires)] * 3
            + [(n_layers, n_if)] * 2
            + [(n_layers, n_wires)] * 4
        )

        return shapes

