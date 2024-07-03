def Hypercube(length: int, n_dim: int = 1, *, pbc: bool = True, **kwargs) -> Lattice:
    r"""Constructs a hypercubic lattice with equal side length in all dimensions.
    Periodic boundary conditions can also be imposed.

    Args:
        length: Side length of the hypercube; must always be >=1
        n_dim: Dimension of the hypercube; must be at least 1.
        pbc: Whether the hypercube should have periodic boundary conditions
            (in all directions)
        kwargs: Additional keyword arguments are passed on to the constructor of
            :ref:`netket.graph.Lattice`.

    Examples:
         A 10x10x10 cubic lattice with periodic boundary conditions can be
         constructed as follows:

         >>> import netket
         >>> g = netket.graph.Hypercube(10, n_dim=3, pbc=True)
         >>> print(g.n_nodes)
         1000
    """
    if not isinstance(length, int) or length <= 0:
        raise TypeError("Argument `length` must be a positive integer")
    length_vector = [length] * n_dim
    return Grid(length_vector, pbc=pbc, **kwargs)