def attribute(
    val: T, doc: Optional[str] = None, **kwargs: Any
) -> DatasetAttribute[HDF5Any, T, Any]:
    """Creates a dataset attribute that contains both a value and associated metadata.

    Args:
        val (any): the dataset attribute value
        doc (str): the docstring that describes the attribute
        **kwargs: Additional keyword arguments may be passed, which represents metadata
            which describes the attribute.

    Returns:
        DatasetAttribute: an attribute object

    .. seealso:: :class:`~.Dataset`

    **Example**

    >>> hamiltonian = qml.Hamiltonian([1., 1.], [qml.Z(0), qml.Z(1)])
    >>> eigvals, eigvecs = np.linalg.eigh(qml.matrix(hamiltonian))
    >>> dataset = qml.data.Dataset(hamiltonian = qml.data.attribute(
    ...     hamiltonian,
    ...     doc="The hamiltonian of the system"))
    >>> dataset.eigen = qml.data.attribute(
    ...     {"eigvals": eigvals, "eigvecs": eigvecs},
    ...     doc="Eigenvalues and eigenvectors of the hamiltonain")

    This metadata can then be accessed using the :meth:`~.Dataset.attr_info` mapping:

    >>> dataset.attr_info["eigen"]["doc"]
    'Eigenvalues and eigenvectors of the hamiltonain'
    """
    return match_obj_type(val)(val, AttributeInfo(doc=doc, py_type=type(val), **kwargs))

