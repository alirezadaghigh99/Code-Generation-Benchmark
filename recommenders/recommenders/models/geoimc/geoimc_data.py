class DataPtr:
    """
    Holds data and its respective indices
    """

    def __init__(self, data, entities):
        """Initialize a data pointer

        Args:
            data (csr_matrix): The target data matrix.
            entities (Iterator): An iterator (of 2 elements (ndarray)) containing
            the features of row, col entities.
        """
        assert isspmatrix_csr(data)

        self.data = data
        self.entities = entities
        self.data_indices = None
        self.entity_indices = [None, None]

    def get_data(self):
        """
        Returns:
            csr_matrix: Target matrix (based on the data_indices filter)
        """
        if self.data_indices is None:
            return self.data
        return self.data[self.data_indices]

    def get_entity(self, of="row"):
        """Get entity

        Args:
            of (str): The entity, either 'row' or 'col'
        Returns:
            numpy.ndarray: Entity matrix (based on the entity_indices filter)
        """
        idx = 0 if of == "row" else 1
        if self.entity_indices[idx] is None:
            return self.entities[idx]
        return self.entities[idx][self.entity_indices[idx]]

