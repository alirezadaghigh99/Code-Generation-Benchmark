        def expand_matrix(matrix):
            return reduce(np.kron, (pre_wires_identity, matrix, post_wires_identity))