def line_graph(incidence):
    """
    Creates the line graph adjacency matrices for graphs with particular
    incidence matrices.
    :param incidence: The incidence matrices. Should have shape
        ([batch], n_nodes, n_edges).
    :return: The computed line graph adjacency matrices. It will have a shape
        of ([batch], n_edges, n_edges).
    """
    incidence = tf.convert_to_tensor(incidence, dtype=tf.float32)

    incidence_t = tf.linalg.matrix_transpose(incidence)
    incidence_sq = tf.matmul(incidence_t, incidence)

    num_rows = tf.shape(incidence_sq)[-2]
    identity = tf.eye(num_rows)
    return incidence_sq - identity * 2