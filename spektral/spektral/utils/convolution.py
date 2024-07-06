def incidence_matrix(adjacency):
    """
    Creates the corresponding incidence matrices for graphs with particular
    adjacency matrices.

    :param adjacency: The binary adjacency matrices. Should have shape
        ([batch], n_nodes, n_nodes).
    :return: The computed incidence matrices. It will have a shape of
        ([batch], n_nodes, n_edges).
    """
    adjacency = tf.convert_to_tensor(adjacency, dtype=tf.float32)
    added_batch = False
    if tf.size(tf.shape(adjacency)) == 2:
        # Add the extra batch dimension if needed.
        adjacency = tf.expand_dims(adjacency, axis=0)
        added_batch = True

    # Compute the maximum number of edges. We will pad everything in the
    # batch to this dimension.
    adjacency_upper = _triangular_adjacency(adjacency)
    num_edges = tf.math.count_nonzero(adjacency_upper, axis=(1, 2))
    max_num_edges = tf.reduce_max(num_edges)

    # Compute all the transformation matrices.
    make_single_matrix = partial(_incidence_matrix_single, num_edges=max_num_edges)
    transformation_matrices = tf.map_fn(
        make_single_matrix,
        adjacency_upper,
        fn_output_signature=tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    )

    if added_batch:
        # Remove the extra batch dimension before returning.
        transformation_matrices = transformation_matrices[0]
    return transformation_matrices

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

