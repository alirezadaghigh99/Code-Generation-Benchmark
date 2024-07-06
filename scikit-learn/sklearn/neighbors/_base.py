def _is_sorted_by_data(graph):
    """Return whether the graph's non-zero entries are sorted by data.

    The non-zero entries are stored in graph.data and graph.indices.
    For each row (or sample), the non-zero entries can be either:
        - sorted by indices, as after graph.sort_indices();
        - sorted by data, as after _check_precomputed(graph);
        - not sorted.

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Neighbors graph as given by `kneighbors_graph` or
        `radius_neighbors_graph`. Matrix should be of format CSR format.

    Returns
    -------
    res : bool
        Whether input graph is sorted by data.
    """
    assert graph.format == "csr"
    out_of_order = graph.data[:-1] > graph.data[1:]
    line_change = np.unique(graph.indptr[1:-1] - 1)
    line_change = line_change[line_change < out_of_order.shape[0]]
    return out_of_order.sum() == out_of_order[line_change].sum()

