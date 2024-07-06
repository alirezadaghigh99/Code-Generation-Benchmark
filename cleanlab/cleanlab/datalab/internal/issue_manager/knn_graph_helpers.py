def set_knn_graph(
    features: Optional[npt.NDArray],
    find_issues_kwargs: Dict[str, Any],
    metric: Optional[Metric],
    k: int,
    statistics: Dict[str, Any],
) -> Tuple[csr_matrix, Metric, Optional["NearestNeighbors"]]:
    # This only fetches graph (optionally)
    knn_graph = _process_knn_graph_from_inputs(
        find_issues_kwargs, statistics, k_for_recomputation=k
    )
    old_knn_metric = statistics.get("knn_metric", metric)

    missing_knn_graph = knn_graph is None
    metric_changes = metric and metric != old_knn_metric

    knn: Optional[NearestNeighbors] = None
    if missing_knn_graph or metric_changes:
        assert features is not None, "Features must be provided to compute the knn graph."
        knn_graph, knn = create_knn_graph_and_index(features, n_neighbors=k, metric=metric)
        metric = knn.metric
    return cast(csr_matrix, knn_graph), cast(Metric, metric), knn

