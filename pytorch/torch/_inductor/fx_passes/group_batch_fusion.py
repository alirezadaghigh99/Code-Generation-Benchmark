def find_independent_subset_greedy(
    node_list: Iterable[torch.fx.Node],
    graph_search_options: Dict[str, Any],
) -> Iterator[Iterable[torch.fx.Node]]:
    """
    Yields a list of subsets of `node_list` where no element in the subset
    depends on any other element in the subset. This results in a set of
    independent nodes which can be fused together.

    The order of `node_list` is preserved within each subset so we can benefit
    from split-cat elimination in later passes.

    During iteration it is only safe to mutate the graph by changing the nodes
    that have been returned.

    graph_search_options:
      - min_fuse_set_size: Minimum size of the subset to consider. Subsets below
        this size will be ignored.
      - max_fuse_set_size: Maximum size of the subset to consider. Subsets will
        be broken to be at most this size.
    """

    # Compute all the children of `node` which are members of
    # `interesting_nodes`.
    def find_dependent_nodes(node, interesting_nodes):
        visited_node_set: Set[torch.fx.Node] = {node}
        dep_set: Set[torch.fx.Node] = set()

        work = [node]
        while work:
            node = work.pop()
            for input_node in node.all_input_nodes:
                if input_node in interesting_nodes:
                    dep_set.add(input_node)

                if input_node not in visited_node_set:
                    visited_node_set.add(input_node)
                    work.append(input_node)

        return dep_set

    min_fuse_set_size = graph_search_options["min_fuse_set_size"]
    max_fuse_set_size = graph_search_options["max_fuse_set_size"]

    # node_list needs to be a set because we only track the nodes that are left
    # in it (and we want to do the `in` on a set, not a list). But we want to
    # keep the correct order.
    node_list = _OrderedSet(node_list)

    cache: Dict[torch.fx.Node, Set[torch.fx.Node]] = {}
    while node_list:
        subset: List[torch.fx.Node] = []
        subset_deps: Set[torch.fx.Node] = set()

        next_round_node_list = _OrderedSet()
        for node in node_list:
            if len(subset) >= max_fuse_set_size or node in subset_deps:
                next_round_node_list.append(node)
                continue

            dep_set = cache.pop(node, None)
            if dep_set is None:
                dep_set = find_dependent_nodes(node, node_list)

            if not dep_set.intersection(subset):
                subset.append(node)
                subset_deps.update(dep_set)
            else:
                next_round_node_list.append(node)
                cache[node] = dep_set

        if len(subset) >= min_fuse_set_size:
            # Careful here - the caller uses the subsets to fuse nodes together
            # so we need to clear any cache entry that contains one of the
            # returned nodes because the dependency list could be different
            # (larger) after the merge.
            cache = {k: v for k, v in cache.items() if v.isdisjoint(subset)}
            yield subset

        node_list = next_round_node_list