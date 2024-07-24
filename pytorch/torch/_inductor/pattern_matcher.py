class PatternMatcherPass:
    def __init__(
        self,
        prevent_match_across_mutations: bool = False,
        pass_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.patterns: DefaultDict[
            Tuple[str, torch.fx.node.Target], List[PatternEntry]
        ] = defaultdict(list)
        self.prevent_match_across_mutations = prevent_match_across_mutations
        self.pass_name = pass_name

    def __getitem__(self, item: Tuple[str, torch.fx.node.Target]) -> List[PatternEntry]:
        return self.patterns[item]

    def apply(self, gm: torch.fx.GraphModule) -> int:
        if not self.patterns:
            return 0
        if isinstance(gm, torch.fx.GraphModule):
            graph = gm.graph
        elif isinstance(gm, torch.fx.Graph):
            graph = gm
            gm = graph.owning_module
        else:
            raise RuntimeError(
                f"The input to PatternMatcherPass must be a GraphModule or a Graph, but got {type(gm)}"
            )
        if self.prevent_match_across_mutations:
            if should_compute_mutation_region_ids(graph):
                compute_mutation_region_ids(graph)
            get_mutation_region_id_partial = functools.partial(
                get_mutation_region_id, graph
            )
        count = 0
        nodes = []
        has_call_module = False
        for op, target in self.patterns:
            if op == "call_module":
                has_call_module = True
            else:
                nodes.append(graph.find_nodes(op=op, target=target, sort=False))
        if has_call_module:
            nodes.append(graph.find_nodes(op="call_module", sort=False))
        pass_name = self.pass_name if self.pass_name is not None else "pattern_matcher"
        with GraphTransformObserver(
            gm, pass_name, trace_config.log_url_for_graph_xform
        ):
            for node in sorted(itertools.chain.from_iterable(nodes), reverse=True):
                target = extract_target(node)
                if node.op == "call_module":
                    if (node.op, target) not in self.patterns:
                        continue

                # conservatively not applying pattern for cpu input,
                # since some of the patterns induce codegen and split nodes.
                # Note: we will only skip cpu compute if disable_cpp_codegen=True
                if fallback_node_due_to_unsupported_type(node, allow_cpu_inputs=False):
                    continue

                for entry in self.patterns[(node.op, target)]:
                    if node._erased:
                        break
                    m = entry.pattern.match(node)
                    # pattern match crosses mutation barrier - discard
                    if (
                        self.prevent_match_across_mutations
                        and is_match(m)
                        and len(set(map(get_mutation_region_id_partial, m.nodes))) != 1  # type: ignore[possibly-undefined]
                    ):
                        continue
                    if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG") == node.name:
                        log.warning("%s%s %s %s", node, node.args, m, entry.pattern)
                    if is_match(m) and entry.extra_check(m):
                        count += 1
                        entry.apply(m, graph, node)  # type: ignore[arg-type]
                        counters["inductor"]["pattern_matcher_count"] += 1
                        counters["inductor"]["pattern_matcher_nodes"] += len(m.nodes)
        return count

    def clear(self) -> None:
        self.patterns.clear()

