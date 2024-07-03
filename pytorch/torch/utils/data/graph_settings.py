def apply_sharding(
    datapipe: DataPipe,
    num_of_instances: int,
    instance_id: int,
    sharding_group=SHARDING_PRIORITIES.DEFAULT,
) -> DataPipe:
    r"""
    Apply dynamic sharding over the ``sharding_filter`` DataPipe that has a method ``apply_sharding``.

    RuntimeError will be raised when multiple ``sharding_filter`` are presented in the same branch.
    """
    graph = traverse_dps(datapipe)

    def _helper(graph, prev_applied=None):
        for dp, sub_graph in graph.values():
            applied = None
            if _is_sharding_datapipe(dp):
                if prev_applied is not None:
                    raise RuntimeError(
                        "Sharding twice on a single pipeline is likely unintended and will cause data loss. "
                        f"Sharding already applied to {prev_applied} while trying to apply to {dp}"
                    )
                # For BC, only provide sharding_group if accepted
                sig = inspect.signature(dp.apply_sharding)
                if len(sig.parameters) < 3:
                    dp.apply_sharding(num_of_instances, instance_id)
                else:
                    dp.apply_sharding(
                        num_of_instances, instance_id, sharding_group=sharding_group
                    )
                applied = dp
            if applied is None:
                applied = prev_applied
            _helper(sub_graph, applied)

    _helper(graph)

    return datapipe