class NoopPerfModel(PerfModel):
    """
    A no-op model that returns the maximum perf among all shards. Here, no-op
    means we estimate the performance of a model without actually running it.
    """

    def __init__(self, topology: Topology) -> None:
        self._topology = topology

    def rate(self, plan: List[ShardingOption]) -> float:
        perfs = [0] * self._topology.world_size
        for sharding_option in plan:
            for shard in sharding_option.shards:
                # pyre-ignore [6]: Expected `typing_extensions.SupportsIndex`
                perfs[shard.rank] += cast(Perf, shard.perf).total

        return max(perfs)

class NoopStorageModel(PerfModel):
    """
    A no-op model that returns the maximum hbm usage among all shards. Here, no-op
    means we estimate the performance of a model without actually running it.
    """

    def __init__(self, topology: Topology) -> None:
        self._topology = topology

    def rate(self, plan: List[ShardingOption]) -> float:
        hbms = [0] * self._topology.world_size
        for sharding_option in plan:
            for shard in sharding_option.shards:
                # pyre-ignore [6]: Expected `typing_extensions.SupportsIndex`
                hbms[shard.rank] += cast(Storage, shard.storage).hbm

        return max(hbms)

