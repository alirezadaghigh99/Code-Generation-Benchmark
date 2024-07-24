class NodeLatency(NamedTuple):
    # Latency due to the memory bandwidth
    mem_latency_sec: float
    # Latency due to the computation
    computer_latency_sec: float

