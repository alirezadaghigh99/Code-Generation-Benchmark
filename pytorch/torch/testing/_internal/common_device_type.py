def _has_sufficient_memory(device, size):
    if torch.device(device).type == "cuda":
        if not torch.cuda.is_available():
            return False
        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.mem_get_info, aka cudaMemGetInfo, returns a tuple of (free memory, total memory) of a GPU
        if device == "cuda":
            device = "cuda:0"
        return torch.cuda.memory.mem_get_info(device)[0] >= size

    if device == "xla":
        raise unittest.SkipTest("TODO: Memory availability checks for XLA?")

    if device == "xpu":
        raise unittest.SkipTest("TODO: Memory availability checks for Intel GPU?")

    if device != "cpu":
        raise unittest.SkipTest("Unknown device type")

    # CPU
    if not HAS_PSUTIL:
        raise unittest.SkipTest("Need psutil to determine if memory is sufficient")

    # The sanitizers have significant memory overheads
    if TEST_WITH_ASAN or TEST_WITH_TSAN or TEST_WITH_UBSAN:
        effective_size = size * 10
    else:
        effective_size = size

    if psutil.virtual_memory().available < effective_size:
        gc.collect()
    return psutil.virtual_memory().available >= effective_size

def get_all_device_types() -> List[str]:
    return ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]

