def kernel_bw_lookup(
    compute_device: str,
    compute_kernel: str,
    hbm_mem_bw: float,
    ddr_mem_bw: float,
    caching_ratio: Optional[float] = None,
    prefetch_pipeline: bool = False,
) -> Optional[float]:
    """
    Calculates the device bandwidth based on given compute device, compute kernel, and
    caching ratio.

    Args:
        compute_kernel (str): compute kernel.
        compute_device (str): compute device.
        hbm_mem_bw (float): the bandwidth of the device HBM.
        ddr_mem_bw (float): the bandwidth of the system DDR memory.
        caching_ratio (Optional[float]): caching ratio used to determine device bandwidth
            if UVM caching is enabled.
        prefetch_pipeline (bool): whether prefetch pipeline is enabled.

    Returns:
        Optional[float]: the device bandwidth.
    """
    caching_ratio = caching_ratio if caching_ratio else UVM_CACHING_RATIO
    lookup = {
        # CPU
        ("cpu", EmbeddingComputeKernel.DENSE.value): 0.5 * ddr_mem_bw,
        ("cpu", EmbeddingComputeKernel.FUSED.value): 1 * ddr_mem_bw,
        ("cpu", EmbeddingComputeKernel.QUANT.value): 1 * ddr_mem_bw,
        # TODO: Determine the correct value later. MTIA uses values same as CPU's.
        # MTIA
        ("mtia", EmbeddingComputeKernel.DENSE.value): 0.5 * ddr_mem_bw,
        ("mtia", EmbeddingComputeKernel.FUSED.value): 1 * ddr_mem_bw,
        ("mtia", EmbeddingComputeKernel.QUANT.value): 1 * ddr_mem_bw,
        # CUDA
        ("cuda", EmbeddingComputeKernel.DENSE.value): 0.5 * hbm_mem_bw,
        ("cuda", EmbeddingComputeKernel.FUSED.value): 1 * hbm_mem_bw,
        ("cuda", EmbeddingComputeKernel.FUSED_UVM.value): ddr_mem_bw / 10,
        ("cuda", EmbeddingComputeKernel.FUSED_UVM_CACHING.value): (
            caching_ratio * hbm_mem_bw + (1 - caching_ratio) * ddr_mem_bw
        )
        / 10,
        ("cuda", EmbeddingComputeKernel.QUANT.value): 1 * hbm_mem_bw,
        ("cuda", EmbeddingComputeKernel.QUANT_UVM.value): ddr_mem_bw / 10,
        ("cuda", EmbeddingComputeKernel.QUANT_UVM_CACHING.value): (
            caching_ratio * hbm_mem_bw + (1 - caching_ratio) * ddr_mem_bw
        )
        / 10,
        ("cuda", EmbeddingComputeKernel.KEY_VALUE.value): ddr_mem_bw,
    }

    if (
        prefetch_pipeline
        and compute_device == "cuda"
        and compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
    ):
        return lookup.get(("cuda", EmbeddingComputeKernel.FUSED.value))

    return lookup.get((compute_device, compute_kernel))