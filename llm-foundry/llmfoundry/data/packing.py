def auto_packing_ratio(
    dataloader_cfg: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
    num_packing_ratios: int = 20,
) -> float:
    """Find a packing ratio that minimizes padding with zero waste.

    By packing examples, we can increase training efficiency, training on more data with less batches.
    However, in practice, the selected packing_ratio may produce some waste because profiling is done on only
    a subset of the dataset.

    We select a min_ratio of 1 and a max_ratio that is the max_seq_len / 100, and profile up to
    num_packing_ratios packing ratios between min_ratio and max_ratio, inclusive.
    When a packing_ratio with non-zero waste is found, we stop and select the previous ratio,
    which has zero waste.

    Args:
        dataloader_cfg (DictConfig): The dataloader configuration for profiling.
        tokenizer (PreTrainedTokenizerBase): The tokenizer for profiling.
        device_batch_size (int): The size of the batches (number of examples) per device.
        num_packing_ratio (int): The number of packing ratios to try.

    Returns:
        A packing ratio that minimizes padding while maintaining zero waste.
    """
    from composer.utils import dist, get_device, reproducibility

    log.debug('Searching for optimal packing ratio.')

    # Stash the rng state to restore later.
    rng_state = reproducibility.get_rng_state()
    # Set the seed so that auto packing is deterministic.
    reproducibility.seed_all(0)

    # If max_seq_len is very small, skip profiling and select packing ratio of 1.
    dataset_config = dataloader_cfg['dataset']
    max_seq_len = dataset_config.get('max_seq_len')
    if max_seq_len <= 100:
        return 1

    min_ratio = 1
    max_ratio = max_seq_len / 100
    profiling_results = profile_packing(
        dataloader_cfg=dataloader_cfg,
        tokenizer=tokenizer,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        num_packing_ratios=num_packing_ratios,
        device_batch_size=device_batch_size,
    )

    # Obtain the maximum packing_ratio/minimum padding that has no waste.
    # profiling_results are sorted from smallest to largest packing_ratio.
    packing_ratio = 1
    for packing_ratio_candidate, _, waste in profiling_results:
        if waste is None or waste > 0:
            break
        packing_ratio = packing_ratio_candidate

    # Select the minimum packing ratio across all ranks.
    if dist.is_available() and dist.is_initialized():
        device = get_device(None)
        packing_ratio_tensor = device.tensor_to_device(
            torch.tensor(packing_ratio),
        )
        dist.all_reduce(packing_ratio_tensor, reduce_operation='MIN')
        packing_ratio = packing_ratio_tensor.item()

    # Restore rng state.
    reproducibility.load_rng_state(rng_state)

    return packing_ratio

