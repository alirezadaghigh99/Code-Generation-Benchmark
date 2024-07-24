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

class BinPackCollator:
    """Utility collator for packing to reduce padding."""

    def __init__(
        self,
        collator: Callable,
        target_batch_size: int,
        max_seq_len: int,
        pad_token_id: int,
        padding_side: Literal['left', 'right'],
        max_leftover_bins_to_keep: Optional[int] = None,
    ):
        self.base_collator = collator
        self.out_size = int(target_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.pad_token_id = int(pad_token_id)
        self.padding_side = padding_side

        if self.out_size <= 0:
            raise ValueError(f'{target_batch_size=} must be >0.')
        if self.max_seq_len <= 0:
            raise ValueError(f'{max_seq_len=} must be >0.')
        if self.pad_token_id < 0:
            raise ValueError(f'{pad_token_id=} must be >=0.')

        if max_leftover_bins_to_keep is not None and max_leftover_bins_to_keep < 0:
            raise ValueError(
                f'{max_leftover_bins_to_keep=} must be >=0 or None.',
            )
        self.max_leftover_bins_to_keep = max_leftover_bins_to_keep

        self.n_packed_tokens = 0
        self.n_total_tokens = 0
        self.n_packed_examples = 0

        self._leftover_bins: List[Tuple[int, Dict[str, torch.Tensor]]] = []

    @property
    def waste(self) -> float:
        return 1 - (self.n_packed_tokens / self.n_total_tokens)

    @property
    def efficiency(self) -> float:
        return self.n_packed_tokens / (
            self.max_seq_len * self.n_packed_examples
        )

    def __call__(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)
        return self.pack(batch)

    def pack(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'attention_mask' in batch
        assert 'input_ids' in batch

        for key in batch.keys():
            assert key in [
                'input_ids',
                'labels',
                'attention_mask',
                'sequence_id',
            ]
        # Cut everything down to size
        sizes, trimmed_examples = _trim_batch(batch)
        return self._pack_trimmed_examples(trimmed_examples, sizes)

    def _pack_trimmed_examples(
        self,
        trimmed_examples: List[Dict[str, torch.Tensor]],
        sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Packs trimmed examples into fixed-size bins and repads them.

        Args:
            trimmed_examples (List[Dict[str, torch.Tensor]]): A list of trimmed examples.
            sizes (List[int]): The sizes of the trimmed examples.

        Returns:
            Dict[str, torch.Tensor]: A batch of repadded examples ready for processing
        """
        # Apply our CS 101 bin packing algorithm.
        packed_examples, n_packed_tokens, n_total_tokens, leftover_bins = _first_fit_bin_packing(
            sizes=sizes,
            examples=trimmed_examples,
            num_bins=self.out_size,
            max_bin_size=self.max_seq_len,
            existing_bins=self._leftover_bins,
        )
        self.n_packed_tokens += n_packed_tokens
        self.n_total_tokens += n_total_tokens
        self.n_packed_examples += self.out_size
        self._leftover_bins = leftover_bins[:self.max_leftover_bins_to_keep]

        # Re-pad to max_seq_len and batch
        batch = _repad(
            packed_examples,
            max_seq_len=self.max_seq_len,
            pad_token_id=self.pad_token_id,
            padding_side=self.padding_side,
        )
        return batch

