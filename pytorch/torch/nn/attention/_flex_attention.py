def _create_empty_block_mask(query, key, value) -> BlockMask:
    device = query.device
    kv_len = key.size()[-2]
    q_len = query.size()[-2]
    return BlockMask(
        kv_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32, device=device),
        kv_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device),
        q_num_blocks=torch.ones([1, 1, 1], dtype=torch.int32, device=device),
        q_indices=torch.zeros([1, 1, 1, 1], dtype=torch.int32, device=device),
        KV_BLOCK_SIZE=kv_len,
        Q_BLOCK_SIZE=q_len,
    )

