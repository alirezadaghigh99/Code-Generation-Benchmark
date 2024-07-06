def copy_to_cache(source, cache, lengths, block_tables, type: str = "prefill"):
    """
    Func: copy key/value into key/value cache.

    Args:   key/value(source): shape [bsz,seq_len,num_heads,head_size]
            cache: shape [num_blocks, num_kv_heads, head_size, block_size]
            lengths: key/value lengths
            block_tables
    """
    num_blocks, num_heads, block_size, head_size = cache.shape
    bsz, max_blocks_per_seq = block_tables.shape
    needed_blocks = (lengths + block_size - 1) // block_size

    if type == "prefill":
        for i in range(bsz):
            seq_len = lengths[i]
            block_num = needed_blocks[i]
            token_id = 0
            for block_idx in range(block_num - 1):
                cache[block_tables[i][block_idx]] = source[i][token_id : token_id + block_size].permute(1, 0, 2)
                token_id += block_size
            cache[block_tables[i][block_num - 1], :, : seq_len - token_id, :] = source[i][token_id:seq_len].permute(
                1, 0, 2
            )
    elif type == "decoding":
        assert source.size(1) == 1, "seq_len should be equal to 1 when decoding."
        source = source.squeeze(1)
        slot_idx = (lengths + block_size - 1) % block_size
        for i in range(bsz):
            cache[block_tables[i, needed_blocks[i] - 1], :, slot_idx[i], :] = source[i]

    return cache

