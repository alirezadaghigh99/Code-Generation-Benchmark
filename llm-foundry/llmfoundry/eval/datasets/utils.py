def trim_context(
    context_enc: List,
    continuation_enc: List,
    max_seq_len: int,
) -> List:
    """Trims a list of tokens down to `max_seq_len` if the length of the list.

    plus the continuation is more than `max_seq_len`. It will always trim tokens
    from the left, i.e. tokens at the beginning of the context will be removed.

    Args:
        context_enc (list): List of tokens in the context
        continuation_enc (list): List of tokens in the continuation
        max_seq_len (int): Maximum length the model can ingest

    Returns:
        list: The encoded context trimmed from the left
    """
    if len(continuation_enc) + len(context_enc) > max_seq_len:
        context_max_subseq_len = max_seq_len - len(continuation_enc)

        if context_max_subseq_len < 0:
            # can't support continuations which are longer than the max seq len
            raise Exception(
                f'Dataset included continuation longer than the max seq len',
            )

        # clip from the end
        context_enc = context_enc[-(context_max_subseq_len):]
    return context_enc