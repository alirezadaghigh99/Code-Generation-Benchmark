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

def tokenizer_needs_prefix_space(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> bool:
    """Test for whether a prefix space is needed before the continuation.

    Sentencepiece tokenization should not have a prefix space, but gpt2 style
    BPE should.

    Args:
        tokenizer: Tokenizer to test

    Returns:
        bool: Whether or not the tokenizer needs a prefix space
    """
    test_tokens = tokenizer(' a', add_special_tokens=False)['input_ids']
    assert isinstance(test_tokens, list)
    return len(test_tokens) == 1

def get_fewshot_sample_idxs(
    dataset_size: int,
    num_fewshot: int,
    example_idx: int,
    rng: random.Random,
) -> Set[int]:
    """Samples indices without replacement. If num_fewshot exceeds the number.

    of unique examples in the dataset, then we will have fewer than num_fewshot examples in context.

    Args:
        dataset_size (int): Length of the dataset
        num_fewshot (int): Number of examples to prepend
        example_idx (int): Current example's index (excluded from fewshot choices)
        rng (random.Random): RNG for repeatable sample selection

    Returns:
        list: Indices of the examples chosen for fewshot selection
    """
    num_fewshot = min(dataset_size - 1, num_fewshot)
    fewshot_idxs = set(rng.sample(range(0, dataset_size), num_fewshot))

    if example_idx in fewshot_idxs:
        fewshot_idxs.remove(example_idx)
        if len(fewshot_idxs) >= dataset_size - 1:
            return fewshot_idxs

        replacement_sample = rng.choice(range(0, dataset_size))
        while replacement_sample in fewshot_idxs or replacement_sample == example_idx:
            replacement_sample = rng.choice(range(0, dataset_size))
        fewshot_idxs.add(replacement_sample)
    return fewshot_idxs

def get_continuation_span(
    context_enc: List,
    continuation_enc: List,
) -> torch.Tensor:
    """Gets the list of indices of the continuation tokens for language.

    modeling.

    or generation tasks.

    Args:
        context_enc (list): List of context tokens
        continuation_enc (list): List of continuation tokens

    Returns:
        torch.tensor: A tensor containing indices corresponding to continuation tokens
    """
    return torch.tensor(
        range(len(context_enc),
              len(context_enc) + len(continuation_enc)),
    )

def make_padded_input(
    context_enc: List,
    continuation_enc: List,
    max_seq_len: int,
    pad_tok_id: int,
    padding_side: str = 'right',
) -> torch.Tensor:
    """Takes an encoded context and continuation and clips the beginning of the.

    context if they're too long. Adds the padding token to the specified side.

    Args:
        context_enc (List): The encoded input to the model
        continuation_enc (List): The encoded desired output for the example
        max_seq_list (int): Maximum length sequences can be
        pad_tok_id (int): The token id we pad with
        padding_side (str): Which side to pad the context on. Can be 'right' or 'left

    Returns:
        input (torch.tensor): The padded and encoded context
        continuation_span (torch.tensor): The _inclusive_ range of indices corresponding to the continuation
    """
    inp = torch.tensor(
        (context_enc + continuation_enc),
        dtype=torch.long,
    )
    (inp_len,) = inp.shape

    # Sometimes tokenizers that have neither a pad_tok_id or eos_tok_id will pass None in as the padding
    # token and cause errors
    if not isinstance(pad_tok_id, int):
        raise ValueError(
            f'`pad_tok_id` must be an integer. Found {type(pad_tok_id)} instead',
        )
    # pad length from seq to padding_length
    if padding_side == 'right':
        inp = torch.cat(
            [
                inp,  # [seq]
                torch.LongTensor((max_seq_len - inp_len) * [pad_tok_id]),
            ],
            dim=0,
        )
    elif padding_side == 'left':
        inp = torch.cat(
            [
                torch.LongTensor((max_seq_len - inp_len) * [pad_tok_id]),
                inp,  # [seq]
            ],
            dim=0,
        )
    else:
        raise ValueError(
            f"Unknown padding_side {padding_side}. padding_side must be either 'left' or 'right'",
        )

    return inp

