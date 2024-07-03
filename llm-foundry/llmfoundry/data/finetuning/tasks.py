def tokenize_formatted_example(
    example: Example,
    tokenizer: PreTrainedTokenizerBase,
) -> TokenizedExample:
    """Tokenizes a formatted example using the provided tokenizer.

    Args:
        example (Example): The input example to be tokenized.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to be used for tokenization.

    Returns:
        TokenizedExample: The tokenized example.

    Raises:
        ValueError: If the example format is unknown.
    """
    example_format = _get_example_type(example)

    if example_format == 'chat':
        chat_example = cast(ChatFormattedDict, example)
        return _tokenize_chat_formatted_example(chat_example, tokenizer)
    elif example_format == 'prompt_response':
        prompt_response_example: PromptResponseDict = cast(
            PromptResponseDict,
            example,
        )
        return _tokenize_prompt_response_formatted_example(
            prompt_response_example,
            tokenizer,
        )
    else:
        raise NotImplementedError