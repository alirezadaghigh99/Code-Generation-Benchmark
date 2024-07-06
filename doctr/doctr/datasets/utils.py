def translate(
    input_string: str,
    vocab_name: str,
    unknown_char: str = "■",
) -> str:
    """Translate a string input in a given vocabulary

    Args:
    ----
        input_string: input string to translate
        vocab_name: vocabulary to use (french, latin, ...)
        unknown_char: unknown character for non-translatable characters

    Returns:
    -------
        A string translated in a given vocab
    """
    if VOCABS.get(vocab_name) is None:
        raise KeyError("output vocabulary must be in vocabs dictionnary")

    translated = ""
    for char in input_string:
        if char not in VOCABS[vocab_name]:
            # we need to translate char into a vocab char
            if char in string.whitespace:
                # remove whitespaces
                continue
            # normalize character if it is not in vocab
            char = unicodedata.normalize("NFD", char).encode("ascii", "ignore").decode("ascii")
            if char == "" or char not in VOCABS[vocab_name]:
                # if normalization fails or char still not in vocab, return unknown character)
                char = unknown_char
        translated += char
    return translated

