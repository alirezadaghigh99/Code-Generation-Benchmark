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

def encode_string(
    input_string: str,
    vocab: str,
) -> List[int]:
    """Given a predefined mapping, encode the string to a sequence of numbers

    Args:
    ----
        input_string: string to encode
        vocab: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
    -------
        A list encoding the input_string
    """
    try:
        return list(map(vocab.index, input_string))
    except ValueError:
        raise ValueError(
            f"some characters cannot be found in 'vocab'. \
                         Please check the input string {input_string} and the vocabulary {vocab}"
        )

def decode_sequence(
    input_seq: Union[np.ndarray, SequenceType[int]],
    mapping: str,
) -> str:
    """Given a predefined mapping, decode the sequence of numbers to a string

    Args:
    ----
        input_seq: array to decode
        mapping: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
    -------
        A string, decoded from input_seq
    """
    if not isinstance(input_seq, (Sequence, np.ndarray)):
        raise TypeError("Invalid sequence type")
    if isinstance(input_seq, np.ndarray) and (input_seq.dtype != np.int_ or input_seq.max() >= len(mapping)):
        raise AssertionError("Input must be an array of int, with max less than mapping size")

    return "".join(map(mapping.__getitem__, input_seq))

