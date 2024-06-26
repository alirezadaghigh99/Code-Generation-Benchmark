def merge_multi_strings(seq_list: List[str], dil_factor: float) -> str:
    """Recursively merges consecutive string sequences with overlapping characters.

    Args:
    ----
        seq_list: list of sequences to merge. Sequences need to be ordered from left to right.
        dil_factor: dilation factor of the boxes to overlap, should be > 1. This parameter is
            only used when the mother sequence is splitted on a character repetition

    Returns:
    -------
        A merged character sequence

    Example::
        >>> from doctr.model.recognition.utils import merge_multi_sequences
        >>> merge_multi_sequences(['abc', 'bcdef', 'difghi', 'aijkl'], 1.4)
        'abcdefghijkl'
    """

    def _recursive_merge(a: str, seq_list: List[str], dil_factor: float) -> str:
        # Recursive version of compute_overlap
        if len(seq_list) == 1:
            return merge_strings(a, seq_list[0], dil_factor)
        return _recursive_merge(merge_strings(a, seq_list[0], dil_factor), seq_list[1:], dil_factor)

    return _recursive_merge("", seq_list, dil_factor)