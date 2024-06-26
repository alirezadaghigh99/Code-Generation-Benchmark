def fill(sequence: List[V], desired_size: int, content: V) -> List[V]:
    """
    Fill the sequence with padding elements until the sequence reaches
    the desired size.

    Args:
        sequence (List[V]): The input sequence.
        desired_size (int): The expected size of the output list. The
            difference between this value and the actual length of `sequence`
            (if positive) dictates how many elements will be added as padding.
        content (V): The element to be placed at the end of the input
            `sequence` as padding.

    Returns:
        (List[V]): A padded version of the input `sequence` (if needed).

    Examples:
        ```python
        fill([1, 2], 4, 0)
        # [1, 2, 0, 0]

        fill(['a', 'b'], 3, 'c')
        # ['a', 'b', 'c']
        ```
    """
    missing_size = max(0, desired_size - len(sequence))
    sequence.extend([content] * missing_size)
    return sequence