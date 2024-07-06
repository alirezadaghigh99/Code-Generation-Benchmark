def modified_precision(references: Sequence[Sequence[Any]], candidate: Any, n: int) -> Tuple[int, int]:
    """
    Compute the modified precision

    .. math::
       p_{n} = \frac{m_{n}}{l_{n}}

    where m_{n} is the number of matched n-grams between translation T and its reference R, and l_{n} is the
    total number of n-grams in the translation T.

    More details can be found in `Papineni et al. 2002`__.

    __ https://www.aclweb.org/anthology/P02-1040.pdf

    Args:
        references: list of references R
        candidate: translation T
        n: n-gram order

    Returns:
        The length of the longest common subsequence

    .. versionadded:: 0.4.5
    """
    # ngrams of the candidate
    counts = ngrams(candidate, n)

    # union of ngrams of references
    max_counts: Counter = Counter()
    for reference in references:
        max_counts |= ngrams(reference, n)

    # clipped count of the candidate and references
    clipped_counts = counts & max_counts

    return sum(clipped_counts.values()), sum(counts.values())

