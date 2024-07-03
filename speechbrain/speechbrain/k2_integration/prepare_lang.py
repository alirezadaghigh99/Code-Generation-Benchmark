def add_disambig_symbols(lexicon: Lexicon) -> Tuple[Lexicon, int]:
    """
    It adds pseudo-token disambiguation symbols #1, #2 and so on
    at the ends of tokens to ensure that all pronunciations are different,
    and that none is a prefix of another.

    See also add_lex_disambig.pl from kaldi.

    Arguments
    ---------
    lexicon: Lexicon
        It is returned by :func:`read_lexicon`.

    Returns
    -------
    ans:
        The output lexicon with disambiguation symbols
    max_disambig:
        The ID of the max disambiguation symbol that appears
        in the lexicon
    """

    # (1) Work out the count of each token-sequence in the
    # lexicon.
    count = defaultdict(int)
    for _, tokens in lexicon:
        count[" ".join(tokens)] += 1

    # (2) For each left sub-sequence of each token-sequence, note down
    # that it exists (for identifying prefixes of longer strings).
    issubseq = defaultdict(int)
    for _, tokens in lexicon:
        tokens = tokens.copy()
        tokens.pop()
        while tokens:
            issubseq[" ".join(tokens)] = 1
            tokens.pop()

    # (3) For each entry in the lexicon:
    # if the token sequence is unique and is not a
    # prefix of another word, no disambig symbol.
    # Else output #1, or #2, #3, ... if the same token-seq
    # has already been assigned a disambig symbol.
    ans = []

    # We start with #1 since #0 has its own purpose
    first_allowed_disambig = 1
    max_disambig = first_allowed_disambig - 1
    last_used_disambig_symbol_of = defaultdict(int)

    for word, tokens in lexicon:
        tokenseq = " ".join(tokens)
        assert tokenseq != ""
        if issubseq[tokenseq] == 0 and count[tokenseq] == 1:
            ans.append((word, tokens))
            continue

        cur_disambig = last_used_disambig_symbol_of[tokenseq]
        if cur_disambig == 0:
            cur_disambig = first_allowed_disambig
        else:
            cur_disambig += 1

        if cur_disambig > max_disambig:
            max_disambig = cur_disambig
        last_used_disambig_symbol_of[tokenseq] = cur_disambig
        tokenseq += f" #{cur_disambig}"
        ans.append((word, tokenseq.split()))
    return ans, max_disambigdef get_tokens(
    lexicon: Lexicon, sil_token="SIL", manually_add_sil_to_tokens=False
) -> List[str]:
    """
    Get tokens from a lexicon.

    Arguments
    ---------
    lexicon: Lexicon
        It is the return value of :func:`read_lexicon`.
    sil_token: str
        The optional silence token between words. It should not appear in the lexicon,
        otherwise it will cause an error.
    manually_add_sil_to_tokens: bool
        If true, add `sil_token` to the tokens. This is useful when the lexicon
        does not contain `sil_token` but it is needed in the tokens.

    Returns
    -------
    sorted_ans: List[str]
        A list of unique tokens.
    """
    ans = set()
    if manually_add_sil_to_tokens:
        ans.add(sil_token)
    for _, tokens in lexicon:
        assert (
            sil_token not in tokens
        ), f"{sil_token} should not appear in the lexicon but it is found in {_}"
        ans.update(tokens)
    sorted_ans = sorted(list(ans))
    return sorted_ans