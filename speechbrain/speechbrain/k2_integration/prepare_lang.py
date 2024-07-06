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
    return ans, max_disambig

def get_tokens(
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

def prepare_lang(lang_dir, sil_token="SIL", sil_prob=0.5, cache=True):
    """
    This function takes as input a lexicon file "$lang_dir/lexicon.txt"
    consisting of words and tokens (i.e., phones) and does the following:

    1. Add disambiguation symbols to the lexicon and generate lexicon_disambig.txt

    2. Generate tokens.txt, the token table mapping a token to a unique integer.

    3. Generate words.txt, the word table mapping a word to a unique integer.

    4. Generate L.pt, in k2 format. It can be loaded by

            d = torch.load("L.pt")
            lexicon = k2.Fsa.from_dict(d)

    5. Generate L_disambig.pt, in k2 format.


    Arguments
    ---------
    lang_dir: str
        The directory to store the output files and read the input file lexicon.txt.
    sil_token: str
        The silence token. Default is "SIL".
    sil_prob: float
        The probability for adding a silence at the beginning and end of the word.
        Default is 0.5.
    cache: bool
        Whether or not to load/cache from/to the .pt format.

    Returns
    -------
    None

    Example
    -------
    >>> from speechbrain.k2_integration.prepare_lang import prepare_lang

    >>> # Create a small lexicon containing only two words and write it to a file.
    >>> lang_tmpdir = getfixture('tmpdir')
    >>> lexicon_sample = '''hello h e l l o\\nworld w o r l d'''
    >>> lexicon_file = lang_tmpdir.join("lexicon.txt")
    >>> lexicon_file.write(lexicon_sample)

    >>> prepare_lang(lang_tmpdir)
    >>> for expected_file in ["tokens.txt", "words.txt", "L.pt", "L_disambig.pt", "Linv.pt" ]:
    ...     assert os.path.exists(os.path.join(lang_tmpdir, expected_file))
    """

    out_dir = Path(lang_dir)
    lexicon_filename = out_dir / "lexicon.txt"

    # if source lexicon_filename has been re-created (only use 'Linv.pt' for date modification query)
    if (
        cache
        and (out_dir / "Linv.pt").exists()
        and (out_dir / "Linv.pt").stat().st_mtime
        < lexicon_filename.stat().st_mtime
    ):
        logger.warning(
            f"Skipping lang preparation of '{out_dir}'."
            " Set 'caching: False' in the yaml"
            " if this is not what you want."
        )
        return

    # backup L.pt, L_disambig.pt, tokens.txt and words.txt, Linv.pt and lexicon_disambig.txt
    for f in [
        "L.pt",
        "L_disambig.pt",
        "tokens.txt",
        "words.txt",
        "Linv.pt",
        "lexicon_disambig.txt",
    ]:
        if (out_dir / f).exists():
            os.makedirs(out_dir / "backup", exist_ok=True)
            logger.debug(f"Backing up {out_dir / f} to {out_dir}/backup/{f}")
            os.rename(out_dir / f, out_dir / "backup" / f)

    lexicon = read_lexicon(str(lexicon_filename))
    if sil_prob != 0:
        # add silence to the tokens
        tokens = get_tokens(
            lexicon, sil_token=sil_token, manually_add_sil_to_tokens=True
        )
    else:
        tokens = get_tokens(lexicon, manually_add_sil_to_tokens=False)
    words = get_words(lexicon)

    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    for i in range(max_disambig + 1):
        disambig = f"#{i}"
        assert disambig not in tokens
        tokens.append(f"#{i}")

    assert EPS not in tokens
    tokens = [EPS] + tokens

    assert EPS not in words
    assert "#0" not in words
    assert "<s>" not in words
    assert "</s>" not in words

    words = [EPS] + words + ["#0", "<s>", "</s>"]

    token2id = generate_id_map(tokens)
    word2id = generate_id_map(words)

    logger.info(
        f"Saving tokens.txt, words.txt, lexicon_disambig.txt to '{out_dir}'"
    )
    write_mapping(out_dir / "tokens.txt", token2id)
    write_mapping(out_dir / "words.txt", word2id)
    write_lexicon(out_dir / "lexicon_disambig.txt", lexicon_disambig)

    if sil_prob != 0:
        L = lexicon_to_fst(
            lexicon,
            token2id=token2id,
            word2id=word2id,
            sil_token=sil_token,
            sil_prob=sil_prob,
        )
    else:
        L = lexicon_to_fst_no_sil(
            lexicon,
            token2id=token2id,
            word2id=word2id,
        )

    if sil_prob != 0:
        L_disambig = lexicon_to_fst(
            lexicon_disambig,
            token2id=token2id,
            word2id=word2id,
            sil_token=sil_token,
            sil_prob=sil_prob,
            need_self_loops=True,
        )
    else:
        L_disambig = lexicon_to_fst_no_sil(
            lexicon_disambig,
            token2id=token2id,
            word2id=word2id,
            need_self_loops=True,
        )

    L_inv = k2.arc_sort(L.invert())
    logger.info(f"Saving L.pt, Linv.pt, L_disambig.pt to '{out_dir}'")
    torch.save(L.as_dict(), out_dir / "L.pt")
    torch.save(L_disambig.as_dict(), out_dir / "L_disambig.pt")
    torch.save(L_inv.as_dict(), out_dir / "Linv.pt")

