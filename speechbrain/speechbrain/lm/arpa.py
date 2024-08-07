def read_arpa(fstream):
    r"""
    Reads an ARPA format N-gram language model from a stream

    Arguments
    ---------
    fstream : TextIO
        Text file stream (as commonly returned by open()) to read the model
        from.

    Returns
    -------
    dict
        Maps N-gram orders to the number ngrams of that order. Essentially the
        \data\ section of an ARPA format file.
    dict
        The log probabilities (first column) in the ARPA file.
        This is a triply nested dict.
        The first layer is indexed by N-gram order (integer).
        The second layer is indexed by the context (tuple of tokens).
        The third layer is indexed by tokens, and maps to the log prob.
        This format is compatible with `speechbrain.lm.ngram.BackoffNGramLM`
        Example:
        In ARPA format, log(P(fox|a quick red)) = -5.3 is expressed:
            `-5.3 a quick red fox`
        And to access that probability, use:
            `ngrams_by_order[4][('a', 'quick', 'red')]['fox']`
    dict
        The log backoff weights (last column) in the ARPA file.
        This is a doubly nested dict.
        The first layer is indexed by N-gram order (integer).
        The second layer is indexed by the backoff history (tuple of tokens)
        i.e. the context on which the probability distribution is conditioned
        on. This maps to the log weights.
        This format is compatible with `speechbrain.lm.ngram.BackoffNGramLM`
        Example:
        If log(P(fox|a quick red)) is not listed, we find
        log(backoff(a quick red)) = -23.4 which in ARPA format is:
            `<logp> a quick red -23.4`
        And to access that here, use:
            `backoffs_by_order[3][('a', 'quick', 'red')]`

    Raises
    ------
    ValueError
        If no LM is found or the file is badly formatted.
    """
    # Developer's note:
    # This is a long function.
    # It is because we support cases where a new section starts suddenly without
    # an empty line in between.
    #
    # \data\ section:
    _find_data_section(fstream)
    num_ngrams = {}
    for line in fstream:
        line = line.strip()
        if line[:5] == "ngram":
            lhs, rhs = line.split("=")
            order = int(lhs.split()[1])
            num_grams = int(rhs)
            num_ngrams[order] = num_grams
        elif not line:  # Normal case, empty line ends section
            ended, order = _next_section_or_end(fstream)
            break  # Good, proceed to next section
        elif _starts_ngrams_section(line):  # No empty line between sections
            ended = False
            order = _parse_order(line)
            break  # Good, proceed to next section
        else:
            raise ValueError("Not a properly formatted line")
    # At this point:
    # ended == False
    # type(order) == int
    #
    # \N-grams: sections
    # NOTE: This is the section that most time is spent on, so it's been written
    # with processing speed in mind.
    ngrams_by_order = {}
    backoffs_by_order = {}
    while not ended:
        probs = collections.defaultdict(dict)
        backoffs = {}
        backoff_line_length = order + 2
        # Use try-except because it is faster than always checking
        try:
            for line in fstream:
                line = line.strip()
                all_parts = tuple(line.split())
                prob = float(all_parts[0])
                if len(all_parts) == backoff_line_length:
                    context = all_parts[1:-2]
                    token = all_parts[-2]
                    backoff = float(all_parts[-1])
                    backoff_context = context + (token,)
                    backoffs[backoff_context] = backoff
                else:
                    context = all_parts[1:-1]
                    token = all_parts[-1]
                probs[context][token] = prob
        except (IndexError, ValueError):
            ngrams_by_order[order] = probs
            backoffs_by_order[order] = backoffs
            if not line:  # Normal case, empty line ends section
                ended, order = _next_section_or_end(fstream)
            elif _starts_ngrams_section(line):  # No empty line between sections
                ended = False
                order = _parse_order(line)
            elif _ends_arpa(line):  # No empty line before End of file
                ended = True
                order = None
            else:
                raise ValueError("Not a properly formatted ARPA file")
    # Got to the \end\. Still have to check whether all promised sections were
    # delivered.
    if not num_ngrams.keys() == ngrams_by_order.keys():
        raise ValueError("Not a properly formatted ARPA file")
    return num_ngrams, ngrams_by_order, backoffs_by_order

