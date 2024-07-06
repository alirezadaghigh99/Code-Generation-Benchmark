def reassemble_doc_from_tokens(tokens, mwts, expansions, raw_text):
    """Assemble a Stanza document list format from a list of string tokens, calculating offsets as needed.

    Parameters
    ----------
    tokens : List[List[str]]
        A list of sentences, which includes string tokens.
    mwts : List[List[bool]]
        Whether or not each of the tokens are MWTs to be analyzed by the MWT system.
    expansions : List[List[Optional[List[str]]]]
        A list of possible expansions for MWTs, or None if no user-defined expansion
        is given.
    parser_text : str
        The raw text off of which we can compare offsets.

    Returns
    -------
    List[List[Dict]]
        List of words and their offsets, used as `doc`.
    """

    # oov count and offset stays the same; doc gets regenerated
    new_offset = 0
    corrected_doc = []

    for sent_words, sent_mwts, sent_expansions in zip(tokens, mwts, expansions):
        sentence_doc = []

        for indx, (word, mwt, expansion) in enumerate(zip(sent_words, sent_mwts, sent_expansions)):
            try:
                offset_index = raw_text.index(word, new_offset)
            except ValueError as e:
                sub_start = max(0, new_offset - 20)
                sub_end = min(len(raw_text), new_offset + 20)
                sub = raw_text[sub_start:sub_end]
                raise ValueError("Could not find word |%s| starting from char_offset %d.  Surrounding text: |%s|. \n Hint: did you accidentally add/subtract a symbol/character such as a space when combining tokens?" % (word, new_offset, sub)) from e

            wd = {
                "id": (indx+1,), "text": word,
                "start_char":  offset_index,
                "end_char":    offset_index+len(word)
            }
            if expansion:
                wd["manual_expansion"] = True
            elif mwt:
                wd["misc"] = "MWT=Yes"

            sentence_doc.append(wd)

            # start the next search after the previous word ended
            new_offset = offset_index+len(word)

        corrected_doc.append(sentence_doc)

    # use the built in MWT system to expand MWTs
    doc = Document(corrected_doc, raw_text)
    doc.set_mwt_expansions([j
                            for i in expansions
                            for j in i if j],
                           process_manual_expanded=True)
    return doc.to_dict()

