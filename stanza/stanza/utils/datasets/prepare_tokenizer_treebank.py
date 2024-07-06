def augment_initial_punct(sents, ratio=0.20):
    """
    If a sentence starts with certain punct marks, occasionally use the same sentence without the initial punct.

    Currently this just handles ¿
    This helps languages such as CA and ES where the models go awry when the initial ¿ is missing.
    """
    new_sents = []
    for sent in sents:
        if random.random() > ratio:
            continue

        text_idx = find_text_idx(sent)
        text_line = sent[text_idx]
        if text_line.count("¿") != 1:
            # only handle sentences with exactly one ¿
            continue

        # find the first line with actual text
        for idx, line in enumerate(sent):
            if line.startswith("#"):
                continue
            break
        if idx >= len(sent) - 1:
            raise ValueError("Unexpectedly an entire sentence is comments")
        pieces = line.split("\t")
        if pieces[1] != '¿':
            continue
        if has_space_after_no(pieces[-1]):
            replace_text = "¿"
        else:
            replace_text = "¿ "

        new_sent = sent[:idx] + sent[idx+1:]
        new_sent[text_idx] = text_line.replace(replace_text, "")

        # now need to update all indices
        new_sent = [change_indices(x, -1) for x in new_sent]
        new_sents.append(new_sent)

    if len(new_sents) > 0:
        print("Added %d sentences with the leading ¿ removed" % len(new_sents))

    return sents + new_sents

