def read_document(lines, spaces_after, split_clauses):
    document = []
    sentence = []
    for line in lines:
        line = line.strip()
        if not line:
            if sentence:
                if spaces_after:
                    sentence[-1] = (sentence[-1][0], True)
                document.append(sentence)
                sentence = []
        else:
            pieces = line.split("\t")
            # there are some nbsp in tokens in lst20, but the downstream tools expect spaces
            pieces = [p.replace("\xa0", " ") for p in pieces]
            if split_clauses and pieces[0] == '_' and pieces[3] == 'O':
                if sentence:
                    # note that we don't need to check spaces_after
                    # the "token" is a space anyway
                    sentence[-1] = (sentence[-1][0], True)
                    document.append(sentence)
                    sentence = []
            elif pieces[0] == '_':
                sentence[-1] = (sentence[-1][0], True)
            else:
                sentence.append((pieces[0], False))

    if sentence:
        if spaces_after:
            sentence[-1] = (sentence[-1][0], True)
        document.append(sentence)
        sentence = []
    # TODO: is there any way to divide up a single document into paragraphs?
    return [[document]]

