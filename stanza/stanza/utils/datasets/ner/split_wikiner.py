def read_sentences(filename, encoding):
    sents = []
    cache = []
    skipped = 0
    skip = False
    with open(filename, encoding=encoding) as infile:
        for i, line in enumerate(infile):
            line = line.rstrip()
            if len(line) == 0:
                if len(cache) > 0:
                    if not skip:
                        sents.append(cache)
                    else:
                        skipped += 1
                        skip = False
                    cache = []
                continue
            array = line.split()
            if len(array) != 2:
                skip = True
                continue
            #assert len(array) == 2, "Format error at line {}: {}".format(i+1, line)
            w, t = array
            cache.append([w, t])
        if len(cache) > 0:
            if not skip:
                sents.append(cache)
            else:
                skipped += 1
            cache = []
    print("Skipped {} examples due to formatting issues.".format(skipped))
    return sents

