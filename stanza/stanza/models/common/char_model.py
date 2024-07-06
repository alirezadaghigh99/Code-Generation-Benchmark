def build_charlm_vocab(path, cutoff=0):
    """
    Build a vocab for a CharacterLanguageModel

    Requires a large amount of memory, but only need to build once

    here we need some trick to deal with excessively large files
    for each file we accumulate the counter of characters, and
    at the end we simply pass a list of chars to the vocab builder
    """
    counter = Counter()
    if os.path.isdir(path):
        filenames = sorted(os.listdir(path))
    else:
        filenames = [os.path.split(path)[1]]
        path = os.path.split(path)[0]

    for filename in filenames:
        filename = os.path.join(path, filename)
        with open_read_text(filename) as fin:
            for line in fin:
                counter.update(list(line))

    if len(counter) == 0:
        raise ValueError("Training data was empty!")
    # remove infrequent characters from vocab
    for k in list(counter.keys()):
        if counter[k] < cutoff:
            del counter[k]
    # a singleton list of all characters
    data = [sorted([x[0] for x in counter.most_common()])]
    if len(data[0]) == 0:
        raise ValueError("All characters in the training data were less frequent than --cutoff!")
    vocab = CharVocab(data) # skip cutoff argument because this has been dealt with
    return vocab

