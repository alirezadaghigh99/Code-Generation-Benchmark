def read_dataset(dataset, wordvec_type: WVType, min_len: int) -> List[SentimentDatum]:
    """
    returns a list where the values of the list are
      label, [token...]
    """
    lines = []
    for filename in str(dataset).split(","):
        with open(filename, encoding="utf-8") as fin:
            new_lines = json.load(fin)
        new_lines = [(str(x['sentiment']), x['text'], x.get('constituency', None)) for x in new_lines]
        lines.extend(new_lines)
    # TODO: maybe do this processing later, once the model is built.
    # then move the processing into the model so we can use
    # overloading to potentially make future model types
    lines = [SentimentDatum(x[0], update_text(x[1], wordvec_type), tree_reader.read_trees(x[2])[0] if x[2] else None) for x in lines]
    if min_len:
        lines = [x for x in lines if len(x.text) >= min_len]
    return lines

