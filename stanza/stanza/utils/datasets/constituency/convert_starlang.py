def read_tree(text):
    """
    Reads in a tree, then extracts specifically the word from the specific format used

    Also converts LCB/RCB as needed
    """
    trees = tree_reader.read_trees(text)
    if len(trees) > 1:
        raise ValueError("Tree file had two trees!")
    tree = trees[0]
    labels = tree.leaf_labels()
    new_labels = []
    for label in labels:
        match = TURKISH_RE.search(label)
        if match is None:
            raise ValueError("Could not find word in |{}|".format(label))
        word = match.group(1)
        word = word.replace("-LCB-", "{").replace("-RCB-", "}")
        new_labels.append(word)

    tree = tree.replace_words(new_labels)
    #tree = tree.remap_constituent_labels(LABEL_MAP)
    con_labels = tree.get_unique_constituent_labels([tree])
    if any(label in DISALLOWED_LABELS for label in con_labels):
        raise ValueError("found an unexpected phrasal node {}".format(label))
    return tree

