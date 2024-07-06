def read_trees(text, broken_ok=False, tree_callback=None, use_tqdm=True):
    """
    Reads multiple trees from the text

    TODO: some of the error cases we hit can be recovered from
    """
    token_iterator = TextTokenIterator(text, use_tqdm)
    return read_token_iterator(token_iterator, broken_ok=broken_ok, tree_callback=tree_callback)

