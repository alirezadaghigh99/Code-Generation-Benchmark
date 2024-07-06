def build_treebank(trees, transition_scheme=TransitionScheme.TOP_DOWN_UNARY, reverse=False):
    """
    Turn each of the trees in the treebank into a list of transitions based on the TransitionScheme
    """
    if reverse:
        return [build_sequence(tree.reverse(), transition_scheme) for tree in trees]
    else:
        return [build_sequence(tree, transition_scheme) for tree in trees]

