def tarjan(tree):
    """Finds the cycles in a dependency graph

    The input should be a numpy array of integers,
    where in the standard use case,
    tree[i] is the head of node i.

    tree[0] == 0 to represent the root

    so for example, for the English sentence "This is a test",
    the input is

    [0 4 4 4 0]

    "Arthritis makes my hip hurt"

    [0 2 0 4 2 2]

    The return is a list of cycles, where in cycle has True if the
    node at that index is participating in the cycle.
    So, for example, the previous examples both return empty lists,
    whereas an input of
      np.array([0, 3, 1, 2])
    has an output of
      [np.array([False,  True,  True,  True])]
    """
    indices = -np.ones_like(tree)
    lowlinks = -np.ones_like(tree)
    onstack = np.zeros_like(tree, dtype=bool)
    stack = list()
    _index = [0]
    cycles = []
    #-------------------------------------------------------------
    def maybe_pop_cycle(i):
        if lowlinks[i] == indices[i]:
            # There's a cycle!
            cycle = np.zeros_like(indices, dtype=bool)
            while stack[-1] != i:
                j = stack.pop()
                onstack[j] = False
                cycle[j] = True
            stack.pop()
            onstack[i] = False
            cycle[i] = True
            if cycle.sum() > 1:
                cycles.append(cycle)

    def initialize_strong_connect(i):
        _index[0] += 1
        index = _index[-1]
        indices[i] = lowlinks[i] = index - 1
        stack.append(i)
        onstack[i] = True

    def strong_connect(i):
        # this ridiculous atrocity is because somehow people keep
        # coming up with graphs which overflow python's call stack
        # so instead we make our own call stack and turn the recursion
        # into a loop
        # see for example
        #   https://github.com/stanfordnlp/stanza/issues/962
        #   https://github.com/spraakbanken/sparv-pipeline/issues/166
        # in an ideal world this block of code would look like this
        #    initialize_strong_connect(i)
        #    dependents = iter(np.where(np.equal(tree, i))[0])
        #    for j in dependents:
        #        if indices[j] == -1:
        #            strong_connect(j)
        #            lowlinks[i] = min(lowlinks[i], lowlinks[j])
        #        elif onstack[j]:
        #            lowlinks[i] = min(lowlinks[i], indices[j])
        #
        #     maybe_pop_cycle(i)
        call_stack = [(i, None, None)]
        while len(call_stack) > 0:
            i, dependents_iterator, j = call_stack.pop()
            if dependents_iterator is None: # first time getting here for this i
                initialize_strong_connect(i)
                dependents_iterator = iter(np.where(np.equal(tree, i))[0])
            else: # been here before.  j was the dependent we were just considering
                lowlinks[i] = min(lowlinks[i], lowlinks[j])
            for j in dependents_iterator:
                if indices[j] == -1:
                    # have to remember where we were...
                    # put the current iterator & its state on the "call stack"
                    # we will come back to it later
                    call_stack.append((i, dependents_iterator, j))
                    # also, this is what we do next...
                    call_stack.append((j, None, None))
                    # this will break this iterator for now
                    # the next time through, we will continue progressing this iterator
                    break
                elif onstack[j]:
                    lowlinks[i] = min(lowlinks[i], indices[j])
            else:
                # this is an intended use of for/else
                # please stop filing git issues on obscure language features
                # we finished iterating without a break
                # and can finally resolve any possible cycles
                maybe_pop_cycle(i)
            # at this point, there are two cases:
            #
            # we iterated all the way through an iterator (the else in the for/else)
            # and have resolved any possible cycles.  can then proceed to the previous
            # iterator we were considering (or finish, if there are no others)
            # OR
            # we have hit a break in the iteration over the dependents
            # for a node
            # and we need to dig deeper into the graph and resolve the dependent's dependents
            # before we can continue the previous node
            #
            # either way, we check to see if there are unfinished subtrees
            # when that is finally done, we can return

    #-------------------------------------------------------------
    for i in range(len(tree)):
        if indices[i] == -1:
            strong_connect(i)
    return cycles

