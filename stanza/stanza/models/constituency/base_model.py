class SimpleModel(BaseModel):
    """
    This model allows pushing and popping with no extra data

    This class is primarily used for testing various operations which
    don't need the NN's weights

    Also, for rebuilding trees from transitions when verifying the
    transitions in situations where the NN state is not relevant,
    as this class will be faster than using the NN
    """
    def __init__(self, transition_scheme=TransitionScheme.TOP_DOWN_UNARY, unary_limit=UNARY_LIMIT, reverse_sentence=False, root_labels=("ROOT",)):
        super().__init__(transition_scheme=transition_scheme, unary_limit=unary_limit, reverse_sentence=reverse_sentence, root_labels=root_labels)

    def initial_word_queues(self, tagged_word_lists):
        word_queues = []
        for tagged_words in tagged_word_lists:
            word_queue =  [None]
            word_queue += [tag_node for tag_node in tagged_words]
            word_queue.append(None)
            if self.reverse_sentence:
                word_queue.reverse()
            word_queues.append(word_queue)
        return word_queues

    def initial_transitions(self):
        return TreeStack(value=None, parent=None, length=1)

    def initial_constituents(self):
        return TreeStack(value=None, parent=None, length=1)

    def get_word(self, word_node):
        return word_node

    def transform_word_to_constituent(self, state):
        return state.get_word(state.word_position)

    def dummy_constituent(self, dummy):
        return dummy

    def build_constituents(self, labels, children_lists):
        constituents = []
        for label, children in zip(labels, children_lists):
            if isinstance(label, str):
                label = (label,)
            for value in reversed(label):
                children = Tree(label=value, children=children)
            constituents.append(children)
        return constituents

    def push_constituents(self, constituent_stacks, constituents):
        return [stack.push(constituent) for stack, constituent in zip(constituent_stacks, constituents)]

    def get_top_constituent(self, constituents):
        return constituents.value

    def push_transitions(self, transition_stacks, transitions):
        return [stack.push(transition) for stack, transition in zip(transition_stacks, transitions)]

    def get_top_transition(self, transitions):
        return transitions.value

