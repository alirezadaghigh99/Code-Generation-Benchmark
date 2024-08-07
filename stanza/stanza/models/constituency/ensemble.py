class Ensemble:
    def __init__(self, filenames, args, foundation_cache=None):
        """
        Loads each model in filenames

        If foundation_cache is None, we build one on our own,
        as the expectation is the models will reuse modules
        such as pretrain, charlm, bert
        """
        if foundation_cache is None:
            foundation_cache = FoundationCache()

        if isinstance(filenames, str):
            filenames = [filenames]
        logger.info("Models used for ensemble:\n  %s", "\n  ".join(filenames))
        self.models = [Trainer.load(filename, args, load_optimizer=False, foundation_cache=foundation_cache).model for filename in filenames]

        for model_idx, model in enumerate(self.models):
            if self.models[0].transition_scheme() != model.transition_scheme():
                raise ValueError("Models {} and {} are incompatible.  {} vs {}".format(filenames[0], filenames[model_idx], self.models[0].transition_scheme(), model.transition_scheme()))
            if self.models[0].transitions != model.transitions:
                raise ValueError("Models %s and %s are incompatible: different transitions" % (filenames[0], filenames[model_idx]))
            if self.models[0].constituents != model.constituents:
                raise ValueError("Models %s and %s are incompatible: different constituents" % (filenames[0], filenames[model_idx]))
            if self.models[0].root_labels != model.root_labels:
                raise ValueError("Models %s and %s are incompatible: different root_labels" % (filenames[0], filenames[model_idx]))
            if self.models[0].uses_xpos() != model.uses_xpos():
                raise ValueError("Models %s and %s are incompatible: different uses_xpos" % (filenames[0], filenames[model_idx]))
            if self.models[0].reverse_sentence != model.reverse_sentence:
                raise ValueError("Models %s and %s are incompatible: different reverse_sentence" % (filenames[0], filenames[model_idx]))

        self._reverse_sentence = self.models[0].reverse_sentence

    def eval(self):
        for model in self.models:
            model.eval()

    @property
    def reverse_sentence(self):
        return self._reverse_sentence

    def uses_xpos(self):
        return self.models[0].uses_xpos()

    def build_batch_from_tagged_words(self, batch_size, data_iterator):
        """
        Read from the data_iterator batch_size tagged sentences and turn them into new parsing states

        Expects a list of list of (word, tag)
        """
        state_batch = []
        for _ in range(batch_size):
            sentence = next(data_iterator, None)
            if sentence is None:
                break
            state_batch.append(sentence)

        if len(state_batch) > 0:
            state_batch = [model.initial_state_from_words(state_batch) for model in self.models]
            state_batch = list(zip(*state_batch))
            state_batch = [MultiState(states, None, None, 0.0) for states in state_batch]
        return state_batch

    def build_batch_from_trees(self, batch_size, data_iterator):
        """
        Read from the data_iterator batch_size trees and turn them into N lists of parsing states
        """
        state_batch = []
        for _ in range(batch_size):
            gold_tree = next(data_iterator, None)
            if gold_tree is None:
                break
            state_batch.append(gold_tree)

        if len(state_batch) > 0:
            state_batch = [model.initial_state_from_gold_trees(state_batch) for model in self.models]
            state_batch = list(zip(*state_batch))
            state_batch = [MultiState(states, None, None, 0.0) for states in state_batch]
        return state_batch

    def predict(self, states, is_legal=True):
        states = list(zip(*[x.states for x in states]))
        predictions = [model.forward(state_batch) for model, state_batch in zip(self.models, states)]
        predictions = torch.stack(predictions)
        predictions = torch.sum(predictions, dim=0)

        model = self.models[0]

        # TODO: possibly refactor with lstm_model.predict
        pred_max = torch.argmax(predictions, dim=1)
        scores = torch.take_along_dim(predictions, pred_max.unsqueeze(1), dim=1)
        pred_max = pred_max.detach().cpu()

        pred_trans = [model.transitions[pred_max[idx]] for idx in range(len(states[0]))]
        if is_legal:
            for idx, (state, trans) in enumerate(zip(states[0], pred_trans)):
                if not trans.is_legal(state, model):
                    _, indices = predictions[idx, :].sort(descending=True)
                    for index in indices:
                        if model.transitions[index].is_legal(state, model):
                            pred_trans[idx] = model.transitions[index]
                            scores[idx] = predictions[idx, index]
                            break
                    else: # yeah, else on a for loop, deal with it
                        pred_trans[idx] = None
                        scores[idx] = None

        return predictions, pred_trans, scores.squeeze(1)

    def bulk_apply(self, state_batch, transitions, fail=False):
        new_states = []

        states = list(zip(*[x.states for x in state_batch]))
        states = [x.bulk_apply(y, transitions, fail=fail) for x, y in zip(self.models, states)]
        states = list(zip(*states))
        state_batch = [x._replace(states=y) for x, y in zip(state_batch, states)]
        return state_batch

    def parse_sentences(self, data_iterator, build_batch_fn, batch_size, transition_choice, keep_state=False, keep_constituents=False, keep_scores=False):
        """
        Repeat transitions to build a list of trees from the input batches.

        The data_iterator should be anything which returns the data for a parse task via next()
        build_batch_fn is a function that turns that data into State objects
        This will be called to generate batches of size batch_size until the data is exhausted

        The return is a list of tuples: (gold_tree, [(predicted, score) ...])
        gold_tree will be left blank if the data did not include gold trees
        currently score is always 1.0, but the interface may be expanded
        to get a score from the result of the parsing

        transition_choice: which method of the model to use for
        choosing the next transition

        TODO: refactor with base_model
        """
        treebank = []
        treebank_indices = []
        # this will produce tuples of states
        # batch size lists of num models tuples
        state_batch = build_batch_fn(batch_size, data_iterator)
        batch_indices = list(range(len(state_batch)))
        horizon_iterator = iter([])

        if keep_constituents:
            constituents = defaultdict(list)

        while len(state_batch) > 0:
            pred_scores, transitions, scores = transition_choice(state_batch)
            # num models lists of batch size states
            state_batch = self.bulk_apply(state_batch, transitions)

            remove = set()
            for idx, states in enumerate(state_batch):
                if states.finished(self):
                    predicted_tree = states.get_tree(self)
                    if self.reverse_sentence:
                        predicted_tree = predicted_tree.reverse()
                    gold_tree = states.gold_tree
                    # TODO: could easily store the score here
                    # not sure what it means to store the state,
                    # since each model is tracking its own state
                    treebank.append(ParseResult(gold_tree, [ScoredTree(predicted_tree, None)], None, None))
                    treebank_indices.append(batch_indices[idx])
                    remove.add(idx)

            if len(remove) > 0:
                state_batch = [state for idx, state in enumerate(state_batch) if idx not in remove]
                batch_indices = [batch_idx for idx, batch_idx in enumerate(batch_indices) if idx not in remove]

            for _ in range(batch_size - len(state_batch)):
                horizon_state = next(horizon_iterator, None)
                if not horizon_state:
                    horizon_batch = build_batch_fn(batch_size, data_iterator)
                    if len(horizon_batch) == 0:
                        break
                    horizon_iterator = iter(horizon_batch)
                    horizon_state = next(horizon_iterator, None)

                state_batch.append(horizon_state)
                batch_indices.append(len(treebank) + len(state_batch))

        treebank = utils.unsort(treebank, treebank_indices)
        return treebank

    def parse_sentences_no_grad(self, data_iterator, build_batch_fn, batch_size, transition_choice, keep_state=False, keep_constituents=False, keep_scores=False):
        with torch.no_grad():
            return self.parse_sentences(data_iterator, build_batch_fn, batch_size, transition_choice, keep_state, keep_constituents, keep_scores)

