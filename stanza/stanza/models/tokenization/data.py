class DataLoader(TokenizationDataset):
    """
    This is the training version of the dataset.
    """
    def __init__(self, args, input_files={'txt': None, 'label': None}, input_text=None, vocab=None, evaluation=False, dictionary=None):
        super().__init__(args, input_files, input_text, vocab, evaluation, dictionary)

        self.vocab = vocab if vocab is not None else self.init_vocab()

        # data comes in a list of paragraphs, where each paragraph is a list of units with unit-level labels.
        # At evaluation time, each paragraph is treated as single "sentence" as we don't know a priori where
        # sentence breaks occur. We make prediction from left to right for each paragraph and move forward to
        # the last predicted sentence break to start afresh.
        self.sentences = [self.para_to_sentences(para) for para in self.data]

        self.init_sent_ids()
        logger.debug(f"{len(self.sentence_ids)} sentences loaded.")

    def __len__(self):
        return len(self.sentence_ids)

    def init_vocab(self):
        vocab = Vocab(self.data, self.args['lang'])
        return vocab

    def init_sent_ids(self):
        self.sentence_ids = []
        self.cumlen = [0]
        for i, para in enumerate(self.sentences):
            for j in range(len(para)):
                self.sentence_ids += [(i, j)]
                self.cumlen += [self.cumlen[-1] + len(self.sentences[i][j][0])]

    def has_mwt(self):
        # presumably this only needs to be called either 0 or 1 times,
        # 1 when training and 0 any other time, so no effort is put
        # into caching the result
        for sentence in self.data:
            for word in sentence:
                if word[1] > 2:
                    return True
        return False

    def shuffle(self):
        for para in self.sentences:
            random.shuffle(para)
        self.init_sent_ids()

    def next(self, eval_offsets=None, unit_dropout=0.0, feat_unit_dropout=0.0):
        ''' Get a batch of converted and padded PyTorch data from preprocessed raw text for training/prediction. '''
        feat_size = len(self.sentences[0][0][2][0])
        unkid = self.vocab.unit2id('<UNK>')
        padid = self.vocab.unit2id('<PAD>')

        def strings_starting(id_pair, offset=0, pad_len=self.args['max_seqlen']):
            # At eval time, this combines sentences in paragraph (indexed by id_pair[0]) starting sentence (indexed 
            # by id_pair[1]) into a long string for evaluation. At training time, we just select random sentences
            # from the entire dataset until we reach max_seqlen.
            pid, sid = id_pair if self.eval else random.choice(self.sentence_ids)
            sentences = [copy([x[offset:] for x in self.sentences[pid][sid]])]

            drop_sents = False if self.eval or (self.args.get('sent_drop_prob', 0) == 0) else (random.random() < self.args.get('sent_drop_prob', 0))
            total_len = len(sentences[0][0])

            assert self.eval or total_len <= self.args['max_seqlen'], 'The maximum sequence length {} is less than that of the longest sentence length ({}) in the data, consider increasing it! {}'.format(self.args['max_seqlen'], total_len, ' '.join(["{}/{}".format(*x) for x in zip(self.sentences[pid][sid])]))
            if self.eval:
                for sid1 in range(sid+1, len(self.sentences[pid])):
                    total_len += len(self.sentences[pid][sid1][0])
                    sentences.append(self.sentences[pid][sid1])

                    if total_len >= self.args['max_seqlen']:
                        break
            else:
                while True:
                    pid1, sid1 = random.choice(self.sentence_ids)
                    total_len += len(self.sentences[pid1][sid1][0])
                    sentences.append(self.sentences[pid1][sid1])

                    if total_len >= self.args['max_seqlen']:
                        break

            if drop_sents and len(sentences) > 1:
                if total_len > self.args['max_seqlen']:
                    sentences = sentences[:-1]
                if len(sentences) > 1:
                    p = [.5 ** i for i in range(1, len(sentences) + 1)] # drop a large number of sentences with smaller probability
                    cutoff = random.choices(list(range(len(sentences))), weights=list(reversed(p)))[0]
                    sentences = sentences[:cutoff+1]

            units = np.concatenate([s[0] for s in sentences])
            labels = np.concatenate([s[1] for s in sentences])
            feats = np.concatenate([s[2] for s in sentences])
            raw_units = [x for s in sentences for x in s[3]]

            if not self.eval:
                cutoff = self.args['max_seqlen']
                units, labels, feats, raw_units = units[:cutoff], labels[:cutoff], feats[:cutoff], raw_units[:cutoff]

            return units, labels, feats, raw_units

        if eval_offsets is not None:
            # find max padding length
            pad_len = 0
            for eval_offset in eval_offsets:
                if eval_offset < self.cumlen[-1]:
                    pair_id = bisect_right(self.cumlen, eval_offset) - 1
                    pair = self.sentence_ids[pair_id]
                    pad_len = max(pad_len, len(strings_starting(pair, offset=eval_offset-self.cumlen[pair_id])[0]))

            pad_len += 1
            id_pairs = [bisect_right(self.cumlen, eval_offset) - 1 for eval_offset in eval_offsets]
            pairs = [self.sentence_ids[pair_id] for pair_id in id_pairs]
            offsets = [eval_offset - self.cumlen[pair_id] for eval_offset, pair_id in zip(eval_offsets, id_pairs)]

            offsets_pairs = list(zip(offsets, pairs))
        else:
            id_pairs = random.sample(self.sentence_ids, min(len(self.sentence_ids), self.args['batch_size']))
            offsets_pairs = [(0, x) for x in id_pairs]
            pad_len = self.args['max_seqlen']

        # put everything into padded and nicely shaped NumPy arrays and eventually convert to PyTorch tensors
        units = np.full((len(id_pairs), pad_len), padid, dtype=np.int64)
        labels = np.full((len(id_pairs), pad_len), -1, dtype=np.int64)
        features = np.zeros((len(id_pairs), pad_len, feat_size), dtype=np.float32)
        raw_units = []
        for i, (offset, pair) in enumerate(offsets_pairs):
            u_, l_, f_, r_ = strings_starting(pair, offset=offset, pad_len=pad_len)
            units[i, :len(u_)] = u_
            labels[i, :len(l_)] = l_
            features[i, :len(f_), :] = f_
            raw_units.append(r_ + ['<PAD>'] * (pad_len - len(r_)))

        if unit_dropout > 0 and not self.eval:
            # dropout characters/units at training time and replace them with UNKs
            mask = np.random.random_sample(units.shape) < unit_dropout
            mask[units == padid] = 0
            units[mask] = unkid
            for i in range(len(raw_units)):
                for j in range(len(raw_units[i])):
                    if mask[i, j]:
                        raw_units[i][j] = '<UNK>'

        # dropout unit feature vector in addition to only torch.dropout in the model.
        # experiments showed that only torch.dropout hurts the model
        # we believe it is because the dict feature vector is mostly scarse so it makes
        # more sense to drop out the whole vector instead of only single element.
        if self.args['use_dictionary'] and feat_unit_dropout > 0 and not self.eval:
            mask_feat = np.random.random_sample(units.shape) < feat_unit_dropout
            mask_feat[units == padid] = 0
            for i in range(len(raw_units)):
                for j in range(len(raw_units[i])):
                    if mask_feat[i,j]:
                        features[i,j,:] = 0
                        
        units = torch.from_numpy(units)
        labels = torch.from_numpy(labels)
        features = torch.from_numpy(features)

        return units, labels, features, raw_units

