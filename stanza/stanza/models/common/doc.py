def set(self, fields, contents, to_token=False, to_sentence=False):
        """Set fields based on contents. If only one field (string or
        singleton list) is provided, then a list of content will be
        expected; otherwise a list of list of contents will be expected.

        Args:
            fields: name of the fields as a list or a single string
            contents: field values to set; total length should be equal to number of words/tokens
            to_token: if True, set field values to tokens; otherwise to words

        """
        if isinstance(fields, str):
            fields = [fields]
        assert isinstance(fields, (tuple, list)), "Must provide field names as a list."
        assert isinstance(contents, (tuple, list)), "Must provide contents as a list (one item per line)."
        assert len(fields) >= 1, "Must have at least one field."

        assert not to_sentence or not to_token, "Both to_token and to_sentence set to True, which is very confusing"

        if to_sentence:
            assert len(self.sentences) == len(contents), \
                "Contents must have the same length as the sentences"
            for sentence, content in zip(self.sentences, contents):
                if len(fields) == 1:
                    setattr(sentence, fields[0], content)
                else:
                    for field, piece in zip(fields, content):
                        setattr(sentence, field, piece)
        else:
            assert (to_token and self.num_tokens == len(contents)) or self.num_words == len(contents), \
                "Contents must have the same length as the original file."

            cidx = 0
            for sentence in self.sentences:
                # decide word or token
                if to_token:
                    units = sentence.tokens
                else:
                    units = sentence.words
                for unit in units:
                    if len(fields) == 1:
                        setattr(unit, fields[0], contents[cidx])
                    else:
                        for field, content in zip(fields, contents[cidx]):
                            setattr(unit, field, content)
                    cidx += 1

