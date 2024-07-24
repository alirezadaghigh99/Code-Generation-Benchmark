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

class Document(StanzaObject):
    """ A document class that stores attributes of a document and carries a list of sentences.
    """

    def __init__(self, sentences, text=None, comments=None, empty_sentences=None):
        """ Construct a document given a list of sentences in the form of lists of CoNLL-U dicts.

        Args:
            sentences: a list of sentences, which being a list of token entry, in the form of a CoNLL-U dict.
            text: the raw text of the document.
            comments: A list of list of strings to use as comments on the sentences, either None or the same length as sentences
        """
        self._sentences = []
        self._lang = None
        self._text = text
        self._num_tokens = 0
        self._num_words = 0

        self._process_sentences(sentences, comments, empty_sentences)
        self._ents = []
        self._coref = []
        if self._text is not None:
            self.build_ents()
            self.mark_whitespace()

    def mark_whitespace(self):
        for sentence in self._sentences:
            # TODO: pairwise, once we move to minimum 3.10
            for prev_token, next_token in zip(sentence.tokens[:-1], sentence.tokens[1:]):
                whitespace = self._text[prev_token.end_char:next_token.start_char]
                prev_token.spaces_after = whitespace
        for prev_sentence, next_sentence in zip(self._sentences[:-1], self._sentences[1:]):
            prev_token = prev_sentence.tokens[-1]
            next_token = next_sentence.tokens[0]
            whitespace = self._text[prev_token.end_char:next_token.start_char]
            prev_token.spaces_after = whitespace
        if len(self._sentences) > 0 and len(self._sentences[-1].tokens) > 0:
            final_token = self._sentences[-1].tokens[-1]
            whitespace = self._text[final_token.end_char:]
            final_token.spaces_after = whitespace
        if len(self._sentences) > 0 and len(self._sentences[0].tokens) > 0:
            first_token = self._sentences[0].tokens[0]
            whitespace = self._text[:first_token.start_char]
            first_token.spaces_before = whitespace


    @property
    def lang(self):
        """ Access the language of this document """
        return self._lang

    @lang.setter
    def lang(self, value):
        """ Set the language of this document """
        self._lang = value

    @property
    def text(self):
        """ Access the raw text for this document. """
        return self._text

    @text.setter
    def text(self, value):
        """ Set the raw text for this document. """
        self._text = value

    @property
    def sentences(self):
        """ Access the list of sentences for this document. """
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        """ Set the list of tokens for this document. """
        self._sentences = value

    @property
    def num_tokens(self):
        """ Access the number of tokens for this document. """
        return self._num_tokens

    @num_tokens.setter
    def num_tokens(self, value):
        """ Set the number of tokens for this document. """
        self._num_tokens = value

    @property
    def num_words(self):
        """ Access the number of words for this document. """
        return self._num_words

    @num_words.setter
    def num_words(self, value):
        """ Set the number of words for this document. """
        self._num_words = value

    @property
    def ents(self):
        """ Access the list of entities in this document. """
        return self._ents

    @ents.setter
    def ents(self, value):
        """ Set the list of entities in this document. """
        self._ents = value

    @property
    def entities(self):
        """ Access the list of entities. This is just an alias of `ents`. """
        return self._ents

    @entities.setter
    def entities(self, value):
        """ Set the list of entities in this document. """
        self._ents = value

    def _process_sentences(self, sentences, comments=None, empty_sentences=None):
        self.sentences = []
        if empty_sentences is None:
            empty_sentences = repeat([])
        for sent_idx, (tokens, empty_words) in enumerate(zip(sentences, empty_sentences)):
            try:
                sentence = Sentence(tokens, doc=self, empty_words=empty_words)
            except IndexError as e:
                raise IndexError("Could not process document at sentence %d" % sent_idx) from e
            except ValueError as e:
                raise ValueError("Could not process document at sentence %d" % sent_idx) from e
            self.sentences.append(sentence)
            begin_idx, end_idx = sentence.tokens[0].start_char, sentence.tokens[-1].end_char
            if all((self.text is not None, begin_idx is not None, end_idx is not None)): sentence.text = self.text[begin_idx: end_idx]
            sentence.index = sent_idx

        self._count_words()

        # Add a #text comment to each sentence in a doc if it doesn't already exist
        if not comments:
            comments = [[] for x in self.sentences]
        else:
            comments = [list(x) for x in comments]
        for sentence, sentence_comments in zip(self.sentences, comments):
            # the space after text can occur in treebanks such as the Naija-NSC treebank,
            # which extensively uses `# text_en =` and `# text_ortho`
            if sentence.text and not any(comment.startswith("# text ") or comment.startswith("#text ") or comment.startswith("# text=") or comment.startswith("#text=") for comment in sentence_comments):
                # split/join to handle weird whitespace, especially newlines
                sentence_comments.append("# text = " + ' '.join(sentence.text.split()))
            elif not sentence.text:
                for comment in sentence_comments:
                    if comment.startswith("# text ") or comment.startswith("#text ") or comment.startswith("# text=") or comment.startswith("#text="):
                        sentence.text = comment.split("=", 1)[-1].strip()
                        break

            for comment in sentence_comments:
                sentence.add_comment(comment)

            # look for sent_id in the comments
            # if it's there, overwrite the sent_idx id from above
            for comment in sentence_comments:
                if comment.startswith("# sent_id"):
                    sentence.sent_id = comment.split("=", 1)[-1].strip()
                    break
            else:
                # no sent_id found.  add a comment with our enumerated id
                # setting the sent_id on the sentence will automatically add the comment
                sentence.sent_id = str(sentence.index)

    def _count_words(self):
        """
        Count the number of tokens and words
        """
        self.num_tokens = sum([len(sentence.tokens) for sentence in self.sentences])
        self.num_words = sum([len(sentence.words) for sentence in self.sentences])

    def get(self, fields, as_sentences=False, from_token=False):
        """ Get fields from a list of field names.
        If only one field name (string or singleton list) is provided,
        return a list of that field; if more than one, return a list of list.
        Note that all returned fields are after multi-word expansion.

        Args:
            fields: name of the fields as a list or a single string
            as_sentences: if True, return the fields as a list of sentences; otherwise as a whole list
            from_token: if True, get the fields from Token; otherwise from Word

        Returns:
            All requested fields.
        """
        if isinstance(fields, str):
            fields = [fields]
        assert isinstance(fields, list), "Must provide field names as a list."
        assert len(fields) >= 1, "Must have at least one field."

        results = []
        for sentence in self.sentences:
            cursent = []
            # decide word or token
            if from_token:
                units = sentence.tokens
            else:
                units = sentence.words
            for unit in units:
                if len(fields) == 1:
                    cursent += [getattr(unit, fields[0])]
                else:
                    cursent += [[getattr(unit, field) for field in fields]]

            # decide whether append the results as a sentence or a whole list
            if as_sentences:
                results.append(cursent)
            else:
                results += cursent
        return results

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

    def set_mwt_expansions(self, expansions,
                           fake_dependencies=False,
                           process_manual_expanded=None):
        """ Extend the multi-word tokens annotated by tokenizer. A list of list of expansions
        will be expected for each multi-word token. Use `process_manual_expanded` to limit
        processing for tokens marked manually expanded:

        There are two types of MWT expansions: those with `misc`: `MWT=True`, and those with
        `manual_expansion`: True. The latter of which means that it is an expansion which the
        user manually specified through a postprocessor; the former means that it is a MWT
        which the detector picked out, but needs to be automatically expanded.

        process_manual_expanded = None - default; doesn't process manually expanded tokens
                                = True - process only manually expanded tokens (with `manual_expansion`: True)
                                = False - process only tokens explicitly tagged as MWT (`misc`: `MWT=True`)
        """

        idx_e = 0
        for sentence in self.sentences:
            idx_w = 0
            for token in sentence.tokens:
                idx_w += 1
                is_multi = (len(token.id) > 1)
                is_mwt = (multi_word_token_misc.match(token.misc) if token.misc is not None else None)
                is_manual_expansion = token.manual_expansion

                perform_mwt_processing = MWTProcessingType.FLATTEN

                if (process_manual_expanded and is_manual_expansion):
                    perform_mwt_processing = MWTProcessingType.PROCESS
                elif (process_manual_expanded==False and is_mwt):
                    perform_mwt_processing = MWTProcessingType.PROCESS
                elif (process_manual_expanded==False and is_manual_expansion):
                    perform_mwt_processing = MWTProcessingType.SKIP
                elif (process_manual_expanded==None and (is_mwt or is_multi)):
                    perform_mwt_processing = MWTProcessingType.PROCESS

                if perform_mwt_processing == MWTProcessingType.FLATTEN:
                    for word in token.words:
                        token.id = (idx_w, )
                        # delete dependency information
                        word.deps = None
                        word.head, word.deprel = None, None
                        word.id = idx_w
                elif perform_mwt_processing == MWTProcessingType.PROCESS:
                    expanded = [x for x in expansions[idx_e].split(' ') if len(x) > 0]
                    idx_e += 1
                    idx_w_end = idx_w + len(expanded) - 1
                    if token.misc:  # None can happen when using a prebuilt doc
                        token.misc = None if token.misc == 'MWT=Yes' else '|'.join([x for x in token.misc.split('|') if x != 'MWT=Yes'])
                    token.id = (idx_w, idx_w_end)
                    token.words = []
                    for i, e_word in enumerate(expanded):
                        token.words.append(Word(sentence, {ID: idx_w + i, TEXT: e_word}))
                    idx_w = idx_w_end
                elif perform_mwt_processing == MWTProcessingType.SKIP:
                    token.id = tuple(orig_id + idx_e for orig_id in token.id)
                    for i in token.words:
                        i.id += idx_e
                    idx_w = token.id[-1]
                    token.manual_expansion = None

            # reprocess the words using the new tokens
            sentence.words = []
            for token in sentence.tokens:
                token.sent = sentence
                for word in token.words:
                    word.sent = sentence
                    word.parent = token
                    sentence.words.append(word)
                if len(token.words) > 1 and token.start_char is not None and token.end_char is not None and "".join(word.text for word in token.words) == token.text:
                    start_char = token.start_char
                    for word in token.words:
                        end_char = start_char + len(word.text)
                        word.start_char = start_char
                        word.end_char = end_char
                        start_char = end_char

            if fake_dependencies:
                sentence.build_fake_dependencies()
            else:
                sentence.rebuild_dependencies()

        self._count_words() # update number of words & tokens
        assert idx_e == len(expansions), "{} {}".format(idx_e, len(expansions))
        return

    def get_mwt_expansions(self, evaluation=False):
        """ Get the multi-word tokens. For training, return a list of
        (multi-word token, extended multi-word token); otherwise, return a list of
        multi-word token only. By default doesn't skip already expanded tokens, but
        `skip_already_expanded` will return only tokens marked as MWT.
        """
        expansions = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                is_multi = (len(token.id) > 1)
                is_mwt = multi_word_token_misc.match(token.misc) if token.misc is not None else None
                is_manual_expansion = token.manual_expansion
                if (is_multi and not is_manual_expansion) or is_mwt:
                    src = token.text
                    dst = ' '.join([word.text for word in token.words])
                    expansions.append([src, dst])
        if evaluation: expansions = [e[0] for e in expansions]
        return expansions

    def build_ents(self):
        """ Build the list of entities by iterating over all words. Return all entities as a list. """
        self.ents = []
        for s in self.sentences:
            s_ents = s.build_ents()
            self.ents += s_ents
        return self.ents

    def iter_words(self):
        """ An iterator that returns all of the words in this Document. """
        for s in self.sentences:
            yield from s.words

    def iter_tokens(self):
        """ An iterator that returns all of the tokens in this Document. """
        for s in self.sentences:
            yield from s.tokens

    def sentence_comments(self):
        """ Returns a list of list of comments for the sentences """
        return [[comment for comment in sentence.comments] for sentence in self.sentences]

    @property
    def coref(self):
        """
        Access the coref lists of the document
        """
        return self._coref

    @coref.setter
    def coref(self, chains):
        """ Set the document's coref lists """
        self._coref = chains
        self._attach_coref_mentions(chains)

    def _attach_coref_mentions(self, chains):
        for sentence in self.sentences:
            for word in sentence.words:
                word.coref_chains = []

        for chain in chains:
            for mention_idx, mention in enumerate(chain.mentions):
                sentence = self.sentences[mention.sentence]
                for word_idx in range(mention.start_word, mention.end_word):
                    is_start = word_idx == mention.start_word
                    is_end = word_idx == mention.end_word - 1
                    is_representative = mention_idx == chain.representative_index
                    attachment = CorefAttachment(chain, is_start, is_end, is_representative)
                    sentence.words[word_idx].coref_chains.append(attachment)

    def reindex_sentences(self, start_index):
        for sent_id, sentence in zip(range(start_index, start_index + len(self.sentences)), self.sentences):
            sentence.sent_id = str(sent_id)

    def to_dict(self):
        """ Dumps the whole document into a list of list of dictionary for each token in each sentence in the doc.
        """
        return [sentence.to_dict() for sentence in self.sentences]

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def __format__(self, spec):
        if spec == 'c':
            return "\n\n".join("{:c}".format(s) for s in self.sentences)
        elif spec == 'C':
            return "\n\n".join("{:C}".format(s) for s in self.sentences)
        else:
            return str(self)

    def to_serialized(self):
        """ Dumps the whole document including text to a byte array containing a list of list of dictionaries for each token in each sentence in the doc.
        """
        return pickle.dumps((self.text, self.to_dict(), self.sentence_comments()))

    @classmethod
    def from_serialized(cls, serialized_string):
        """ Create and initialize a new document from a serialized string generated by Document.to_serialized_string():
        """
        stuff = pickle.loads(serialized_string)
        if not isinstance(stuff, tuple):
            raise TypeError("Serialized data was not a tuple when building a Document")
        if len(stuff) == 2:
            text, sentences = pickle.loads(serialized_string)
            doc = cls(sentences, text)
        else:
            text, sentences, comments = pickle.loads(serialized_string)
            doc = cls(sentences, text, comments)
        return doc

class Document(StanzaObject):
    """ A document class that stores attributes of a document and carries a list of sentences.
    """

    def __init__(self, sentences, text=None, comments=None, empty_sentences=None):
        """ Construct a document given a list of sentences in the form of lists of CoNLL-U dicts.

        Args:
            sentences: a list of sentences, which being a list of token entry, in the form of a CoNLL-U dict.
            text: the raw text of the document.
            comments: A list of list of strings to use as comments on the sentences, either None or the same length as sentences
        """
        self._sentences = []
        self._lang = None
        self._text = text
        self._num_tokens = 0
        self._num_words = 0

        self._process_sentences(sentences, comments, empty_sentences)
        self._ents = []
        self._coref = []
        if self._text is not None:
            self.build_ents()
            self.mark_whitespace()

    def mark_whitespace(self):
        for sentence in self._sentences:
            # TODO: pairwise, once we move to minimum 3.10
            for prev_token, next_token in zip(sentence.tokens[:-1], sentence.tokens[1:]):
                whitespace = self._text[prev_token.end_char:next_token.start_char]
                prev_token.spaces_after = whitespace
        for prev_sentence, next_sentence in zip(self._sentences[:-1], self._sentences[1:]):
            prev_token = prev_sentence.tokens[-1]
            next_token = next_sentence.tokens[0]
            whitespace = self._text[prev_token.end_char:next_token.start_char]
            prev_token.spaces_after = whitespace
        if len(self._sentences) > 0 and len(self._sentences[-1].tokens) > 0:
            final_token = self._sentences[-1].tokens[-1]
            whitespace = self._text[final_token.end_char:]
            final_token.spaces_after = whitespace
        if len(self._sentences) > 0 and len(self._sentences[0].tokens) > 0:
            first_token = self._sentences[0].tokens[0]
            whitespace = self._text[:first_token.start_char]
            first_token.spaces_before = whitespace


    @property
    def lang(self):
        """ Access the language of this document """
        return self._lang

    @lang.setter
    def lang(self, value):
        """ Set the language of this document """
        self._lang = value

    @property
    def text(self):
        """ Access the raw text for this document. """
        return self._text

    @text.setter
    def text(self, value):
        """ Set the raw text for this document. """
        self._text = value

    @property
    def sentences(self):
        """ Access the list of sentences for this document. """
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        """ Set the list of tokens for this document. """
        self._sentences = value

    @property
    def num_tokens(self):
        """ Access the number of tokens for this document. """
        return self._num_tokens

    @num_tokens.setter
    def num_tokens(self, value):
        """ Set the number of tokens for this document. """
        self._num_tokens = value

    @property
    def num_words(self):
        """ Access the number of words for this document. """
        return self._num_words

    @num_words.setter
    def num_words(self, value):
        """ Set the number of words for this document. """
        self._num_words = value

    @property
    def ents(self):
        """ Access the list of entities in this document. """
        return self._ents

    @ents.setter
    def ents(self, value):
        """ Set the list of entities in this document. """
        self._ents = value

    @property
    def entities(self):
        """ Access the list of entities. This is just an alias of `ents`. """
        return self._ents

    @entities.setter
    def entities(self, value):
        """ Set the list of entities in this document. """
        self._ents = value

    def _process_sentences(self, sentences, comments=None, empty_sentences=None):
        self.sentences = []
        if empty_sentences is None:
            empty_sentences = repeat([])
        for sent_idx, (tokens, empty_words) in enumerate(zip(sentences, empty_sentences)):
            try:
                sentence = Sentence(tokens, doc=self, empty_words=empty_words)
            except IndexError as e:
                raise IndexError("Could not process document at sentence %d" % sent_idx) from e
            except ValueError as e:
                raise ValueError("Could not process document at sentence %d" % sent_idx) from e
            self.sentences.append(sentence)
            begin_idx, end_idx = sentence.tokens[0].start_char, sentence.tokens[-1].end_char
            if all((self.text is not None, begin_idx is not None, end_idx is not None)): sentence.text = self.text[begin_idx: end_idx]
            sentence.index = sent_idx

        self._count_words()

        # Add a #text comment to each sentence in a doc if it doesn't already exist
        if not comments:
            comments = [[] for x in self.sentences]
        else:
            comments = [list(x) for x in comments]
        for sentence, sentence_comments in zip(self.sentences, comments):
            # the space after text can occur in treebanks such as the Naija-NSC treebank,
            # which extensively uses `# text_en =` and `# text_ortho`
            if sentence.text and not any(comment.startswith("# text ") or comment.startswith("#text ") or comment.startswith("# text=") or comment.startswith("#text=") for comment in sentence_comments):
                # split/join to handle weird whitespace, especially newlines
                sentence_comments.append("# text = " + ' '.join(sentence.text.split()))
            elif not sentence.text:
                for comment in sentence_comments:
                    if comment.startswith("# text ") or comment.startswith("#text ") or comment.startswith("# text=") or comment.startswith("#text="):
                        sentence.text = comment.split("=", 1)[-1].strip()
                        break

            for comment in sentence_comments:
                sentence.add_comment(comment)

            # look for sent_id in the comments
            # if it's there, overwrite the sent_idx id from above
            for comment in sentence_comments:
                if comment.startswith("# sent_id"):
                    sentence.sent_id = comment.split("=", 1)[-1].strip()
                    break
            else:
                # no sent_id found.  add a comment with our enumerated id
                # setting the sent_id on the sentence will automatically add the comment
                sentence.sent_id = str(sentence.index)

    def _count_words(self):
        """
        Count the number of tokens and words
        """
        self.num_tokens = sum([len(sentence.tokens) for sentence in self.sentences])
        self.num_words = sum([len(sentence.words) for sentence in self.sentences])

    def get(self, fields, as_sentences=False, from_token=False):
        """ Get fields from a list of field names.
        If only one field name (string or singleton list) is provided,
        return a list of that field; if more than one, return a list of list.
        Note that all returned fields are after multi-word expansion.

        Args:
            fields: name of the fields as a list or a single string
            as_sentences: if True, return the fields as a list of sentences; otherwise as a whole list
            from_token: if True, get the fields from Token; otherwise from Word

        Returns:
            All requested fields.
        """
        if isinstance(fields, str):
            fields = [fields]
        assert isinstance(fields, list), "Must provide field names as a list."
        assert len(fields) >= 1, "Must have at least one field."

        results = []
        for sentence in self.sentences:
            cursent = []
            # decide word or token
            if from_token:
                units = sentence.tokens
            else:
                units = sentence.words
            for unit in units:
                if len(fields) == 1:
                    cursent += [getattr(unit, fields[0])]
                else:
                    cursent += [[getattr(unit, field) for field in fields]]

            # decide whether append the results as a sentence or a whole list
            if as_sentences:
                results.append(cursent)
            else:
                results += cursent
        return results

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

    def set_mwt_expansions(self, expansions,
                           fake_dependencies=False,
                           process_manual_expanded=None):
        """ Extend the multi-word tokens annotated by tokenizer. A list of list of expansions
        will be expected for each multi-word token. Use `process_manual_expanded` to limit
        processing for tokens marked manually expanded:

        There are two types of MWT expansions: those with `misc`: `MWT=True`, and those with
        `manual_expansion`: True. The latter of which means that it is an expansion which the
        user manually specified through a postprocessor; the former means that it is a MWT
        which the detector picked out, but needs to be automatically expanded.

        process_manual_expanded = None - default; doesn't process manually expanded tokens
                                = True - process only manually expanded tokens (with `manual_expansion`: True)
                                = False - process only tokens explicitly tagged as MWT (`misc`: `MWT=True`)
        """

        idx_e = 0
        for sentence in self.sentences:
            idx_w = 0
            for token in sentence.tokens:
                idx_w += 1
                is_multi = (len(token.id) > 1)
                is_mwt = (multi_word_token_misc.match(token.misc) if token.misc is not None else None)
                is_manual_expansion = token.manual_expansion

                perform_mwt_processing = MWTProcessingType.FLATTEN

                if (process_manual_expanded and is_manual_expansion):
                    perform_mwt_processing = MWTProcessingType.PROCESS
                elif (process_manual_expanded==False and is_mwt):
                    perform_mwt_processing = MWTProcessingType.PROCESS
                elif (process_manual_expanded==False and is_manual_expansion):
                    perform_mwt_processing = MWTProcessingType.SKIP
                elif (process_manual_expanded==None and (is_mwt or is_multi)):
                    perform_mwt_processing = MWTProcessingType.PROCESS

                if perform_mwt_processing == MWTProcessingType.FLATTEN:
                    for word in token.words:
                        token.id = (idx_w, )
                        # delete dependency information
                        word.deps = None
                        word.head, word.deprel = None, None
                        word.id = idx_w
                elif perform_mwt_processing == MWTProcessingType.PROCESS:
                    expanded = [x for x in expansions[idx_e].split(' ') if len(x) > 0]
                    idx_e += 1
                    idx_w_end = idx_w + len(expanded) - 1
                    if token.misc:  # None can happen when using a prebuilt doc
                        token.misc = None if token.misc == 'MWT=Yes' else '|'.join([x for x in token.misc.split('|') if x != 'MWT=Yes'])
                    token.id = (idx_w, idx_w_end)
                    token.words = []
                    for i, e_word in enumerate(expanded):
                        token.words.append(Word(sentence, {ID: idx_w + i, TEXT: e_word}))
                    idx_w = idx_w_end
                elif perform_mwt_processing == MWTProcessingType.SKIP:
                    token.id = tuple(orig_id + idx_e for orig_id in token.id)
                    for i in token.words:
                        i.id += idx_e
                    idx_w = token.id[-1]
                    token.manual_expansion = None

            # reprocess the words using the new tokens
            sentence.words = []
            for token in sentence.tokens:
                token.sent = sentence
                for word in token.words:
                    word.sent = sentence
                    word.parent = token
                    sentence.words.append(word)
                if len(token.words) > 1 and token.start_char is not None and token.end_char is not None and "".join(word.text for word in token.words) == token.text:
                    start_char = token.start_char
                    for word in token.words:
                        end_char = start_char + len(word.text)
                        word.start_char = start_char
                        word.end_char = end_char
                        start_char = end_char

            if fake_dependencies:
                sentence.build_fake_dependencies()
            else:
                sentence.rebuild_dependencies()

        self._count_words() # update number of words & tokens
        assert idx_e == len(expansions), "{} {}".format(idx_e, len(expansions))
        return

    def get_mwt_expansions(self, evaluation=False):
        """ Get the multi-word tokens. For training, return a list of
        (multi-word token, extended multi-word token); otherwise, return a list of
        multi-word token only. By default doesn't skip already expanded tokens, but
        `skip_already_expanded` will return only tokens marked as MWT.
        """
        expansions = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                is_multi = (len(token.id) > 1)
                is_mwt = multi_word_token_misc.match(token.misc) if token.misc is not None else None
                is_manual_expansion = token.manual_expansion
                if (is_multi and not is_manual_expansion) or is_mwt:
                    src = token.text
                    dst = ' '.join([word.text for word in token.words])
                    expansions.append([src, dst])
        if evaluation: expansions = [e[0] for e in expansions]
        return expansions

    def build_ents(self):
        """ Build the list of entities by iterating over all words. Return all entities as a list. """
        self.ents = []
        for s in self.sentences:
            s_ents = s.build_ents()
            self.ents += s_ents
        return self.ents

    def iter_words(self):
        """ An iterator that returns all of the words in this Document. """
        for s in self.sentences:
            yield from s.words

    def iter_tokens(self):
        """ An iterator that returns all of the tokens in this Document. """
        for s in self.sentences:
            yield from s.tokens

    def sentence_comments(self):
        """ Returns a list of list of comments for the sentences """
        return [[comment for comment in sentence.comments] for sentence in self.sentences]

    @property
    def coref(self):
        """
        Access the coref lists of the document
        """
        return self._coref

    @coref.setter
    def coref(self, chains):
        """ Set the document's coref lists """
        self._coref = chains
        self._attach_coref_mentions(chains)

    def _attach_coref_mentions(self, chains):
        for sentence in self.sentences:
            for word in sentence.words:
                word.coref_chains = []

        for chain in chains:
            for mention_idx, mention in enumerate(chain.mentions):
                sentence = self.sentences[mention.sentence]
                for word_idx in range(mention.start_word, mention.end_word):
                    is_start = word_idx == mention.start_word
                    is_end = word_idx == mention.end_word - 1
                    is_representative = mention_idx == chain.representative_index
                    attachment = CorefAttachment(chain, is_start, is_end, is_representative)
                    sentence.words[word_idx].coref_chains.append(attachment)

    def reindex_sentences(self, start_index):
        for sent_id, sentence in zip(range(start_index, start_index + len(self.sentences)), self.sentences):
            sentence.sent_id = str(sent_id)

    def to_dict(self):
        """ Dumps the whole document into a list of list of dictionary for each token in each sentence in the doc.
        """
        return [sentence.to_dict() for sentence in self.sentences]

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def __format__(self, spec):
        if spec == 'c':
            return "\n\n".join("{:c}".format(s) for s in self.sentences)
        elif spec == 'C':
            return "\n\n".join("{:C}".format(s) for s in self.sentences)
        else:
            return str(self)

    def to_serialized(self):
        """ Dumps the whole document including text to a byte array containing a list of list of dictionaries for each token in each sentence in the doc.
        """
        return pickle.dumps((self.text, self.to_dict(), self.sentence_comments()))

    @classmethod
    def from_serialized(cls, serialized_string):
        """ Create and initialize a new document from a serialized string generated by Document.to_serialized_string():
        """
        stuff = pickle.loads(serialized_string)
        if not isinstance(stuff, tuple):
            raise TypeError("Serialized data was not a tuple when building a Document")
        if len(stuff) == 2:
            text, sentences = pickle.loads(serialized_string)
            doc = cls(sentences, text)
        else:
            text, sentences, comments = pickle.loads(serialized_string)
            doc = cls(sentences, text, comments)
        return doc

class Document(StanzaObject):
    """ A document class that stores attributes of a document and carries a list of sentences.
    """

    def __init__(self, sentences, text=None, comments=None, empty_sentences=None):
        """ Construct a document given a list of sentences in the form of lists of CoNLL-U dicts.

        Args:
            sentences: a list of sentences, which being a list of token entry, in the form of a CoNLL-U dict.
            text: the raw text of the document.
            comments: A list of list of strings to use as comments on the sentences, either None or the same length as sentences
        """
        self._sentences = []
        self._lang = None
        self._text = text
        self._num_tokens = 0
        self._num_words = 0

        self._process_sentences(sentences, comments, empty_sentences)
        self._ents = []
        self._coref = []
        if self._text is not None:
            self.build_ents()
            self.mark_whitespace()

    def mark_whitespace(self):
        for sentence in self._sentences:
            # TODO: pairwise, once we move to minimum 3.10
            for prev_token, next_token in zip(sentence.tokens[:-1], sentence.tokens[1:]):
                whitespace = self._text[prev_token.end_char:next_token.start_char]
                prev_token.spaces_after = whitespace
        for prev_sentence, next_sentence in zip(self._sentences[:-1], self._sentences[1:]):
            prev_token = prev_sentence.tokens[-1]
            next_token = next_sentence.tokens[0]
            whitespace = self._text[prev_token.end_char:next_token.start_char]
            prev_token.spaces_after = whitespace
        if len(self._sentences) > 0 and len(self._sentences[-1].tokens) > 0:
            final_token = self._sentences[-1].tokens[-1]
            whitespace = self._text[final_token.end_char:]
            final_token.spaces_after = whitespace
        if len(self._sentences) > 0 and len(self._sentences[0].tokens) > 0:
            first_token = self._sentences[0].tokens[0]
            whitespace = self._text[:first_token.start_char]
            first_token.spaces_before = whitespace


    @property
    def lang(self):
        """ Access the language of this document """
        return self._lang

    @lang.setter
    def lang(self, value):
        """ Set the language of this document """
        self._lang = value

    @property
    def text(self):
        """ Access the raw text for this document. """
        return self._text

    @text.setter
    def text(self, value):
        """ Set the raw text for this document. """
        self._text = value

    @property
    def sentences(self):
        """ Access the list of sentences for this document. """
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        """ Set the list of tokens for this document. """
        self._sentences = value

    @property
    def num_tokens(self):
        """ Access the number of tokens for this document. """
        return self._num_tokens

    @num_tokens.setter
    def num_tokens(self, value):
        """ Set the number of tokens for this document. """
        self._num_tokens = value

    @property
    def num_words(self):
        """ Access the number of words for this document. """
        return self._num_words

    @num_words.setter
    def num_words(self, value):
        """ Set the number of words for this document. """
        self._num_words = value

    @property
    def ents(self):
        """ Access the list of entities in this document. """
        return self._ents

    @ents.setter
    def ents(self, value):
        """ Set the list of entities in this document. """
        self._ents = value

    @property
    def entities(self):
        """ Access the list of entities. This is just an alias of `ents`. """
        return self._ents

    @entities.setter
    def entities(self, value):
        """ Set the list of entities in this document. """
        self._ents = value

    def _process_sentences(self, sentences, comments=None, empty_sentences=None):
        self.sentences = []
        if empty_sentences is None:
            empty_sentences = repeat([])
        for sent_idx, (tokens, empty_words) in enumerate(zip(sentences, empty_sentences)):
            try:
                sentence = Sentence(tokens, doc=self, empty_words=empty_words)
            except IndexError as e:
                raise IndexError("Could not process document at sentence %d" % sent_idx) from e
            except ValueError as e:
                raise ValueError("Could not process document at sentence %d" % sent_idx) from e
            self.sentences.append(sentence)
            begin_idx, end_idx = sentence.tokens[0].start_char, sentence.tokens[-1].end_char
            if all((self.text is not None, begin_idx is not None, end_idx is not None)): sentence.text = self.text[begin_idx: end_idx]
            sentence.index = sent_idx

        self._count_words()

        # Add a #text comment to each sentence in a doc if it doesn't already exist
        if not comments:
            comments = [[] for x in self.sentences]
        else:
            comments = [list(x) for x in comments]
        for sentence, sentence_comments in zip(self.sentences, comments):
            # the space after text can occur in treebanks such as the Naija-NSC treebank,
            # which extensively uses `# text_en =` and `# text_ortho`
            if sentence.text and not any(comment.startswith("# text ") or comment.startswith("#text ") or comment.startswith("# text=") or comment.startswith("#text=") for comment in sentence_comments):
                # split/join to handle weird whitespace, especially newlines
                sentence_comments.append("# text = " + ' '.join(sentence.text.split()))
            elif not sentence.text:
                for comment in sentence_comments:
                    if comment.startswith("# text ") or comment.startswith("#text ") or comment.startswith("# text=") or comment.startswith("#text="):
                        sentence.text = comment.split("=", 1)[-1].strip()
                        break

            for comment in sentence_comments:
                sentence.add_comment(comment)

            # look for sent_id in the comments
            # if it's there, overwrite the sent_idx id from above
            for comment in sentence_comments:
                if comment.startswith("# sent_id"):
                    sentence.sent_id = comment.split("=", 1)[-1].strip()
                    break
            else:
                # no sent_id found.  add a comment with our enumerated id
                # setting the sent_id on the sentence will automatically add the comment
                sentence.sent_id = str(sentence.index)

    def _count_words(self):
        """
        Count the number of tokens and words
        """
        self.num_tokens = sum([len(sentence.tokens) for sentence in self.sentences])
        self.num_words = sum([len(sentence.words) for sentence in self.sentences])

    def get(self, fields, as_sentences=False, from_token=False):
        """ Get fields from a list of field names.
        If only one field name (string or singleton list) is provided,
        return a list of that field; if more than one, return a list of list.
        Note that all returned fields are after multi-word expansion.

        Args:
            fields: name of the fields as a list or a single string
            as_sentences: if True, return the fields as a list of sentences; otherwise as a whole list
            from_token: if True, get the fields from Token; otherwise from Word

        Returns:
            All requested fields.
        """
        if isinstance(fields, str):
            fields = [fields]
        assert isinstance(fields, list), "Must provide field names as a list."
        assert len(fields) >= 1, "Must have at least one field."

        results = []
        for sentence in self.sentences:
            cursent = []
            # decide word or token
            if from_token:
                units = sentence.tokens
            else:
                units = sentence.words
            for unit in units:
                if len(fields) == 1:
                    cursent += [getattr(unit, fields[0])]
                else:
                    cursent += [[getattr(unit, field) for field in fields]]

            # decide whether append the results as a sentence or a whole list
            if as_sentences:
                results.append(cursent)
            else:
                results += cursent
        return results

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

    def set_mwt_expansions(self, expansions,
                           fake_dependencies=False,
                           process_manual_expanded=None):
        """ Extend the multi-word tokens annotated by tokenizer. A list of list of expansions
        will be expected for each multi-word token. Use `process_manual_expanded` to limit
        processing for tokens marked manually expanded:

        There are two types of MWT expansions: those with `misc`: `MWT=True`, and those with
        `manual_expansion`: True. The latter of which means that it is an expansion which the
        user manually specified through a postprocessor; the former means that it is a MWT
        which the detector picked out, but needs to be automatically expanded.

        process_manual_expanded = None - default; doesn't process manually expanded tokens
                                = True - process only manually expanded tokens (with `manual_expansion`: True)
                                = False - process only tokens explicitly tagged as MWT (`misc`: `MWT=True`)
        """

        idx_e = 0
        for sentence in self.sentences:
            idx_w = 0
            for token in sentence.tokens:
                idx_w += 1
                is_multi = (len(token.id) > 1)
                is_mwt = (multi_word_token_misc.match(token.misc) if token.misc is not None else None)
                is_manual_expansion = token.manual_expansion

                perform_mwt_processing = MWTProcessingType.FLATTEN

                if (process_manual_expanded and is_manual_expansion):
                    perform_mwt_processing = MWTProcessingType.PROCESS
                elif (process_manual_expanded==False and is_mwt):
                    perform_mwt_processing = MWTProcessingType.PROCESS
                elif (process_manual_expanded==False and is_manual_expansion):
                    perform_mwt_processing = MWTProcessingType.SKIP
                elif (process_manual_expanded==None and (is_mwt or is_multi)):
                    perform_mwt_processing = MWTProcessingType.PROCESS

                if perform_mwt_processing == MWTProcessingType.FLATTEN:
                    for word in token.words:
                        token.id = (idx_w, )
                        # delete dependency information
                        word.deps = None
                        word.head, word.deprel = None, None
                        word.id = idx_w
                elif perform_mwt_processing == MWTProcessingType.PROCESS:
                    expanded = [x for x in expansions[idx_e].split(' ') if len(x) > 0]
                    idx_e += 1
                    idx_w_end = idx_w + len(expanded) - 1
                    if token.misc:  # None can happen when using a prebuilt doc
                        token.misc = None if token.misc == 'MWT=Yes' else '|'.join([x for x in token.misc.split('|') if x != 'MWT=Yes'])
                    token.id = (idx_w, idx_w_end)
                    token.words = []
                    for i, e_word in enumerate(expanded):
                        token.words.append(Word(sentence, {ID: idx_w + i, TEXT: e_word}))
                    idx_w = idx_w_end
                elif perform_mwt_processing == MWTProcessingType.SKIP:
                    token.id = tuple(orig_id + idx_e for orig_id in token.id)
                    for i in token.words:
                        i.id += idx_e
                    idx_w = token.id[-1]
                    token.manual_expansion = None

            # reprocess the words using the new tokens
            sentence.words = []
            for token in sentence.tokens:
                token.sent = sentence
                for word in token.words:
                    word.sent = sentence
                    word.parent = token
                    sentence.words.append(word)
                if len(token.words) > 1 and token.start_char is not None and token.end_char is not None and "".join(word.text for word in token.words) == token.text:
                    start_char = token.start_char
                    for word in token.words:
                        end_char = start_char + len(word.text)
                        word.start_char = start_char
                        word.end_char = end_char
                        start_char = end_char

            if fake_dependencies:
                sentence.build_fake_dependencies()
            else:
                sentence.rebuild_dependencies()

        self._count_words() # update number of words & tokens
        assert idx_e == len(expansions), "{} {}".format(idx_e, len(expansions))
        return

    def get_mwt_expansions(self, evaluation=False):
        """ Get the multi-word tokens. For training, return a list of
        (multi-word token, extended multi-word token); otherwise, return a list of
        multi-word token only. By default doesn't skip already expanded tokens, but
        `skip_already_expanded` will return only tokens marked as MWT.
        """
        expansions = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                is_multi = (len(token.id) > 1)
                is_mwt = multi_word_token_misc.match(token.misc) if token.misc is not None else None
                is_manual_expansion = token.manual_expansion
                if (is_multi and not is_manual_expansion) or is_mwt:
                    src = token.text
                    dst = ' '.join([word.text for word in token.words])
                    expansions.append([src, dst])
        if evaluation: expansions = [e[0] for e in expansions]
        return expansions

    def build_ents(self):
        """ Build the list of entities by iterating over all words. Return all entities as a list. """
        self.ents = []
        for s in self.sentences:
            s_ents = s.build_ents()
            self.ents += s_ents
        return self.ents

    def iter_words(self):
        """ An iterator that returns all of the words in this Document. """
        for s in self.sentences:
            yield from s.words

    def iter_tokens(self):
        """ An iterator that returns all of the tokens in this Document. """
        for s in self.sentences:
            yield from s.tokens

    def sentence_comments(self):
        """ Returns a list of list of comments for the sentences """
        return [[comment for comment in sentence.comments] for sentence in self.sentences]

    @property
    def coref(self):
        """
        Access the coref lists of the document
        """
        return self._coref

    @coref.setter
    def coref(self, chains):
        """ Set the document's coref lists """
        self._coref = chains
        self._attach_coref_mentions(chains)

    def _attach_coref_mentions(self, chains):
        for sentence in self.sentences:
            for word in sentence.words:
                word.coref_chains = []

        for chain in chains:
            for mention_idx, mention in enumerate(chain.mentions):
                sentence = self.sentences[mention.sentence]
                for word_idx in range(mention.start_word, mention.end_word):
                    is_start = word_idx == mention.start_word
                    is_end = word_idx == mention.end_word - 1
                    is_representative = mention_idx == chain.representative_index
                    attachment = CorefAttachment(chain, is_start, is_end, is_representative)
                    sentence.words[word_idx].coref_chains.append(attachment)

    def reindex_sentences(self, start_index):
        for sent_id, sentence in zip(range(start_index, start_index + len(self.sentences)), self.sentences):
            sentence.sent_id = str(sent_id)

    def to_dict(self):
        """ Dumps the whole document into a list of list of dictionary for each token in each sentence in the doc.
        """
        return [sentence.to_dict() for sentence in self.sentences]

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, cls=DocJSONEncoder)

    def __format__(self, spec):
        if spec == 'c':
            return "\n\n".join("{:c}".format(s) for s in self.sentences)
        elif spec == 'C':
            return "\n\n".join("{:C}".format(s) for s in self.sentences)
        else:
            return str(self)

    def to_serialized(self):
        """ Dumps the whole document including text to a byte array containing a list of list of dictionaries for each token in each sentence in the doc.
        """
        return pickle.dumps((self.text, self.to_dict(), self.sentence_comments()))

    @classmethod
    def from_serialized(cls, serialized_string):
        """ Create and initialize a new document from a serialized string generated by Document.to_serialized_string():
        """
        stuff = pickle.loads(serialized_string)
        if not isinstance(stuff, tuple):
            raise TypeError("Serialized data was not a tuple when building a Document")
        if len(stuff) == 2:
            text, sentences = pickle.loads(serialized_string)
            doc = cls(sentences, text)
        else:
            text, sentences, comments = pickle.loads(serialized_string)
            doc = cls(sentences, text, comments)
        return doc

