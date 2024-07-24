class Pretrain:
    """ A loader and saver for pretrained embeddings. """

    def __init__(self, filename=None, vec_filename=None, max_vocab=-1, save_to_file=True, csv_filename=None):
        self.filename = filename
        self._vec_filename = vec_filename
        self._csv_filename = csv_filename
        self._max_vocab = max_vocab
        self._save_to_file = save_to_file

    @property
    def vocab(self):
        if not hasattr(self, '_vocab'):
            self.load()
        return self._vocab

    @property
    def emb(self):
        if not hasattr(self, '_emb'):
            self.load()
        return self._emb

    def load(self):
        if self.filename is not None and os.path.exists(self.filename):
            try:
                data = torch.load(self.filename, lambda storage, loc: storage)
                logger.debug("Loaded pretrain from {}".format(self.filename))
                if not isinstance(data, dict):
                    raise RuntimeError("File {} exists but is not a stanza pretrain file.  It is not a dict, whereas a Stanza pretrain should have a dict with 'emb' and 'vocab'".format(self.filename))
                if 'emb' not in data or 'vocab' not in data:
                    raise RuntimeError("File {} exists but is not a stanza pretrain file.  A Stanza pretrain file should have 'emb' and 'vocab' fields in its state dict".format(self.filename))
                self._vocab, self._emb = PretrainedWordVocab.load_state_dict(data['vocab']), data['emb']
                return
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                if not self._vec_filename and not self._csv_filename:
                    raise
                logger.warning("Pretrained file exists but cannot be loaded from {}, due to the following exception:\n\t{}".format(self.filename, e))
                vocab, emb = self.read_pretrain()
        else:
            if not self._vec_filename and not self._csv_filename:
                raise FileNotFoundError("Pretrained file {} does not exist, and no text/xz file was provided".format(self.filename))
            if self.filename is not None:
                logger.info("Pretrained filename %s specified, but file does not exist.  Attempting to load from text file" % self.filename)
            vocab, emb = self.read_pretrain()

        self._vocab = vocab
        self._emb = emb

        if self._save_to_file:
            # save to file
            assert self.filename is not None, "Filename must be provided to save pretrained vector to file."
            self.save(self.filename)

    def save(self, filename):
        directory, _ = os.path.split(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # should not infinite loop since the load function sets _vocab and _emb before trying to save
        data = {'vocab': self.vocab.state_dict(), 'emb': self.emb}
        try:
            torch.save(data, filename, _use_new_zipfile_serialization=False, pickle_protocol=3)
            logger.info("Saved pretrained vocab and vectors to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            logger.warning("Saving pretrained data failed due to the following exception... continuing anyway.\n\t{}".format(e))


    def write_text(self, filename):
        """
        Write the vocab & values to a text file
        """
        with open(filename, "w") as fout:
            for i in range(len(self.vocab)):
                row = self.emb[i]
                fout.write(self.vocab[i])
                fout.write("\t")
                fout.write("\t".join(map(str, row)))
                fout.write("\n")


    def read_pretrain(self):
        # load from pretrained filename
        if self._vec_filename is not None:
            words, emb, failed = self.read_from_file(self._vec_filename, self._max_vocab)
        elif self._csv_filename is not None:
            words, emb = self.read_from_csv(self._csv_filename)
        else:
            raise RuntimeError("Vector file is not provided.")

        if len(emb) - len(VOCAB_PREFIX) != len(words):
            raise RuntimeError("Loaded number of vectors does not match number of words.")

        # Use a fixed vocab size
        if self._max_vocab > len(VOCAB_PREFIX) and self._max_vocab < len(words) + len(VOCAB_PREFIX):
            words = words[:self._max_vocab - len(VOCAB_PREFIX)]
            emb = emb[:self._max_vocab]

        vocab = PretrainedWordVocab(words)
        
        return vocab, emb

    @staticmethod
    def read_from_csv(filename):
        """
        Read vectors from CSV

        Skips the first row
        """
        logger.info("Reading pretrained vectors from csv file %s ...", filename)
        with open_read_text(filename) as fin:
            csv_reader = csv.reader(fin)
            # the header of the thai csv vector file we have is just the number of columns
            # so we read past the first line
            for line in csv_reader:
                break
            lines = [line for line in csv_reader]

        rows = len(lines)
        cols = len(lines[0]) - 1

        emb = np.zeros((rows + len(VOCAB_PREFIX), cols), dtype=np.float32)
        for i, line in enumerate(lines):
            emb[i+len(VOCAB_PREFIX)] = [float(x) for x in line[-cols:]]
        words = [line[0].replace(' ', '\xa0') for line in lines]
        return words, emb

    @staticmethod
    def read_from_file(filename, max_vocab=None):
        """
        Open a vector file using the provided function and read from it.
        """
        logger.info("Reading pretrained vectors from %s ...", filename)

        # some vector files, such as Google News, use tabs
        tab_space_pattern = re.compile(r"[ \t]+")
        first = True
        cols = None
        lines = []
        failed = 0
        unk_line = None
        with open_read_binary(filename) as f:
            for i, line in enumerate(f):
                try:
                    line = line.decode()
                except UnicodeDecodeError:
                    failed += 1
                    continue
                line = line.rstrip()
                if not line:
                    continue
                pieces = tab_space_pattern.split(line)
                if first:
                    # the first line contains the number of word vectors and the dimensionality
                    # note that a 1d embedding with a number as the first entry
                    # will fail to read properly.  we ignore that case
                    first = False
                    if len(pieces) == 2:
                        cols = int(pieces[1])
                        continue

                if pieces[0] == '<unk>':
                    if unk_line is not None:
                        logger.error("More than one <unk> line in the pretrain!  Keeping the most recent one")
                    else:
                        logger.debug("Found an unk line while reading the pretrain")
                    unk_line = pieces
                else:
                    if not max_vocab or max_vocab < 0 or len(lines) < max_vocab:
                        lines.append(pieces)

        if cols is None:
            # another failure case: all words have spaces in them
            cols = min(len(x) for x in lines) - 1
        rows = len(lines)
        emb = np.zeros((rows + len(VOCAB_PREFIX), cols), dtype=np.float32)
        if unk_line is not None:
            emb[UNK_ID] = [float(x) for x in unk_line[-cols:]]
        for i, line in enumerate(lines):
            emb[i+len(VOCAB_PREFIX)] = [float(x) for x in line[-cols:]]

        # if there were word pieces separated with spaces, rejoin them with nbsp instead
        # this way, the normalize_unit method in vocab.py can find the word at test time
        words = ['\xa0'.join(line[:-cols]) for line in lines]
        if failed > 0:
            logger.info("Failed to read %d lines from embedding", failed)
        return words, emb, failed

