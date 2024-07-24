class FoundationCache:
    def __init__(self, other=None):
        if other is None:
            self.bert = {}
            self.charlms = {}
            self.pretrains = {}
            # future proof the module by using a lock for the glorious day
            # when the GIL is finally gone
            self.lock = threading.Lock()
        else:
            self.bert = other.bert
            self.charlms = other.charlms
            self.pretrains = other.pretrains
            self.lock = other.lock

    def load_bert(self, transformer_name):
        m, t, _ = self.load_bert_with_peft(transformer_name, None)
        return m, t

    def load_bert_with_peft(self, transformer_name, peft_name):
        """
        Load a transformer only once

        Uses a lock for thread safety
        """
        if transformer_name is None:
            return None, None, None
        with self.lock:
            if transformer_name not in self.bert:
                model, tokenizer = bert_embedding.load_bert(transformer_name)
                self.bert[transformer_name] = BertRecord(model, tokenizer, {})
            else:
                logger.debug("Reusing bert %s", transformer_name)

            bert_record = self.bert[transformer_name]
            if not peft_name:
                return bert_record.model, bert_record.tokenizer, None
            if peft_name not in bert_record.peft_ids:
                bert_record.peft_ids[peft_name] = 0
            else:
                bert_record.peft_ids[peft_name] = bert_record.peft_ids[peft_name] + 1
            peft_name = "%s_%d" % (peft_name, bert_record.peft_ids[peft_name])
            return bert_record.model, bert_record.tokenizer, peft_name

    def load_bert_copy(self, transformer_name):
        """
        If the transformer is already in the FoundationCache, return a copy of the transformer

        Uses a lock for thread safety
        """
        if transformer_name is None:
            return None, None
        with self.lock:
            if transformer_name not in self.bert:
                model, tokenizer = bert_embedding.load_bert(transformer_name)
                return model, tokenizer
            model, tokenizer, _ = self.bert[transformer_name]
            return deepcopy(model), deepcopy(tokenizer)


    def load_charlm(self, filename):
        if not filename:
            return None

        with self.lock:
            if filename not in self.charlms:
                logger.debug("Loading charlm from %s", filename)
                self.charlms[filename] = CharacterLanguageModel.load(filename, finetune=False)
            else:
                logger.debug("Reusing charlm from %s", filename)

            return self.charlms[filename]

    def load_pretrain(self, filename):
        """
        Load a pretrained word embedding only once

        Uses a lock for thread safety
        """
        if filename is None:
            return None
        with self.lock:
            if filename not in self.pretrains:
                logger.debug("Loading pretrain %s", filename)
                self.pretrains[filename] = Pretrain(filename)
            else:
                logger.debug("Reusing pretrain %s", filename)

            return self.pretrains[filename]

