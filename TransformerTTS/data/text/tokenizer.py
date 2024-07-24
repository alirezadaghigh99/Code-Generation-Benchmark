class Tokenizer:
    
    def __init__(self, start_token='>', end_token='<', pad_token='/', add_start_end=True, alphabet=None,
                 model_breathing=True):
        if not alphabet:
            self.alphabet = all_phonemes
        else:
            self.alphabet = sorted(list(set(alphabet)))  # for testing
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet, start=1)}
        self.idx_to_token[0] = pad_token
        self.token_to_idx = {s: [i] for i, s in self.idx_to_token.items()}
        self.vocab_size = len(self.alphabet) + 1
        self.add_start_end = add_start_end
        if add_start_end:
            self.start_token_index = len(self.alphabet) + 1
            self.end_token_index = len(self.alphabet) + 2
            self.vocab_size += 2
            self.idx_to_token[self.start_token_index] = start_token
            self.idx_to_token[self.end_token_index] = end_token
        self.model_breathing = model_breathing
        if model_breathing:
            self.breathing_token_index = self.vocab_size
            self.token_to_idx[' '] = self.token_to_idx[' '] + [self.breathing_token_index]
            self.vocab_size += 1
            self.breathing_token = '@'
            self.idx_to_token[self.breathing_token_index] = self.breathing_token
            self.token_to_idx[self.breathing_token] = [self.breathing_token_index]
    
    def __call__(self, sentence: str) -> list:
        sequence = [self.token_to_idx[c] for c in sentence]  # No filtering: text should only contain known chars.
        sequence = [item for items in sequence for item in items]
        if self.model_breathing:
            sequence = [self.breathing_token_index] + sequence
        if self.add_start_end:
            sequence = [self.start_token_index] + sequence + [self.end_token_index]
        return sequence
    
    def decode(self, sequence: list) -> str:
        return ''.join([self.idx_to_token[int(t)] for t in sequence])

