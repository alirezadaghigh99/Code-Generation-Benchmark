class RingBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.length = 0
        self.data = []

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[idx]

    def __delitem__(self, subscript):
        del self.data[subscript]
        self.length = len(self.data)

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply append to data.
            self.data.append(v)
            self.length += 1

        elif self.length == self.maxlen:
            # No space, remove the first item then append
            self.data.pop(0)
            self.data.append(v)
        else:
            # This should never happen.
            raise nncf.BufferFullError()

