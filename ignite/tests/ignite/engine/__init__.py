class BatchChecker:
    def __init__(self, data, init_counter=0):
        self.counter = init_counter
        self.data = data
        self.true_batch = None

    def check(self, batch):
        self.true_batch = self.data[self.counter % len(self.data)]
        self.counter += 1
        res = self.true_batch == batch
        return res.all() if not isinstance(res, bool) else res

