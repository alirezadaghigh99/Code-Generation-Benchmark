    def eval(self, valid_set):
        eval_rate = self.model.evaluate(valid_set["X"], valid_set["y"], batch_size=self.batch_size)
        return eval_rate