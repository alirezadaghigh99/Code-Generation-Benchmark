class DenseLayerCreator(LayerCreator):
    def __init__(self, name, kernel, bias=None):
        super(DenseLayerCreator, self).__init__(kernel, bias)
        self.name = name

    def __call__(self, delay_build=False):
        layer = keras.layers.Dense(
            self.kernel.shape[-1], activation=None, name=self.name
        )
        if not delay_build:
            layer.build((None, self.kernel.shape[0]))
            assert len(layer.get_weights()) == 2
            if self.bias is None:
                self.bias = np.zeros((self.kernel.shape[-1],))
            layer.set_weights([self.kernel, self.bias])
        return layer

class MockPruningScheduler(PruningScheduler):
    def __init__(self, step_and_sparsity_pairs: List[Tuple]):
        self._org_pairs = step_and_sparsity_pairs
        self.step_and_sparsity_pairs = {
            step: sparsity for (step, sparsity) in step_and_sparsity_pairs
        }

    def should_prune(self, step: int):
        return step in self.step_and_sparsity_pairs

    def target_sparsity(self, step: int):
        update_ready = step in self.step_and_sparsity_pairs
        sparsity = self.step_and_sparsity_pairs[step] if update_ready else None
        return sparsity

    def get_config(self):
        return {
            "class_name": self.__class__.__name__,
            "step_and_sparsity_pairs": self._org_pairs,
        }

