class NoCompressionAlgorithmController(BaseCompressionAlgorithmController):
    def __init__(self, target_model: tf.keras.Model):
        super().__init__(target_model)
        self._loss = TFZeroCompressionLoss()
        self._scheduler = StubCompressionScheduler()

    @property
    def loss(self) -> TFZeroCompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> StubCompressionScheduler:
        return self._scheduler

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def strip(self, do_copy: bool = True) -> tf.keras.Model:
        model = self.model
        if do_copy:
            model = copy_model(self.model)
        return model

