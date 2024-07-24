class NoCompressionAlgorithmController(PTCompressionAlgorithmController):
    def __init__(self, target_model):
        super().__init__(target_model)

        self._loss = ZeroCompressionLoss(get_model_device(target_model))
        self._scheduler = StubCompressionScheduler()

    def compression_stage(self) -> CompressionStage:
        """
        Returns level of compression. Should be used on saving best checkpoints to distinguish between
        uncompressed, partially compressed and fully compressed models.
        """
        return CompressionStage.UNCOMPRESSED

    @property
    def loss(self) -> ZeroCompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def strip(self, do_copy: bool = True) -> NNCFNetwork:
        model = self.model
        if do_copy:
            model = copy_model(self.model)
        return model

