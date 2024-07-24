class TensorStatisticsCollectionController(PTCompressionAlgorithmController):
    def __init__(
        self, target_model: NNCFNetwork, ip_vs_collector_dict: Dict[PTTargetPoint, TensorStatisticCollectorBase]
    ):
        super().__init__(target_model)
        self.ip_vs_collector_dict = ip_vs_collector_dict
        self._scheduler = StubCompressionScheduler()
        self._loss = ZeroCompressionLoss("cpu")

    @property
    def loss(self) -> ZeroCompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> StubCompressionScheduler:
        return self._scheduler

    def start_collection(self):
        for collector in self.ip_vs_collector_dict.values():
            collector.enable()

    def stop_collection(self):
        for collector in self.ip_vs_collector_dict.values():
            collector.disable()

    def compression_stage(self) -> CompressionStage:
        return CompressionStage.FULLY_COMPRESSED

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

