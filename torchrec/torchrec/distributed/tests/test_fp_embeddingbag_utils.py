class TestFPEBCSharder(FeatureProcessedEmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        if fused_params is None:
            fused_params = {}

        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

        ebc_sharder = TestEBCSharder(
            self._sharding_type,
            self._kernel_type,
            fused_params,
            qcomm_codecs_registry,
        )
        super().__init__(ebc_sharder, qcomm_codecs_registry)

    def sharding_types(self, compute_device_type: str) -> List[str]:
        """
        Restricts sharding to single type only.
        """
        return (
            [self._sharding_type]
            if self._sharding_type
            in super().sharding_types(compute_device_type=compute_device_type)
            else []
        )

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        """
        Restricts to single impl.
        """
        return [self._kernel_type]

