class _DummyModel(BenchmarkModule):  # type: ignore[misc]
    def __init__(
        self,
        dataloader_kNN: DataLoader[LightlyDataset],
        knn_k: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__(dataloader_kNN, num_classes=num_classes, knn_k=knn_k)
        self.backbone = Sequential(
            Flatten(),
            Linear(3 * 32 * 32, num_classes),
        )
        self.criterion = CrossEntropyLoss()

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets, _ = batch
        predictions = self.backbone(images)
        loss: Tensor = self.criterion(predictions, targets)
        return loss

    def configure_optimizers(self) -> Optimizer:
        return SGD(self.backbone.parameters(), lr=0.1)

