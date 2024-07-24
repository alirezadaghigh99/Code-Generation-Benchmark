class DatasetExperience(CLExperience, Generic[TCLDataset]):
    """An Experience that provides a dataset."""

    def __init__(
        self, *, dataset: TCLDataset, current_experience: Optional[int] = None
    ):
        super().__init__(current_experience=current_experience, origin_stream=None)
        self._dataset: AvalancheDataset = dataset

    @property
    def dataset(self) -> AvalancheDataset:
        # dataset is a read-only property
        data = self._dataset
        return data

