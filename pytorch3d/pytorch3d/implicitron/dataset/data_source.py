class ImplicitronDataSource(DataSourceBase):  # pyre-ignore[13]
    """
    Represents the data used in Implicitron. This is the only implementation
    of DataSourceBase provided.

    Members:
        dataset_map_provider_class_type: identifies type for dataset_map_provider.
            e.g. JsonIndexDatasetMapProvider for Co3D.
        data_loader_map_provider_class_type: identifies type for data_loader_map_provider.
    """

    dataset_map_provider: DatasetMapProviderBase
    dataset_map_provider_class_type: str
    data_loader_map_provider: DataLoaderMapProviderBase
    data_loader_map_provider_class_type: str = "SequenceDataLoaderMapProvider"

    @classmethod
    def pre_expand(cls) -> None:
        # use try/finally to bypass cinder's lazy imports
        try:
            from .blender_dataset_map_provider import (  # noqa: F401
                BlenderDatasetMapProvider,
            )
            from .json_index_dataset_map_provider import (  # noqa: F401
                JsonIndexDatasetMapProvider,
            )
            from .json_index_dataset_map_provider_v2 import (  # noqa: F401
                JsonIndexDatasetMapProviderV2,
            )
            from .llff_dataset_map_provider import LlffDatasetMapProvider  # noqa: F401
            from .rendered_mesh_dataset_map_provider import (  # noqa: F401
                RenderedMeshDatasetMapProvider,
            )
            from .train_eval_data_loader_provider import (  # noqa: F401
                TrainEvalDataLoaderMapProvider,
            )

            try:
                from .sql_dataset_provider import (  # noqa: F401  # pyre-ignore
                    SqlIndexDatasetMapProvider,
                )
            except ModuleNotFoundError:
                pass  # environment without SQL dataset
        finally:
            pass

    def __post_init__(self):
        run_auto_creation(self)
        self._all_train_cameras_cache: Optional[Tuple[Optional[CamerasBase]]] = None

    def get_datasets_and_dataloaders(self) -> Tuple[DatasetMap, DataLoaderMap]:
        datasets = self.dataset_map_provider.get_dataset_map()
        dataloaders = self.data_loader_map_provider.get_data_loader_map(datasets)
        return datasets, dataloaders

    @property
    def all_train_cameras(self) -> Optional[CamerasBase]:
        """
        DEPRECATED! The property will be removed in future versions.
        """
        if self._all_train_cameras_cache is None:  # pyre-ignore[16]
            all_train_cameras = self.dataset_map_provider.get_all_train_cameras()
            self._all_train_cameras_cache = (all_train_cameras,)

        return self._all_train_cameras_cache[0]

