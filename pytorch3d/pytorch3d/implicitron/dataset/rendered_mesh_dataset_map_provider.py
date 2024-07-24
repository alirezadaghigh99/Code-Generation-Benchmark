class RenderedMeshDatasetMapProvider(DatasetMapProviderBase):  # pyre-ignore [13]
    """
    A simple single-scene dataset based on PyTorch3D renders of a mesh.
    Provides `num_views` renders of the mesh as train, with no val
    and test. The renders are generated from viewpoints sampled at uniformly
    distributed azimuth intervals. The elevation is kept constant so that the
    camera's vertical position coincides with the equator.

    By default, uses Keenan Crane's cow model, and the camera locations are
    set to make sense for that.

    Although the rendering used to generate this dataset will use a GPU
    if one is available, the data it produces is on the CPU just like
    the data returned by implicitron's other dataset map providers.
    This is because both datasets and models can be large, so implicitron's
    training loop expects data on the CPU and only moves
    what it needs to the device.

    For a more detailed explanation of this code, please refer to the
    docs/tutorials/fit_textured_mesh.ipynb notebook.

    Members:
        num_views: The number of generated renders.
        data_file: The folder that contains the mesh file. By default, finds
            the cow mesh in the same repo as this code.
        azimuth_range: number of degrees on each side of the start position to
            take samples
        distance: distance from camera centres to the origin.
        resolution: the common height and width of the output images.
        use_point_light: whether to use a particular point light as opposed
            to ambient white.
        gpu_idx: which gpu to use for rendering the mesh.
        path_manager_factory: (Optional) An object that generates an instance of
            PathManager that can translate provided file paths.
        path_manager_factory_class_type: The class type of `path_manager_factory`.
    """

    num_views: int = 40
    data_file: Optional[str] = None
    azimuth_range: float = 180
    distance: float = 2.7
    resolution: int = 128
    use_point_light: bool = True
    gpu_idx: Optional[int] = 0
    path_manager_factory: PathManagerFactory
    path_manager_factory_class_type: str = "PathManagerFactory"

    def get_dataset_map(self) -> DatasetMap:
        # pyre-ignore[16]
        return DatasetMap(train=self.train_dataset, val=None, test=None)

    def get_all_train_cameras(self) -> CamerasBase:
        # pyre-ignore[16]
        return self.poses

    def __post_init__(self) -> None:
        super().__init__()
        run_auto_creation(self)
        if torch.cuda.is_available() and self.gpu_idx is not None:
            device = torch.device(f"cuda:{self.gpu_idx}")
        else:
            device = torch.device("cpu")
        if self.data_file is None:
            data_file = join(
                dirname(dirname(dirname(dirname(realpath(__file__))))),
                "docs",
                "tutorials",
                "data",
                "cow_mesh",
                "cow.obj",
            )
        else:
            data_file = self.data_file
        io = IO(path_manager=self.path_manager_factory.get())
        mesh = io.load_mesh(data_file, device=device)
        poses, images, masks = _generate_cow_renders(
            num_views=self.num_views,
            mesh=mesh,
            azimuth_range=self.azimuth_range,
            distance=self.distance,
            resolution=self.resolution,
            device=device,
            use_point_light=self.use_point_light,
        )
        # pyre-ignore[16]
        self.poses = poses.cpu()
        # pyre-ignore[16]
        self.train_dataset = SingleSceneDataset(  # pyre-ignore[28]
            object_name="cow",
            images=list(images.permute(0, 3, 1, 2).cpu()),
            fg_probabilities=list(masks[:, None].cpu()),
            poses=[self.poses[i] for i in range(len(poses))],
            frame_types=[DATASET_TYPE_KNOWN] * len(poses),
            eval_batches=None,
        )

