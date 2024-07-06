def init_device_mesh(
        device_type: str,
        mesh_shape: Tuple[int, ...],
        *,
        mesh_dim_names: Optional[Tuple[str, ...]] = None,
    ) -> DeviceMesh:
        """
        Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names` parameters.

        This creates a DeviceMesh with an n-dimensional array layout, where `n` is the length of `mesh_shape`.
        If `mesh_dim_names` is provided, each dimension is labeled as `mesh_dim_names[i]`.

        .. note::
            `init_device_mesh` follows SPMD programming model, meaning the same PyTorch Python program
            runs on all processes/ranks in the cluster. Ensure `mesh_shape` (the dimensions of the nD array
            describing device layout) is identical across all ranks. Inconsistent `mesh_shape` may lead to hanging.

        .. note::
            If no process group is found, init_device_mesh will initialize distributed process group/groups
            required for distributed communications behind the scene.

        Args:
            device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".
                Passing in a device type with a GPU index, such as "cuda:0", is not allowed.
            mesh_shape (Tuple[int]): A tuple defining the dimensions of the multi-dimensional array
                describing the layout of devices.
            mesh_dim_names (Tuple[str], optional): A tuple of mesh dimension names to assign to each dimension
                of the multi-dimensional array describing the layout of devices. Its length must match the length
                of `mesh_shape`. Each string in `mesh_dim_names` must be unique.

        Returns:
            DeviceMesh: A :class:`DeviceMesh` object representing the device layout.

        Example::
            >>> # xdoctest: +SKIP("no rank")
            >>> from torch.distributed.device_mesh import init_device_mesh
            >>>
            >>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,))
            >>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))

        """
        if mesh_dim_names is not None:
            if len(set(mesh_dim_names)) != len(mesh_dim_names):
                raise RuntimeError(
                    "Each mesh_dim_name must be unique.",
                    f"Found repeated mesh_dim_name in mesh_dim_names {mesh_dim_names}",
                )

            if len(mesh_shape) != len(mesh_dim_names):
                raise RuntimeError(
                    "mesh_shape and mesh_dim_names should have same length!",
                    f"Found len(mesh_dim_names): {len(mesh_dim_names)} and len(mesh_shape):{len(mesh_shape)}.",
                )

        # assume valid device types are all letters
        if device_type and not device_type.isalpha():
            raise RuntimeError(
                f"Device type with GPU index is not supported but got {device_type}. ",
                "If you maintained a 'torch.device' object, it's recommended to pass in 'device.type'.",
            )

        # Always initialize the mesh's tensor on CPU, regardless of what the
        # external device type has been set to be (e.g. meta)
        with torch.device("cpu"):
            mesh = torch.arange(math.prod(mesh_shape), dtype=torch.int).view(mesh_shape)
        device_mesh = DeviceMesh(
            device_type=device_type,
            mesh=mesh,
            mesh_dim_names=mesh_dim_names,
        )

        return device_mesh

