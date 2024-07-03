    class DeviceMesh:
        """
        DeviceMesh represents a mesh of devices, where layout of devices could be
        represented as a n-d dimension array, and each value of the n-d dimensional
        array is the global id of the default process group ranks.

        DeviceMesh could be used to describe the layout of devices across the cluster,
        and serves as a proxy for communication among the device lists within the cluster.

        DeviceMesh can be used as a context manager.

        .. note::
            DeviceMesh follows SPMD programming model, which means the same PyTorch Python program
            is running on all processes/ranks in the cluster. Therefore, users need to make sure the
            `mesh` array (which describes the layout of devices) should be identical across all ranks.
            Inconsistent `mesh` will lead to silent hang.

        Args:
            device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".
            mesh (ndarray): A multi-dimensional array or an integer tensor describing the layout
                of devices, where the IDs are global IDs of the default process group.

        Returns:
            DeviceMesh: A :class:`DeviceMesh` object representing the device layout.

        The following program runs on each process/rank in an SPMD manner. In this example, we have 2
        hosts with 4 GPUs each.
        A reduction over the first dimension of mesh will reduce across
        columns (0, 4), .. and (3, 7), a reduction over the second dimension
        of mesh reduces across rows (0, 1, 2, 3) and (4, 5, 6, 7).

        Example::
            >>> # xdoctest: +SKIP("no rank")
            >>> from torch.distributed.device_mesh import DeviceMesh
            >>>
            >>> # Initialize device mesh as (2, 4) to represent the topology
            >>> # of cross-host(dim 0), and within-host (dim 1).
            >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
        """

        device_type: str
        mesh: torch.Tensor
        mesh_dim_names: Optional[Tuple[str, ...]]

        def __init__(
            self,
            device_type: str,
            mesh: Union[torch.Tensor, "ArrayLike"],
            *,
            mesh_dim_names: Optional[Tuple[str, ...]] = None,
            _init_backend: bool = True,
        ) -> None:
            self.device_type = device_type
            if isinstance(mesh, torch.Tensor) and mesh.device.type != "cpu":
                raise ValueError(f"`mesh` must be a CPU tensor, got {mesh}")
            self.mesh = (
                mesh.detach().to(dtype=torch.int)
                if isinstance(mesh, torch.Tensor)
                else torch.tensor(mesh, device="cpu", dtype=torch.int)
            )
            self.mesh_dim_names = tuple(mesh_dim_names) if mesh_dim_names else None

            # private field to pre-generate DeviceMesh's hash
            self._flatten_mesh_list = tuple(self.mesh.flatten().tolist())
            self._parent_mesh: Optional[DeviceMesh] = None
            self._thread_id = threading.get_ident()

            # Skip process group initialization if xla device or init backend is False
            # TODO(yeounoh) implement DeviceMesh backend and register XLA backend.
            if device_type != "xla":
                # always try to create default (world) pg, even if it is not initialized
                # already. The world pg is used for device mesh identity (rank) on each
                # process (we need to know if the current global rank is in the mesh or not).
                if _init_backend:
                    self._get_or_create_default_group()
                    self._init_process_groups()

                # calculate the coordinates of the current global rank on the mesh
                rank_coords = (self.mesh == get_rank()).nonzero()
                assert rank_coords.size(0) in (0, 1)
                self._coordinate_on_dim: Optional[List[int]] = (
                    rank_coords[0].tolist() if rank_coords.size(0) > 0 else None
                )

        def _get_or_create_default_group(self):
            default_initialized = is_initialized()
            if not default_initialized:
                init_process_group()

            world_size = get_world_size()
            if self.mesh.numel() > world_size:
                raise RuntimeError(
                    f"Mesh should not be bigger than default world size, but found {self.mesh.numel()} ranks!"
                )

            device_handle = _get_device_handle(self.device_type)
            # TODO: if user want to pass pg_options, offer a way to do it
            if not default_initialized and device_handle:
                # automatically set the current cuda/cuda-like device base on num of gpu devices available in each host
                # NOTE: This device selection would only work for homogeneous hardware.
                num_devices_per_host = device_handle.device_count()
                if (
                    world_size > num_devices_per_host
                    and world_size % num_devices_per_host != 0
                ):
                    raise RuntimeError(
                        f"DeviceMesh only support homogeneous hardware, but found "
                        f"{world_size} ranks and {num_devices_per_host} {self.device_type} devices!"
                    )
                device_handle.set_device(get_rank() % num_devices_per_host)

            return _get_default_group()

        def _init_process_groups(self):
            # tag/ranks/group_name associated with each mesh dimension, each
            # mesh dimension should have one sub-group per rank
            #
            # TODO(yifu): remove tag and ranks once we fully migrate to native
            # functional collectives. See details in:
            # https://github.com/pytorch/pytorch/issues/93173#issuecomment-1907095208
            dim_group_infos: List[Tuple[str, List[int], str]] = []

            if self.mesh.ndim == 1 and self.mesh.numel() == get_world_size():
                # if the mesh is the same as world_pg, we just append the default
                # pg to the first dim groups, as new_group cannot have the exact
                # same ranks as world
                dim_group_infos.append(
                    (
                        _get_group_tag(_get_default_group()),
                        list(range(get_world_size())),
                        _get_default_group().group_name,
                    )
                )
            else:
                # create sub pgs base on the mesh argument specified
                for dim in range(self.mesh.ndim):
                    # swap the current dim to the last dim
                    # then reshape to flatten out other dims
                    pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(
                        -1, self.mesh.size(dim)
                    )
                    # multi-dim mesh, create subgroups by looping over the pg_ranks
                    # for each dim and append the groups
                    for dim_mesh in pg_ranks_by_dim:
                        subgroup_ranks = dim_mesh.tolist()

                        # Respect dim group options specified via _MeshEnv.set_dim_group_options().
                        # Inherit from the parent group if no options are specified for the group.
                        if dim in _mesh_resources.mesh_dim_group_options:
                            (
                                backend,
                                pg_options,
                            ) = _mesh_resources.mesh_dim_group_options[dim]
                        else:
                            backend, pg_options = None, None

                        # We temporarily revert the re-use subgroup, since it breaks two internal tests.
                        # Temporarily reverting to resolve test timeout while root-causing.
                        # TODO: Add two tests to cover internal tests scenarios and re-enable reuse subgroup if exists.
                        dim_group = new_group(
                            ranks=subgroup_ranks,
                            backend=backend,
                            pg_options=pg_options,
                        )

                        # only add to dim_groups if the current rank in the subgroup
                        if self.get_rank() in subgroup_ranks:
                            if len(dim_group_infos) > dim:
                                raise RuntimeError(
                                    f"Each device mesh dimension should get only one process group, but got {self.get_rank()} "
                                    f"in {subgroup_ranks}!"
                                )
                            dim_group_infos.append(
                                (
                                    _get_group_tag(not_none(dim_group)),
                                    subgroup_ranks,
                                    dim_group.group_name,
                                )
                            )
            self._dim_group_infos = dim_group_infos

        def __enter__(self) -> "DeviceMesh":
            # set this mesh as the current mesh in mesh env
            _mesh_resources.mesh_stack.append(self)
            return self

        # pyre-fixme[2]: Parameter must be annotated.
        def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
            # pop this mesh from mesh env
            _mesh_resources.mesh_stack.pop()

        def __repr__(self) -> str:
            device_mesh_repr = (
                f"DeviceMesh({self.mesh.tolist()})"
                if not self.mesh_dim_names
                else f"DeviceMesh({self.mesh.tolist()}, mesh_dim_names={self.mesh_dim_names})"
            )
            return device_mesh_repr

        def __hash__(self):
            # lazily compute hash
            self._hash = getattr(self, "_hash", None)
            if not self._hash:
                self._hash = hash(
                    (
                        self._flatten_mesh_list,
                        self.mesh.shape,
                        self.device_type,
                        self.mesh_dim_names,
                        self._parent_mesh,
                        self._thread_id,
                    )
                )
            return self._hash

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, DeviceMesh):
                return False
            if id(self) == id(other):
                return True
            else:
                return (
                    self._flatten_mesh_list == other._flatten_mesh_list
                    and self.mesh.shape == other.mesh.shape
                    and self.device_type == other.device_type
                    and self.mesh_dim_names == other.mesh_dim_names
                    and self._parent_mesh == other._parent_mesh
                    and self._thread_id == other._thread_id
                )

        def __getitem__(
            self, mesh_dim_names: Union[str, Tuple[str, ...]]
        ) -> "DeviceMesh":
            """
            Slice the current DeviceMesh based on the mesh_dim_name given to create a child
            DeviceMesh.

            Args:
                mesh_dim_name (Union[str, Tuple[str]]): the name or the tuple of names of the
                mesh dimension of the parent DeviceMesh to create the child DeviceMesh for.
            Returns:
                A :class:`DeviceMesh` object

            The following program runs on each process/rank in an SPMD manner. In this example, we have 2
            hosts with 4 GPUs each.
            Calling mesh["tp"] on rank 0, 1, 2, 3 would return a 1D child DeviceMesh:([0, 1, 2, 3]).
            Calling mesh["tp"] on rank 4, 5, 6, 7 would return a 1D child DeviceMesh:([4, 5, 6, 7]).
            Calling mesh["dp"] on rank 0, 4 would return a 1D child DeviceMesh:([0, 4]).
            Calling mesh["dp"] on rank 1, 5 would return a 1D child DeviceMesh:([1, 5]).
            Calling mesh["dp"] on rank 2, 6 would return a 1D child DeviceMesh:([2, 6]).
            Calling mesh["dp"] on rank 3, 7 would return a 1D child DeviceMesh:([3, 7]).

            Example::
                >>> # xdoctest: +SKIP("no rank")
                >>> from torch.distributed.device_mesh import DeviceMesh
                >>>
                >>> # Initialize device mesh as (2, 4) to represent the topology
                >>> # of cross-host(dim 0), and within-host (dim 1).
                >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
            """
            if not self.mesh_dim_names:
                raise RuntimeError("Cannot slice a DeviceMesh without mesh_dim_names!")

            mesh_dim_names = (
                (mesh_dim_names,) if isinstance(mesh_dim_names, str) else mesh_dim_names
            )

            error_msg = (
                f"Invalid mesh_dim_name {mesh_dim_names} specified. "
                f"Valid mesh_dim_names should be a contiguous subsequence of {self.mesh_dim_names}."
            )

            if mesh_dim_names == self.mesh_dim_names:
                return self
            elif len(mesh_dim_names) > len(self.mesh_dim_names) or not all(
                mesh_dim_name in self.mesh_dim_names for mesh_dim_name in mesh_dim_names
            ):
                raise KeyError(error_msg)
            # Check if the user-provided slicing is a valid contiguous subsequence of the mesh_dim_names
            # of the current DeviceMesh.
            else:
                outermost_dim_name = mesh_dim_names[0]
                outermost_dim_idx = self.mesh_dim_names.index(outermost_dim_name)
                for i, j in zip(
                    mesh_dim_names,
                    self.mesh_dim_names[outermost_dim_idx : len(mesh_dim_names)],
                ):
                    if i != j:
                        raise KeyError(error_msg)

            submesh = _mesh_resources.create_child_mesh(self, mesh_dim_names)
            return submesh

        def get_group(self, mesh_dim: Optional[Union[int, str]] = None) -> ProcessGroup:
            """
            Returns the single ProcessGroup specified by mesh_dim, or, if mesh_dim is not specified and the
            DeviceMesh is 1-dimensional, returns the only ProcessGroup in the mesh.

            Args:
                mesh_dim (str/int, optional): it can be the name of the mesh dimension or the index
                of the mesh dimension. Default is None.

            Returns:
                A :class:`ProcessGroup` object.
            """
            if not hasattr(self, "_dim_group_infos"):
                raise RuntimeError("DeviceMesh process groups not initialized!")

            if self.mesh.ndim > 1 and mesh_dim is None:
                raise RuntimeError(
                    f"Found the DeviceMesh have {self.mesh.ndim} dimensions",
                    "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
                    "If you want to get the list of all the ProcessGroups in the DeviceMesh,"
                    "please use `get_all_groups()` instead.",
                )

            if self.mesh.ndim == 1 and mesh_dim is None:
                mesh_dim = 0
            else:
                mesh_dim = (
                    _mesh_resources.get_mesh_dim_by_name(self, mesh_dim)
                    if isinstance(mesh_dim, str)
                    else mesh_dim
                )

            return not_none(
                _find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim][:2])  # type: ignore[index]
            )

        def get_all_groups(self) -> List[ProcessGroup]:
            """
            Returns a list of ProcessGroups for all mesh dimensions.

            Returns:
                A list of :class:`ProcessGroup` object.
            """
            return [self.get_group(i) for i in range(self.mesh.ndim)]

        @staticmethod
        def from_group(
            group: Union[ProcessGroup, List[ProcessGroup]],
            device_type: str,
            mesh: Optional[Union[torch.Tensor, "ArrayLike"]] = None,
            *,
            mesh_dim_names: Optional[Tuple[str, ...]] = None,
        ) -> "DeviceMesh":
            """
            Contstructs a :class:`DeviceMesh` with ``device_type`` from an
            existing :class:`ProcessGroup`.

            The constructed device mesh has number of dimensions equal to the
            number of groups passed. If more than one group is passed, then the
            ``mesh`` argument is required.
            """
            if isinstance(group, ProcessGroup):
                group_ranks = get_process_group_ranks(group)
                if (
                    isinstance(mesh, torch.Tensor) and mesh.tolist() != group_ranks
                ) or (mesh is not None and mesh != group_ranks):
                    raise ValueError(
                        f"Invalid mesh {str(mesh)} for ProcessGroup with ranks {group_ranks}"
                    )
                mesh = torch.tensor(group_ranks, device="cpu", dtype=torch.int)
                device_mesh = DeviceMesh(
                    device_type,
                    mesh,
                    mesh_dim_names=mesh_dim_names,
                    _init_backend=False,
                )
                device_mesh._dim_group_infos = [
                    (_get_group_tag(group), group_ranks, group.group_name)
                ]
                return device_mesh
            groups = list(group)
            if len(groups) == 0:
                raise ValueError("Expects at least one ProcessGroup to be passed")
            if mesh is None:
                raise ValueError("Must pass mesh if passing multiple ProcessGroups")
            mesh = (
                mesh.detach().to(dtype=torch.int, device="cpu")
                if isinstance(mesh, torch.Tensor)
                else torch.tensor(mesh, device="cpu", dtype=torch.int)
            )
            if mesh.ndim != len(groups):
                raise ValueError(
                    "Expects mesh with ndim equal to number of ProcessGroups but got "
                    f"mesh {mesh.tolist()} and {len(groups)} ProcessGroups"
                )
            device_mesh = DeviceMesh(
                device_type, mesh, mesh_dim_names=mesh_dim_names, _init_backend=False
            )
            device_mesh._dim_group_infos = [
                (
                    _get_group_tag(group),
                    get_process_group_ranks(group),
                    group.group_name,
                )
                for group in groups
            ]
            return device_mesh

        def size(self, mesh_dim: Optional[int] = None) -> int:
            return self.mesh.numel() if mesh_dim is None else self.mesh.size(mesh_dim)

        @property
        def ndim(self) -> int:
            return self.mesh.ndim

        @property
        def shape(self) -> Tuple[int, ...]:
            return tuple(self.mesh.shape)

        def get_rank(self) -> int:
            """
            Returns the current global rank.
            """
            return get_rank()

        def get_local_rank(self, mesh_dim: Optional[Union[int, str]] = None) -> int:
            """
            Returns the local rank of the given mesh_dim of the DeviceMesh.

            Args:
                mesh_dim (str/int, optional): it can be the name of the mesh dimension or the index
                of the mesh dimension. Default is None.

            Returns:
                An integer denotes the local rank.

            The following program runs on each process/rank in an SPMD manner. In this example, we have 2
            hosts with 4 GPUs each.
            Calling mesh_2d.get_local_rank(mesh_dim=0) on rank 0, 1, 2, 3 would return 0.
            Calling mesh_2d.get_local_rank(mesh_dim=0) on rank 4, 5, 6, 7 would return 1.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 0, 4 would return 0.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 1, 5 would return 1.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 2, 6 would return 2.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 3, 7 would return 3.

            Example::
                >>> # xdoctest: +SKIP("no rank")
                >>> from torch.distributed.device_mesh import DeviceMesh
                >>>
                >>> # Initialize device mesh as (2, 4) to represent the topology
                >>> # of cross-host(dim 0), and within-host (dim 1).
                >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
            """
            if self.ndim > 1 and mesh_dim is None:
                raise RuntimeError(
                    f"Found the DeviceMesh have {self.mesh.ndim} dimensions",
                    "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
                )
            elif mesh_dim is None:
                mesh_dim = 0

            mesh_dim_group = not_none(self.get_group(mesh_dim))
            assert isinstance(
                mesh_dim_group, ProcessGroup
            ), "We expect ProcessGroup before calling `get_rank`!"
            return not_none(get_rank(mesh_dim_group))

        def get_coordinate(self) -> Optional[List[int]]:
            """
            Return the relative indices of this rank relative to all
            dimensions of the mesh. If this rank is not part of the mesh, return None.
            """
            return self._coordinate_on_dim if self._coordinate_on_dim else None