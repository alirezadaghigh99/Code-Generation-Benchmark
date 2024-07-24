class Partial(Placement):
    """
    The ``Partial(reduce_op)`` placement describes the DTensor that is pending
    reduction on a specified ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension holds the partial value of the global Tensor. User can
    redistribute the ``Partial`` DTensor to a ``Replicate`` or ``Shard(dim)``
    placement on the specified ``DeviceMesh`` dimension using ``redistribute``,
    which would trigger necessary communication operations under the hood (i.e.
    ``allreduce``, ``reduce_scatter``).

    Args:
        reduce_op (str, optional): The reduction op to be used for the partial DTensor
        to produce Replicated/Sharded DTensor. Only element-wise reduction operations
        are supportd, including: "sum", "avg", "prod", "max", "min", default: "sum".

    ::note:: The ``Partial`` placement can be generated as a result of the DTensor operators,
        and can only be used by the ``DTensor.from_local`` API.
    """

    reduce_op: str = "sum"

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # Partial placement contract #1:
        # _reduce_value: reduce the value of the tensor on the mesh dimension
        return funcol.all_reduce(
            tensor, reduceOp=self.reduce_op, group=(mesh, mesh_dim)
        )

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # Partial placement contract #2:
        # _reduce_shard_value: reduce_scatter the value of the tensor over the mesh dimension
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # Partial placement contract #3:
        # _partition_value: partition the value of a replicated tensor on the mesh dimension

        # _partition_value is the conjugate operation of _reduce_value
        # - i.e. _partition_value on a sum reduce op is just a divison operation
        # - the _reduce_value on a sum reduce op would just be a sum(allreduce) operation
        # TODO: if the reduce_op is min/max, etc. the _partition_value should be a
        # different operation
        assert self.reduce_op == "sum", "only support replicate to PartialSUM for now!"
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        return tensor / num_chunks

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partial):
            return False
        return self.reduce_op == other.reduce_op

    def __hash__(self) -> int:
        return 1 + hash(self.reduce_op)

    def __repr__(self) -> str:
        """
        machine readable representation of the Partial placement
        """
        return f"Partial({self.reduce_op})"

    def __str__(self) -> str:
        """
        human readable representation of the Partial placement
        """
        return "P"

class Replicate(Placement):
    """
    The ``Replicate()`` placement describes the DTensor replicating on a corresponding
    ``DeviceMesh`` dimension, where each rank on the DeviceMesh dimension holds a
    replica of the global Tensor. The ``Replicate`` placement can be used by all
    DTensor APIs (i.e. distribute_tensor, from_local, etc.)
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Replicate):
            return False
        return True

    def __hash__(self) -> int:
        # every replicate placement is the same
        return -1

    def __repr__(self) -> str:
        """
        machine readable representation of the Replicate placement
        """
        return "Replicate()"

    def __str__(self) -> str:
        """
        human readable representation of the Replicate placement
        """
        return "R"

    def _replicate_tensor(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        """
        Replicate (broadcast) a torch.Tensor on a mesh dimension (use
        the first coordinate on the mesh dimension as source of truth)
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        tensor = tensor.contiguous()
        mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim)
        return tensor

