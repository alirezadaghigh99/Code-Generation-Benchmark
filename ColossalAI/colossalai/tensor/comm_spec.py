class CommSpec:
    """
    Communication spec is used to record the communication action. It has two main functions:
    1. Compute the communication cost which will be used in auto parallel solver.
    2. Convert the communication spec to real action which will be used in runtime.
    It contains comm_pattern to determine the
    communication method, sharding_spec to determine the communication size, gather_dim and shard_dim
    to determine the buffer shape, and logical_process_axis

    Argument:
        comm_pattern(CollectiveCommPattern): describe the communication method used in this spec.
        sharding_spec(ShardingSpec): This is sharding spec of the tensor which will join the communication action.
        gather_dim(int, Optional): The gather_dim of the tensor will be gathered.
        shard_dim(int, Optional): The shard_dim of the tensor will be sharded.
        logical_process_axis(Union(int, List[int]), Optional): The mesh_dim to implement the communication action.
    """

    def __init__(
        self,
        comm_pattern,
        sharding_spec,
        gather_dim=None,
        shard_dim=None,
        logical_process_axis=None,
        forward_only=False,
        mix_gather=False,
    ):
        self.comm_pattern = comm_pattern
        self.sharding_spec = sharding_spec
        self.gather_dim = gather_dim
        self.shard_dim = shard_dim
        self.logical_process_axis = logical_process_axis
        self.forward_only = forward_only
        if isinstance(self.logical_process_axis, list):
            if not mix_gather:
                self.device_mesh = self.sharding_spec.device_mesh.flatten()
                self.logical_process_axis = 0
            else:
                self.device_meshes = self.sharding_spec.device_mesh.flatten_device_meshes
                self.device_mesh = self.sharding_spec.device_mesh.flatten_device_mesh
                # Create a new member `logical_process_axes` to distinguish from original flatten
                self.logical_process_axes = logical_process_axis
        else:
            self.device_mesh = self.sharding_spec.device_mesh

    def __repr__(self):
        res_list = ["CommSpec:("]
        if self.comm_pattern == CollectiveCommPattern.GATHER_FWD_SPLIT_BWD:
            res_list.append(f"comm_pattern:GATHER_FWD_SPLIT_BWD, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"shard_dim:{self.shard_dim}, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD:
            res_list.append(f"comm_pattern:ALL2ALL_FWD_ALL2ALL_BWD, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"shard_dim:{self.shard_dim}, ")
            res_list.append(f"logical_process_axis: {self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.SPLIT_FWD_GATHER_BWD:
            res_list.append(f"comm_pattern:SPLIT_FWD_GATHER_BWD, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"shard_dim:{self.shard_dim}, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD:
            res_list.append(f"comm_pattern:ALLREDUCE_FWD_IDENTITY_BWD, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD:
            res_list.append(f"comm_pattern:IDENTITY_FWD_ALLREDUCE_BWD, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD:
            res_list.append(f"comm_pattern:MIXGATHER_FWD_SPLIT_BWD, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"logical_process_axes:{self.logical_process_axes})")

        return "".join(res_list)

    def get_comm_cost(self):
        """
        For all_gather, all2all, and all_reduce operation, the formula provided in DeviceMesh with alpha-beta model is used to
        compute the communication cost.
        For shard operation, it is an on-chip operation, so the communication cost is zero.
        """
        comm_size = reduce(operator.mul, self.sharding_spec.get_sharded_shape_per_device(), 1)
        cost_dict = {}
        if self.comm_pattern == CollectiveCommPattern.GATHER_FWD_SPLIT_BWD:
            forward_communication_cost = self.device_mesh.all_gather_cost(comm_size, self.logical_process_axis)
            # give a tiny cost to shard
            backward_communication_cost = 100

        if self.comm_pattern == CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD:
            forward_communication_cost = self.device_mesh.all_to_all_cost(comm_size, self.logical_process_axis)
            # grad should have same shape as input tensor
            # all to all operation has same logical process axis as forward.
            backward_communication_cost = self.device_mesh.all_to_all_cost(comm_size, self.logical_process_axis)

        if self.comm_pattern == CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD:
            forward_communication_cost = self.device_mesh.all_reduce_cost(comm_size, self.logical_process_axis)
            backward_communication_cost = 0

        if self.comm_pattern == CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD:
            forward_communication_cost = 0
            backward_communication_cost = self.device_mesh.all_reduce_cost(comm_size, self.logical_process_axis)

        if self.comm_pattern == CollectiveCommPattern.SPLIT_FWD_GATHER_BWD:
            # give a tiny cost to shard
            forward_communication_cost = 100
            backward_communication_cost = self.device_mesh.all_gather_cost(comm_size, self.logical_process_axis)

        if self.comm_pattern == CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD:
            # no need for axis because all devices are used in mix_gather
            forward_communication_cost = self.device_mesh.mix_gather_cost(comm_size)
            backward_communication_cost = 100

        if self.forward_only:
            cost_dict["forward"] = forward_communication_cost
            cost_dict["backward"] = 0
            cost_dict["total"] = cost_dict["forward"] + cost_dict["backward"]
        else:
            cost_dict["forward"] = forward_communication_cost
            cost_dict["backward"] = backward_communication_cost
            cost_dict["total"] = cost_dict["forward"] + cost_dict["backward"]

        return cost_dict

    def covert_spec_to_action(self, tensor):
        """
        Convert CommSpec into runtime action, implement real collection communication to target tensor.
        The collection communication action is directed by the CommSpec.

        Argument:
            tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
        """
        if self.comm_pattern in pattern_to_func_dict:
            tensor = pattern_to_func_dict[self.comm_pattern](tensor, self)
        else:
            tensor = tensor
        return tensor

