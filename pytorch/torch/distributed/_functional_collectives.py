def all_reduce(self: torch.Tensor, reduceOp: str, group: RANK_TYPES, tag: str = ""):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    The input tensor is left unmodified.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group_name = _resolve_group_name(group, tag)
    tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
    return _maybe_wrap_tensor(tensor)

def _expand_group(group: RANK_TYPES, tag: str = "") -> Tuple[str, List[int], int]:
    """
    _expand_group desugars the different RANK_TYPES types into a canonical format that is traceable.

    By having this be part of the explicit eager codepath, we avoid having to specialize behavior inside
    torchdynamo and can still interoperate with processgroup objects or other untraceable forms.
    """
    # had to define this hack _inside_ expand_group to avoid
    # graph_break [('torch.* op returned non-Tensor int
    # caused by 'cast_*` functions being treated as 'torch.*' ops (iiuc)
    if TYPE_CHECKING:

        def cast_listlistint(x):
            return cast(List[List[int]], x)

        def cast_listint(x):
            return cast(List[int], x)

    else:
        # fake cast op for use at runtime since dynamo doesn't support real cast
        # also, dynamo didn't like encountering 'typing' objects ()
        # NotImplementedError: argument of type: <class 'typing._GenericAlias'>
        def cast_listlistint(x):
            return x

        def cast_listint(x):
            return x

    rankset: List[int]
    if isinstance(group, list):
        if isinstance(group[0], list):
            nested_list = cast_listlistint(group)
            rankset = []
            group_size = -1
            for rs in nested_list:
                rankset.extend(rs)
                if group_size != -1 and group_size != len(rs):
                    raise ValueError(
                        f"group sizes must be identical found {group_size} and {len(rs)}"
                    )
                group_size = len(rs)
        else:
            rankset = cast_listint(group)
            group_size = len(rankset)
    elif isinstance(group, dist.ProcessGroup):
        rankset = dist.get_process_group_ranks(group)
        group_size = len(rankset)
        tag = tag or c10d._get_group_tag(group)
    elif isinstance(group, DeviceMesh):
        assert (
            group.ndim == 1
        ), "Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D"
        # TODO: it should run collective in the whole mesh instead of dim 0
        tag, rankset, _ = group._dim_group_infos[0]
        group_size = len(rankset)
    elif isinstance(group, tuple):
        if (
            len(group) == 2
            and isinstance(group[0], DeviceMesh)
            and isinstance(group[1], int)
        ):
            dmesh = group[0]
            dim = group[1]
            tag, rankset, _ = dmesh._dim_group_infos[dim]
            group_size = len(rankset)
        else:
            raise ValueError("Invalid tuple for group must be (DeviceMesh, int)")
    else:
        raise ValueError(
            "Invalid type for group, must be one of List, Processgroup, DeviceMesh or (DeviceMesh, int)."
        )

    return (tag, rankset, group_size)

