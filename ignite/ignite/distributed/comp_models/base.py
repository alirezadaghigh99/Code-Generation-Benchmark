def _encode_input_data(x: Union[torch.Tensor, float, str, None], is_src: bool) -> List[int]:
        encoded_msg = [-1] * 512
        if not is_src:
            # Discard input type if not source
            return encoded_msg

        if isinstance(x, torch.Tensor):
            shape = x.shape
            dtype = str(x.dtype)
            msg = [0, len(shape), *shape, len(dtype), *list(bytearray(dtype, "utf-8"))]
            encoded_msg[: len(msg)] = msg
        elif isinstance(x, Number):
            encoded_msg[0] = 1
        elif isinstance(x, str):
            encoded_msg[0] = 2
        return encoded_msg

class _SerialModel(ComputationModel):
    """Private class defines non-distributed computation model for code compatibility with other distributed models."""

    name = "serial"
    available_backends = ()

    def __init__(self, _backend: Optional[str] = None, **kwargs: Any) -> None:
        super(_SerialModel, self).__init__()

    def get_local_rank(self) -> int:
        return 0

    def get_rank(self) -> int:
        return 0

    def get_world_size(self) -> int:
        return 1

    def get_nproc_per_node(self) -> int:
        return 1

    def get_nnodes(self) -> int:
        return 1

    def get_node_rank(self) -> int:
        return 0

    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _torch_version_gt_112 and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def backend(self) -> Optional[str]:
        return None

    def finalize(self) -> None:
        pass

    def _compute_nproc_per_node(self) -> int:
        return 1

    @staticmethod
    def create_from_context() -> "_SerialModel":
        return _SerialModel()

    @staticmethod
    def create_from_backend(backend: Optional[str] = None, **kwargs: Any) -> "_SerialModel":
        return _SerialModel()

    @staticmethod
    def spawn(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Serial computation model does not implement spawn method")

    def all_reduce(
        self, tensor: Union[torch.Tensor, float], op: str = "SUM", group: Optional[Any] = None
    ) -> Union[torch.Tensor, float]:
        return tensor

    def all_gather(
        self, tensor: Union[torch.Tensor, float, str, Any], group: Optional[Any] = None
    ) -> Union[torch.Tensor, float, List[float], List[str], List[Any]]:
        if isinstance(tensor, torch.Tensor):
            return tensor
        return cast(Union[List[float], List[str], List[Any]], [tensor])

    def broadcast(
        self, tensor: Union[torch.Tensor, float, str, None], src: int = 0, safe_mode: bool = False
    ) -> Union[torch.Tensor, float, str]:
        if tensor is None:
            raise ValueError("Argument tensor should not be None")
        return tensor

    def _do_all_reduce(self, tensor: torch.Tensor, op: str = "SUM", group: Optional[Any] = None) -> torch.Tensor:
        return tensor

    def _do_all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None) -> torch.Tensor:
        return tensor

    def _do_all_gather_object(self, tensor: Any, group: Optional[Any] = None) -> Any:
        return tensor

    def _do_new_group(self, ranks: List[int], **kwargs: Any) -> Any:
        return ranks

    def _do_broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        return tensor

    def barrier(self) -> None:
        pass

    def new_group(self, ranks: List[int], **kwargs: Any) -> Any:
        if isinstance(ranks, list) and all(isinstance(item, int) for item in ranks):
            return self._do_new_group(ranks, **kwargs)
        else:
            raise ValueError("Argument ranks should be list of int")

