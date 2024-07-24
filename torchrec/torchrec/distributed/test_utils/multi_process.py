class MultiProcessContext:
    def __init__(
        self,
        rank: int,
        world_size: int,
        backend: str = "gloo",
        local_size: Optional[int] = None,
        use_deterministic_algorithms: bool = True,
        disable_cuda_tf_32: bool = True,
    ) -> None:

        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.local_size = local_size
        self.disable_cuda_tf_32 = disable_cuda_tf_32

        if torch.cuda.is_available() and world_size <= torch.cuda.device_count():
            self.device: torch.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(self.device)

            if self.disable_cuda_tf_32:
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
        else:
            self.device: torch.device = torch.device("cpu")

        if use_deterministic_algorithms:
            if torch.cuda.is_available():
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
            torch.use_deterministic_algorithms(True)

        self.pg: Optional[dist.ProcessGroup] = None

    def __enter__(self) -> "MultiProcessContext":
        """
        Override local_size after pg construction because unit test device count is
        larger than local_size setup. This can be problematic for twrw because we have
        ShardedTensor placement check.

        TODO (T108556130) Mock out functions in comm.py instead of overriding env vars
        """

        os.environ["LOCAL_WORLD_SIZE"] = str(self.local_size or self.world_size)
        if self.local_size is not None:
            os.environ["LOCAL_RANK"] = str(self.rank % self.local_size)

        self.pg = init_distributed_single_host(
            rank=self.rank,
            world_size=self.world_size,
            backend=self.backend,
            local_size=self.local_size,
        )
        return self

    # pyre-ignore
    def __exit__(self, exc_type, exc_instance, traceback) -> None:
        if _INTRA_PG is not None:
            dist.destroy_process_group(_INTRA_PG)
        if _CROSS_PG is not None:
            dist.destroy_process_group(_CROSS_PG)
        dist.destroy_process_group(self.pg)
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available() and self.disable_cuda_tf_32:
            torch.backends.cudnn.allow_tf32 = True

