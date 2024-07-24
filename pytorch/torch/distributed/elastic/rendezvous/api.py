class RendezvousError(Exception):
    """Represents the base type for rendezvous errors."""

class RendezvousParameters:
    """Hold the parameters to construct a :py:class:`RendezvousHandler`.

    Args:
        backend:
            The name of the backend to use to handle the rendezvous.
        endpoint:
            The endpoint of the rendezvous, usually in form <hostname>[:<port>].
        run_id:
            The id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        local_addr:
            The address of the local node.
        **kwargs:
            Additional parameters for the specified backend.
    """

    def __init__(
        self,
        backend: str,
        endpoint: str,
        run_id: str,
        min_nodes: int,
        max_nodes: int,
        local_addr: Optional[str] = None,
        **kwargs,
    ):
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")

        if min_nodes < 1:
            raise ValueError(
                f"The minimum number of rendezvous nodes ({min_nodes}) must be greater than zero."
            )
        if max_nodes < min_nodes:
            raise ValueError(
                f"The maximum number of rendezvous nodes ({max_nodes}) must be greater than or "
                f"equal to the minimum number of rendezvous nodes ({min_nodes})."
            )

        self.backend = backend
        self.endpoint = endpoint
        self.run_id = run_id
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.config = kwargs
        self.local_addr = local_addr

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` if ``key`` exists, else ``default``."""
        return self.config.get(key, default)

    def get_as_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Return the value for ``key`` as a ``bool``."""
        value = self.get(key, default)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value == 1:
                return True
            if value == 0:
                return False
        elif isinstance(value, str):
            if value.lower() in ["1", "true", "t", "yes", "y"]:
                return True
            if value.lower() in ["0", "false", "f", "no", "n"]:
                return False
        raise ValueError(
            f"The rendezvous configuration option '{key}' does not represent a valid boolean value."
        )

    def get_as_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Return the value for ``key`` as an ``int``."""
        value = self.get(key, default)
        if value is None:
            return value
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(
                f"The rendezvous configuration option '{key}' does not represent a valid integer "
                "value."
            ) from e

class RendezvousGracefulExitError(RendezvousError):
    """Raised when node wasn't not included in rendezvous and gracefully exits.

    Exception is a mechanism to exit the stack, however does not mean a failure.
    """

class RendezvousStoreInfo:
    """Store address and port that can be used to bootstrap trainer distributed comms"""

    MASTER_ADDR_KEY: ClassVar[str] = "MASTER_ADDR"
    MASTER_PORT_KEY: ClassVar[str] = "MASTER_PORT"
    master_addr: str
    master_port: int

    @staticmethod
    def build(rank: int, store: Store) -> "RendezvousStoreInfo":
        """Factory method, finds unused new port on rank0 host and addr/port info with all ranks.

        If master_addr/master_port is knowns (useful when sharing existing tcp store server) use the constructor.
        """
        # TODO swap to collectives comms API
        if rank == 0:
            addr = socket.getfqdn()
            port = _get_free_port()
            store.set(RendezvousStoreInfo.MASTER_ADDR_KEY, addr.encode(encoding="UTF-8"))  # type: ignore[arg-type]
            store.set(RendezvousStoreInfo.MASTER_PORT_KEY, str(port).encode(encoding="UTF-8"))  # type: ignore[arg-type]

        addr = store.get(RendezvousStoreInfo.MASTER_ADDR_KEY).decode(encoding="UTF-8")
        port = int(
            store.get(RendezvousStoreInfo.MASTER_PORT_KEY).decode(encoding="UTF-8")
        )
        return RendezvousStoreInfo(master_addr=addr, master_port=port)

