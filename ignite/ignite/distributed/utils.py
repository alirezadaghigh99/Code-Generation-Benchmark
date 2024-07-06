def device() -> torch.device:
    """Returns current device according to current distributed configuration.

    - `torch.device("cpu")` if no distributed configuration or torch native gloo distributed configuration
    - `torch.device("cuda:local_rank")` if torch native nccl or horovod distributed configuration
    - `torch.device("xla:index")` if XLA distributed configuration

    Returns:
        torch.device

    .. versionchanged:: 0.4.2
        Added Horovod distributed framework.
    """
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.device()

def get_rank() -> int:
    """Returns process rank within current distributed configuration. Returns 0 if no distributed configuration."""
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.get_rank()

def spawn(
    backend: str,
    fn: Callable,
    args: Tuple,
    kwargs_dict: Optional[Mapping] = None,
    nproc_per_node: int = 1,
    **kwargs: Any,
) -> None:
    """Spawns ``nproc_per_node`` processes that run ``fn`` with ``args``/``kwargs_dict`` and initialize
    distributed configuration defined by ``backend``.

    Args:
        backend: backend to use: `nccl`, `gloo`, `xla-tpu`, `horovod`
        fn: function to called as the entrypoint of the spawned process.
            This function must be defined at the top level of a module so it can be pickled and spawned.
            This is a requirement imposed by multiprocessing. The function is called as ``fn(i, *args, **kwargs_dict)``,
            where `i` is the process index and args is the passed through tuple of arguments.
        args: arguments passed to `fn`.
        kwargs_dict: kwargs passed to `fn`.
        nproc_per_node: number of processes to spawn on a single node. Default, 1.
        kwargs: acceptable kwargs according to provided backend:

            - | "nccl" or "gloo" : ``nnodes`` (default, 1), ``node_rank`` (default, 0), ``master_addr``
              | (default, "127.0.0.1"), ``master_port`` (default, 2222), ``init_method`` (default, "env://"),
              | `timeout` to `dist.init_process_group`_ function
              | and kwargs for `mp.start_processes`_ function.

            - | "xla-tpu" : ``nnodes`` (default, 1), ``node_rank`` (default, 0) and kwargs to `xmp.spawn`_ function.

            - | "horovod": ``hosts`` (default, None) and other kwargs to `hvd_run`_ function. Arguments ``nnodes=1``
              | and ``node_rank=0`` are tolerated and ignored, otherwise an exception is raised.

    Examples:
        1) Launch single node multi-GPU training using torch native distributed framework

        .. code-block:: python

            # >>> python main.py

            # main.py

            import ignite.distributed as idist

            def train_fn(local_rank, a, b, c, d=12):
                import torch.distributed as dist
                assert dist.is_available() and dist.is_initialized()
                assert dist.get_world_size() == 4

                device = idist.device()
                assert device == torch.device(f"cuda:{local_rank}")


            idist.spawn("nccl", train_fn, args=(a, b, c), kwargs_dict={"d": 23}, nproc_per_node=4)


        2) Launch multi-node multi-GPU training using torch native distributed framework

        .. code-block:: python

            # >>> (node 0): python main.py --node_rank=0 --nnodes=8 --master_addr=master --master_port=2222
            # >>> (node 1): python main.py --node_rank=1 --nnodes=8 --master_addr=master --master_port=2222
            # >>> ...
            # >>> (node 7): python main.py --node_rank=7 --nnodes=8 --master_addr=master --master_port=2222

            # main.py

            import torch
            import ignite.distributed as idist

            def train_fn(local_rank, nnodes, nproc_per_node):
                import torch.distributed as dist
                assert dist.is_available() and dist.is_initialized()
                assert dist.get_world_size() == nnodes * nproc_per_node

                device = idist.device()
                assert device == torch.device(f"cuda:{local_rank}")

            idist.spawn(
                "nccl",
                train_fn,
                args=(nnodes, nproc_per_node),
                nproc_per_node=nproc_per_node,
                nnodes=nnodes,
                node_rank=node_rank,
                master_addr=master_addr,
                master_port=master_port
            )

        3) Launch single node multi-TPU training (for example on Google Colab) using PyTorch/XLA

        .. code-block:: python

            # >>> python main.py

            # main.py

            import ignite.distributed as idist

            def train_fn(local_rank, a, b, c, d=12):
                import torch_xla.core.xla_model as xm
                assert xm.get_world_size() == 8

                device = idist.device()
                assert "xla" in device.type


            idist.spawn("xla-tpu", train_fn, args=(a, b, c), kwargs_dict={"d": 23}, nproc_per_node=8)

    .. _dist.init_process_group: https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    .. _mp.start_processes: https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn.spawn
    .. _xmp.spawn: https://pytorch.org/xla/release/1.6/index.html#torch_xla.distributed.xla_multiprocessing.spawn
    .. _hvd_run: https://horovod.readthedocs.io/en/latest/api.html#module-horovod.run

    .. versionchanged:: 0.4.2
        ``backend`` now accepts `horovod` distributed framework.
    """
    _assert_backend(backend)

    if kwargs_dict is None:
        kwargs_dict = {}

    for comp_model_cls in registered_computation_models:
        if backend not in comp_model_cls.available_backends:
            continue
        comp_model_cls.spawn(
            fn, args=args, kwargs_dict=kwargs_dict, nproc_per_node=nproc_per_node, backend=backend, **kwargs
        )

def get_world_size() -> int:
    """Returns world size of current distributed configuration. Returns 1 if no distributed configuration."""
    if _need_to_sync and isinstance(_model, _SerialModel):
        sync(temporary=True)

    return _model.get_world_size()

def sync(temporary: bool = False) -> None:
    """Helper method to force this module to synchronize with current distributed context.
    This method should be used when distributed context is manually created or destroyed.

    Args:
        temporary: If True, distributed model synchronization is done every call of ``idist.get_*`` methods.
            This may have a negative performance impact.
    """
    global _model

    for comp_model_cls in registered_computation_models:
        if comp_model_cls == _SerialModel:
            continue
        model = comp_model_cls.create_from_context()
        if model is not None:
            _set_model(model, temporary=temporary)
            return

    _model = _SerialModel()

def spawn(
    backend: str,
    fn: Callable,
    args: Tuple,
    kwargs_dict: Optional[Mapping] = None,
    nproc_per_node: int = 1,
    **kwargs: Any,
) -> None:
    """Spawns ``nproc_per_node`` processes that run ``fn`` with ``args``/``kwargs_dict`` and initialize
    distributed configuration defined by ``backend``.

    Args:
        backend: backend to use: `nccl`, `gloo`, `xla-tpu`, `horovod`
        fn: function to called as the entrypoint of the spawned process.
            This function must be defined at the top level of a module so it can be pickled and spawned.
            This is a requirement imposed by multiprocessing. The function is called as ``fn(i, *args, **kwargs_dict)``,
            where `i` is the process index and args is the passed through tuple of arguments.
        args: arguments passed to `fn`.
        kwargs_dict: kwargs passed to `fn`.
        nproc_per_node: number of processes to spawn on a single node. Default, 1.
        kwargs: acceptable kwargs according to provided backend:

            - | "nccl" or "gloo" : ``nnodes`` (default, 1), ``node_rank`` (default, 0), ``master_addr``
              | (default, "127.0.0.1"), ``master_port`` (default, 2222), ``init_method`` (default, "env://"),
              | `timeout` to `dist.init_process_group`_ function
              | and kwargs for `mp.start_processes`_ function.

            - | "xla-tpu" : ``nnodes`` (default, 1), ``node_rank`` (default, 0) and kwargs to `xmp.spawn`_ function.

            - | "horovod": ``hosts`` (default, None) and other kwargs to `hvd_run`_ function. Arguments ``nnodes=1``
              | and ``node_rank=0`` are tolerated and ignored, otherwise an exception is raised.

    Examples:
        1) Launch single node multi-GPU training using torch native distributed framework

        .. code-block:: python

            # >>> python main.py

            # main.py

            import ignite.distributed as idist

            def train_fn(local_rank, a, b, c, d=12):
                import torch.distributed as dist
                assert dist.is_available() and dist.is_initialized()
                assert dist.get_world_size() == 4

                device = idist.device()
                assert device == torch.device(f"cuda:{local_rank}")


            idist.spawn("nccl", train_fn, args=(a, b, c), kwargs_dict={"d": 23}, nproc_per_node=4)


        2) Launch multi-node multi-GPU training using torch native distributed framework

        .. code-block:: python

            # >>> (node 0): python main.py --node_rank=0 --nnodes=8 --master_addr=master --master_port=2222
            # >>> (node 1): python main.py --node_rank=1 --nnodes=8 --master_addr=master --master_port=2222
            # >>> ...
            # >>> (node 7): python main.py --node_rank=7 --nnodes=8 --master_addr=master --master_port=2222

            # main.py

            import torch
            import ignite.distributed as idist

            def train_fn(local_rank, nnodes, nproc_per_node):
                import torch.distributed as dist
                assert dist.is_available() and dist.is_initialized()
                assert dist.get_world_size() == nnodes * nproc_per_node

                device = idist.device()
                assert device == torch.device(f"cuda:{local_rank}")

            idist.spawn(
                "nccl",
                train_fn,
                args=(nnodes, nproc_per_node),
                nproc_per_node=nproc_per_node,
                nnodes=nnodes,
                node_rank=node_rank,
                master_addr=master_addr,
                master_port=master_port
            )

        3) Launch single node multi-TPU training (for example on Google Colab) using PyTorch/XLA

        .. code-block:: python

            # >>> python main.py

            # main.py

            import ignite.distributed as idist

            def train_fn(local_rank, a, b, c, d=12):
                import torch_xla.core.xla_model as xm
                assert xm.get_world_size() == 8

                device = idist.device()
                assert "xla" in device.type


            idist.spawn("xla-tpu", train_fn, args=(a, b, c), kwargs_dict={"d": 23}, nproc_per_node=8)

    .. _dist.init_process_group: https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    .. _mp.start_processes: https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn.spawn
    .. _xmp.spawn: https://pytorch.org/xla/release/1.6/index.html#torch_xla.distributed.xla_multiprocessing.spawn
    .. _hvd_run: https://horovod.readthedocs.io/en/latest/api.html#module-horovod.run

    .. versionchanged:: 0.4.2
        ``backend`` now accepts `horovod` distributed framework.
    """
    _assert_backend(backend)

    if kwargs_dict is None:
        kwargs_dict = {}

    for comp_model_cls in registered_computation_models:
        if backend not in comp_model_cls.available_backends:
            continue
        comp_model_cls.spawn(
            fn, args=args, kwargs_dict=kwargs_dict, nproc_per_node=nproc_per_node, backend=backend, **kwargs
        )

def spawn(
    backend: str,
    fn: Callable,
    args: Tuple,
    kwargs_dict: Optional[Mapping] = None,
    nproc_per_node: int = 1,
    **kwargs: Any,
) -> None:
    """Spawns ``nproc_per_node`` processes that run ``fn`` with ``args``/``kwargs_dict`` and initialize
    distributed configuration defined by ``backend``.

    Args:
        backend: backend to use: `nccl`, `gloo`, `xla-tpu`, `horovod`
        fn: function to called as the entrypoint of the spawned process.
            This function must be defined at the top level of a module so it can be pickled and spawned.
            This is a requirement imposed by multiprocessing. The function is called as ``fn(i, *args, **kwargs_dict)``,
            where `i` is the process index and args is the passed through tuple of arguments.
        args: arguments passed to `fn`.
        kwargs_dict: kwargs passed to `fn`.
        nproc_per_node: number of processes to spawn on a single node. Default, 1.
        kwargs: acceptable kwargs according to provided backend:

            - | "nccl" or "gloo" : ``nnodes`` (default, 1), ``node_rank`` (default, 0), ``master_addr``
              | (default, "127.0.0.1"), ``master_port`` (default, 2222), ``init_method`` (default, "env://"),
              | `timeout` to `dist.init_process_group`_ function
              | and kwargs for `mp.start_processes`_ function.

            - | "xla-tpu" : ``nnodes`` (default, 1), ``node_rank`` (default, 0) and kwargs to `xmp.spawn`_ function.

            - | "horovod": ``hosts`` (default, None) and other kwargs to `hvd_run`_ function. Arguments ``nnodes=1``
              | and ``node_rank=0`` are tolerated and ignored, otherwise an exception is raised.

    Examples:
        1) Launch single node multi-GPU training using torch native distributed framework

        .. code-block:: python

            # >>> python main.py

            # main.py

            import ignite.distributed as idist

            def train_fn(local_rank, a, b, c, d=12):
                import torch.distributed as dist
                assert dist.is_available() and dist.is_initialized()
                assert dist.get_world_size() == 4

                device = idist.device()
                assert device == torch.device(f"cuda:{local_rank}")


            idist.spawn("nccl", train_fn, args=(a, b, c), kwargs_dict={"d": 23}, nproc_per_node=4)


        2) Launch multi-node multi-GPU training using torch native distributed framework

        .. code-block:: python

            # >>> (node 0): python main.py --node_rank=0 --nnodes=8 --master_addr=master --master_port=2222
            # >>> (node 1): python main.py --node_rank=1 --nnodes=8 --master_addr=master --master_port=2222
            # >>> ...
            # >>> (node 7): python main.py --node_rank=7 --nnodes=8 --master_addr=master --master_port=2222

            # main.py

            import torch
            import ignite.distributed as idist

            def train_fn(local_rank, nnodes, nproc_per_node):
                import torch.distributed as dist
                assert dist.is_available() and dist.is_initialized()
                assert dist.get_world_size() == nnodes * nproc_per_node

                device = idist.device()
                assert device == torch.device(f"cuda:{local_rank}")

            idist.spawn(
                "nccl",
                train_fn,
                args=(nnodes, nproc_per_node),
                nproc_per_node=nproc_per_node,
                nnodes=nnodes,
                node_rank=node_rank,
                master_addr=master_addr,
                master_port=master_port
            )

        3) Launch single node multi-TPU training (for example on Google Colab) using PyTorch/XLA

        .. code-block:: python

            # >>> python main.py

            # main.py

            import ignite.distributed as idist

            def train_fn(local_rank, a, b, c, d=12):
                import torch_xla.core.xla_model as xm
                assert xm.get_world_size() == 8

                device = idist.device()
                assert "xla" in device.type


            idist.spawn("xla-tpu", train_fn, args=(a, b, c), kwargs_dict={"d": 23}, nproc_per_node=8)

    .. _dist.init_process_group: https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    .. _mp.start_processes: https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn.spawn
    .. _xmp.spawn: https://pytorch.org/xla/release/1.6/index.html#torch_xla.distributed.xla_multiprocessing.spawn
    .. _hvd_run: https://horovod.readthedocs.io/en/latest/api.html#module-horovod.run

    .. versionchanged:: 0.4.2
        ``backend`` now accepts `horovod` distributed framework.
    """
    _assert_backend(backend)

    if kwargs_dict is None:
        kwargs_dict = {}

    for comp_model_cls in registered_computation_models:
        if backend not in comp_model_cls.available_backends:
            continue
        comp_model_cls.spawn(
            fn, args=args, kwargs_dict=kwargs_dict, nproc_per_node=nproc_per_node, backend=backend, **kwargs
        )

