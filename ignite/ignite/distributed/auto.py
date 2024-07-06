def auto_dataloader(dataset: Dataset, **kwargs: Any) -> Union[DataLoader, "_MpDeviceLoader"]:
    """Helper method to create a dataloader adapted for non-distributed and distributed configurations (supporting
    all available backends from :meth:`~ignite.distributed.utils.available_backends()`).

    Internally, we create a dataloader with provided kwargs while applying the following updates:

    - batch size is scaled by world size: ``batch_size / world_size`` if larger or equal world size.
    - number of workers is scaled by number of local processes: ``num_workers / nprocs`` if larger or equal world size.
    - if no sampler provided by user, a `torch DistributedSampler`_ is setup.
    - if a `torch DistributedSampler`_ is provided by user, it is used without wrapping it.
    - if another sampler is provided, it is wrapped by :class:`~ignite.distributed.auto.DistributedProxySampler`.
    - if the default device is 'cuda', `pin_memory` is automatically set to `True`.

    .. warning::

        Custom batch sampler is not adapted for distributed configuration. Please, make sure that provided batch
        sampler is compatible with distributed configuration.

    Args:
        dataset: input torch dataset. If input dataset is `torch IterableDataset`_ then dataloader will be
            created without any distributed sampling. Please, make sure that the dataset itself produces
            different data on different ranks.
        kwargs: keyword arguments for `torch DataLoader`_.

    Returns:
        `torch DataLoader`_ or `XLA MpDeviceLoader`_ for XLA devices

    Examples:
        .. code-block:: python

            import ignite.distribted as idist

            train_loader = idist.auto_dataloader(
                train_dataset,
                batch_size=32,
                num_workers=4,
                shuffle=True,
                pin_memory="cuda" in idist.device().type,
                drop_last=True,
            )

    .. _torch DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    .. _XLA MpDeviceLoader:
        https://pytorch.org/xla/release/2.0/index.html#running-on-multiple-xla-devices-with-multi-processing
    .. _torch DistributedSampler:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    .. _torch IterableDataset: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """
    rank = idist.get_rank()
    world_size = idist.get_world_size()

    logger = setup_logger(__name__ + ".auto_dataloader")
    if world_size > 1:
        if "batch_size" in kwargs and kwargs["batch_size"] >= world_size:
            kwargs["batch_size"] //= world_size

        nproc = idist.get_nproc_per_node()
        if "num_workers" in kwargs and kwargs["num_workers"] >= nproc:
            kwargs["num_workers"] = (kwargs["num_workers"] + nproc - 1) // nproc

        if "batch_sampler" not in kwargs:
            if isinstance(dataset, IterableDataset):
                logger.info(
                    "Found iterable dataset, dataloader will be created without any distributed sampling. "
                    "Please, make sure that the dataset itself produces different data on different ranks."
                )
            else:
                sampler: Optional[Union[DistributedProxySampler, DistributedSampler, Sampler]]
                sampler = kwargs.get("sampler", None)
                if isinstance(sampler, DistributedSampler):
                    if sampler.rank != rank:
                        warnings.warn(f"Found distributed sampler with rank={sampler.rank}, but process rank is {rank}")
                    if sampler.num_replicas != world_size:
                        warnings.warn(
                            f"Found distributed sampler with num_replicas={sampler.num_replicas}, "
                            f"but world size is {world_size}"
                        )
                elif sampler is None:
                    # removes "shuffle" from kwargs if sampler is used
                    shuffle = kwargs.pop("shuffle", True)
                    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
                else:
                    sampler = DistributedProxySampler(sampler, num_replicas=world_size, rank=rank)
                kwargs["sampler"] = sampler
        else:
            warnings.warn(
                "Found batch_sampler in provided kwargs. Please, make sure that it is compatible "
                "with distributed configuration"
            )

    if idist.has_xla_support and idist.backend() == idist_xla.XLA_TPU and kwargs.get("pin_memory", False):
        # TODO: How about XLA GPU ?
        warnings.warn(
            "Found incompatible options: xla support and pin_memory args equal True. "
            "Argument `pin_memory=False` will be used to construct data loader."
        )
        kwargs["pin_memory"] = False
    else:
        kwargs["pin_memory"] = kwargs.get("pin_memory", "cuda" in idist.device().type)

    logger.info(f"Use data loader kwargs for dataset '{repr(dataset)[:20].strip()}': \n\t{kwargs}")
    dataloader = DataLoader(dataset, **kwargs)

    if idist.has_xla_support and idist.backend() == idist_xla.XLA_TPU and world_size > 1:
        logger.info("DataLoader is wrapped by `MpDeviceLoader` on XLA")

        mp_device_loader_cls = _MpDeviceLoader
        try:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader

            mp_device_loader_cls = MpDeviceLoader
        except ImportError:
            pass

        mp_dataloader = mp_device_loader_cls(dataloader, idist.device())
        mp_dataloader.sampler = dataloader.sampler  # type: ignore[attr-defined]
        return mp_dataloader

    return dataloader

