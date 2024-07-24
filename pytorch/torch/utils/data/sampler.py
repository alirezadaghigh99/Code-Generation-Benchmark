class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, but got replacement={self.replacement}"
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[
                : self.num_samples % n
            ]

    def __len__(self) -> int:
        return self.num_samples

class Sampler(Generic[_T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices or lists of indices (batches) of dataset elements,
    and may provide a :meth:`__len__` method that returns the length of the returned iterators.

    Args:
        data_source (Dataset): This argument is not used and will be removed in 2.2.0.
            You may still have custom implementation that utilizes it.

    Example:
        >>> # xdoctest: +SKIP
        >>> class AccedingSequenceLengthSampler(Sampler[int]):
        >>>     def __init__(self, data: List[str]) -> None:
        >>>         self.data = data
        >>>
        >>>     def __len__(self) -> int:
        >>>         return len(self.data)
        >>>
        >>>     def __iter__(self) -> Iterator[int]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         yield from torch.argsort(sizes).tolist()
        >>>
        >>> class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
        >>>     def __init__(self, data: List[str], batch_size: int) -> None:
        >>>         self.data = data
        >>>         self.batch_size = batch_size
        >>>
        >>>     def __len__(self) -> int:
        >>>         return (len(self.data) + self.batch_size - 1) // self.batch_size
        >>>
        >>>     def __iter__(self) -> Iterator[List[int]]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         for batch in torch.chunk(torch.argsort(sizes), len(self)):
        >>>             yield batch.tolist()

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized] = None) -> None:
        if data_source is not None:
            import warnings

            warnings.warn(
                "`data_source` argument is not used and will be removed in 2.2.0."
                "You may still have custom implementation that utilizes it."
            )

    def __iter__(self) -> Iterator[_T_co]:
        raise NotImplementedError

