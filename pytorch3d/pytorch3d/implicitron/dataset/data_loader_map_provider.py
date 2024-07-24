class DoublePoolBatchSampler(Sampler[List[int]]):
    """
    Batch sampler for making random batches of a single frame
    from one list and a number of known frames from another list.
    """

    def __init__(
        self,
        first_indices: List[int],
        rest_indices: List[int],
        batch_size: int,
        replacement: bool,
        num_batches: Optional[int] = None,
    ) -> None:
        """
        Args:
            first_indices: indexes of dataset items to use as the first element
                        of each batch.
            rest_indices: indexes of dataset items to use as the subsequent
                        elements of each batch. Not used if batch_size==1.
            batch_size: The common size of any batch.
            replacement: Whether the sampling of first items is with replacement.
            num_batches: The number of batches in an epoch. If 0 or None,
                        one epoch is the length of `first_indices`.
        """
        self.first_indices = first_indices
        self.rest_indices = rest_indices
        self.batch_size = batch_size
        self.replacement = replacement
        self.num_batches = None if num_batches == 0 else num_batches

        if batch_size - 1 > len(rest_indices):
            raise ValueError(
                f"Cannot make up ({batch_size})-batches from {len(self.rest_indices)}"
            )

        # copied from RandomSampler
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __len__(self) -> int:
        if self.num_batches is not None:
            return self.num_batches
        return len(self.first_indices)

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = self.num_batches
        if self.replacement:
            i_first = torch.randint(
                len(self.first_indices),
                size=(len(self),),
                generator=self.generator,
            )
        elif num_batches is not None:
            n_copies = 1 + (num_batches - 1) // len(self.first_indices)
            raw_indices = [
                torch.randperm(len(self.first_indices), generator=self.generator)
                for _ in range(n_copies)
            ]
            i_first = torch.cat(raw_indices)[:num_batches]
        else:
            i_first = torch.randperm(len(self.first_indices), generator=self.generator)
        first_indices = [self.first_indices[i] for i in i_first]

        if self.batch_size == 1:
            for first_index in first_indices:
                yield [first_index]
            return

        for first_index in first_indices:
            # Consider using this class in a program which sets the seed. This use
            # of randperm means that rerunning with a higher batch_size
            # results in batches whose first elements as the first run.
            i_rest = torch.randperm(
                len(self.rest_indices),
                generator=self.generator,
            )[: self.batch_size - 1]
            yield [first_index] + [self.rest_indices[i] for i in i_rest]

