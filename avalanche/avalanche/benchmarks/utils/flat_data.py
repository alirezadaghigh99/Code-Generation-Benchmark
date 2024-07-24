class LazyIndices(Sequence[int]):
    """More efficient ops for indices.

    Avoid wasteful allocations, accept generators. Convert to list only
    when needed.

    Do not use for anything outside this file.
    """

    def __init__(
        self,
        *lists: Sequence[SupportsInt],
        known_length: Optional[SupportsInt] = None,
        offset: SupportsInt = 0,
    ):
        new_lists = []
        for ll in lists:
            if isinstance(ll, LazyIndices) and ll._eager_list is not None:
                # already eagerized, don't waste work
                new_lists.append(ll._eager_list)
            else:
                new_lists.append(ll)

        self._lists: Optional[List[Sequence[int]]] = (
            new_lists  # freed after eagerification
        )
        self._offset: int = int(offset)
        self._eager_list: Optional[np.ndarray] = None
        """This is the list where we save indices
        whenever we generate them from the lazy sequence.
        """

        self._known_length: int = (
            int(known_length)
            if known_length is not None
            else sum(len(ll) for ll in new_lists)
        )

        # check depth to avoid RecursionError
        if self._depth(stop_at_depth=3) > 2:
            self._to_eager()

    def _depth(self, *, stop_at_depth: Optional[int] = None, cur_depth: int = 0):
        """Return the depth of the LazyIndices tree.
        Use it only to eagerify early to avoid RecursionErrors.
        """
        if stop_at_depth is not None and cur_depth >= stop_at_depth:
            return 0

        if self._eager_list is not None:
            return 0

        assert self._lists is not None

        # List kept mostly for debugging purposes, can be replaced with a single int
        lens = [0]
        for ll in self._lists:
            if isinstance(ll, LazyIndices):
                lens.append(
                    ll._depth(stop_at_depth=stop_at_depth, cur_depth=cur_depth + 1)
                )

        return max(lens) + 1

    def _to_eager(self):
        if self._eager_list is not None:
            return
        assert self._lists is not None

        try:
            self._eager_list = np.empty(self._known_length, dtype=np.int64)
            my_offset = 0
            for lst in self._lists:
                if isinstance(lst, LazyRange):
                    self._eager_list[my_offset : my_offset + len(lst)] = np.arange(
                        len(lst), dtype=np.int64
                    ) + (lst._start + lst._offset + self._offset)
                elif isinstance(lst, LazyIndices):
                    lst._to_eager()
                    self._eager_list[my_offset : my_offset + len(lst)] = (
                        lst._eager_list + self._offset
                    )
                else:
                    self._eager_list[my_offset : my_offset + len(lst)] = lst
                    self._eager_list[my_offset : my_offset + len(lst)] += self._offset
                my_offset += len(lst)
            assert my_offset == self._known_length

            self._lists = None  # free memory
        except:
            self._eager_list = None
            raise

    def __getitem__(self, item):
        if self._eager_list is None:
            self._to_eager()
        assert self._eager_list is not None
        return int(self._eager_list[item])

    def __add__(self, other: Sequence[int]):
        return LazyIndices(self, other)

    def __radd__(self, other: Sequence[int]):
        return LazyIndices(other, self)

    def __len__(self):
        return self._known_length

class FlatData(IDataset[T_co], Sequence[T_co]):
    """FlatData is a dataset optimized for efficient repeated concatenation
    and subset operations.

    The class combines concatentation and subsampling operations in a single
    class.

    Class for internal use only. Users shuold use `AvalancheDataset` for data
    or `DataAttribute` for attributes such as class and task labels.

    *Notes for subclassing*

    Cat/Sub operations are "flattened" if possible, which means that they will
    take the datasets and indices from their arguments and create a new dataset
    with them, avoiding the creation of large trees of dataset (what would
    happen with PyTorch datasets). Flattening is not always possible, for
    example if the data contains additional info (e.g. custom transformation
    groups), so subclasses MUST set `can_flatten` properly in order to avoid
    nasty bugs.
    """

    def __init__(
        self,
        datasets: Sequence[IDataset[T_co]],
        indices: Optional[List[int]] = None,
        can_flatten: bool = True,
        discard_elements_not_in_indices: bool = False,
    ):
        """Constructor

        :param datasets: list of datasets to concatenate.
        :param indices:  list of indices.
        :param can_flatten: if True, enables flattening.
        :param discard_elements_not_in_indices: if True, will remove
            (drop the reference to) elements not in indices.
            Works only if all datasets are lists. If False (default),
            will use the standard subsetting approach of keeping
            the references to the original datasets.
            Setting this to True is useful when in need to keep
            raw data in memory (such as intermediate activations).
            For an example, please check how this is used
            in the :class:`ClassBalancedBufferWithLogits` of the
            :class:`DER` strategy.
        """
        self._datasets: List[IDataset[T_co]] = list(datasets)
        self._indices: Optional[List[int]] = indices
        self._can_flatten: bool = can_flatten
        self._discard_elements_not_in_indices: bool = discard_elements_not_in_indices

        if can_flatten:
            self._datasets = _flatten_dataset_list(self._datasets)
            self._datasets, self._indices = _flatten_datasets_and_reindex(
                self._datasets, self._indices
            )
        self._cumulative_sizes = ConcatDataset.cumsum(self._datasets)

        if discard_elements_not_in_indices:
            self._remove_unused_elements()

        # NOTE: check disabled to avoid slowing down OCL scenarios
        # # check indices
        # if self._indices is not None and len(self) > 0:
        #     assert min(self._indices) >= 0
        #     assert max(self._indices) < self._cumulative_sizes[-1], \
        #         f"Max index {max(self._indices)} is greater than datasets " \
        #         f"list size {self._cumulative_sizes[-1]}"

    def _remove_unused_elements(self):
        # print("[FlatData] Removing unused elements from FlatData")
        if self._indices is None:
            # print("[FlatData] Not subsetting...")
            return

        for dataset in self._datasets:
            if not isinstance(dataset, list):
                # print("Can't remove unused elements from non-list dataset")
                # TODO: support all iterables
                return

        shrinked_dataset = []

        for i in range(len(self)):
            dataset_idx, sample_idx = self._get_idx(i)
            shrinked_dataset.append(self._datasets[dataset_idx][sample_idx])

        # removed_count = sum(len(x) for x in self._datasets) - len(shrinked_dataset)
        # print(f"[FlatData] Removed {removed_count} elements from FlatData")

        self._datasets = [shrinked_dataset]
        self._indices = None
        self._cumulative_sizes = ConcatDataset.cumsum(self._datasets)

    def _get_lazy_indices(self):
        """This method creates indices on-the-fly if self._indices=None.
        Only for internal use. Call may be expensive if self._indices=None.
        """
        if self._indices is not None:
            return self._indices
        else:
            return LazyRange(0, len(self))

    def subset(self: TFlatData, indices: Optional[Iterable[int]]) -> TFlatData:
        """Subsampling operation.

        :param indices: indices of the new samples
        :return:
        """
        if indices is not None and not isinstance(indices, list):
            indices = list(indices)

        if self._can_flatten and indices is not None:
            if self._indices is None:
                new_indices = indices
            else:
                self_indices = self._get_lazy_indices()
                new_indices = [self_indices[x] for x in indices]
            return self.__class__(
                datasets=self._datasets,
                indices=new_indices,
                discard_elements_not_in_indices=self._discard_elements_not_in_indices,
            )
        return self.__class__(
            datasets=[self],
            indices=indices,
            discard_elements_not_in_indices=self._discard_elements_not_in_indices,
        )

    def concat(self: TFlatData, other: TFlatData) -> TFlatData:
        """Concatenation operation.

        :param other: other dataset.
        :return:
        """
        if (not self._can_flatten) and (not other._can_flatten):
            return self.__class__(datasets=[self, other])

        # Case 1: one is a subset of the other
        if len(self._datasets) == 1 and len(other._datasets) == 1:
            if (
                self._can_flatten
                and self._datasets[0] is other
                and other._indices is None
            ):
                idxs = self._get_lazy_indices() + other._get_lazy_indices()
                return other.subset(idxs)
            elif (
                other._can_flatten
                and other._datasets[0] is self
                and self._indices is None
            ):
                idxs = self._get_lazy_indices() + other._get_lazy_indices()
                return self.subset(idxs)
            elif (
                self._can_flatten
                and other._can_flatten
                and self._datasets[0] is other._datasets[0]
            ):
                idxs = LazyIndices(self._get_lazy_indices(), other._get_lazy_indices())
                return self.__class__(datasets=self._datasets, indices=idxs)

        # Case 2: at least one of them can be flattened
        if self._can_flatten and other._can_flatten:
            if self._indices is None and other._indices is None:
                new_indices = None
            else:
                if len(self._cumulative_sizes) == 0:
                    base_other = 0
                else:
                    base_other = self._cumulative_sizes[-1]
                other_idxs = LazyIndices(other._get_lazy_indices(), offset=base_other)
                new_indices = self._get_lazy_indices() + other_idxs
            return self.__class__(
                datasets=self._datasets + other._datasets, indices=new_indices
            )
        elif self._can_flatten:
            if self._indices is None and other._indices is None:
                new_indices = None
            else:
                if len(self._cumulative_sizes) == 0:
                    base_other = 0
                else:
                    base_other = self._cumulative_sizes[-1]
                other_idxs = LazyRange(0, len(other), offset=base_other)
                new_indices = self._get_lazy_indices() + other_idxs
            return self.__class__(
                datasets=self._datasets + [other], indices=new_indices
            )
        elif other._can_flatten:
            if self._indices is None and other._indices is None:
                new_indices = None
            else:
                base_other = len(self)
                self_idxs = LazyRange(0, len(self))
                other_idxs = LazyIndices(other._get_lazy_indices(), offset=base_other)
                new_indices = self_idxs + other_idxs
            return self.__class__(
                datasets=[self] + other._datasets, indices=new_indices
            )
        else:
            assert False, "should never get here"

    def _get_idx(self, idx) -> Tuple[int, int]:
        """Return the index as a tuple <dataset-idx, sample-idx>.

        The first index indicates the dataset to use from `self._datasets`,
        while the second is the index of the sample in
        `self._datasets[dataset-idx]`.

        Private method.
        """
        if idx >= len(self):
            raise IndexError()

        if self._indices is not None:  # subset indexing
            idx = self._indices[idx]
        if len(self._datasets) == 1:
            dataset_idx = 0
        else:  # concat indexing
            dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
            if dataset_idx == 0:
                idx = idx
            else:
                idx = idx - self._cumulative_sizes[dataset_idx - 1]
        return dataset_idx, int(idx)

    @overload
    def __getitem__(self, item: int) -> T_co: ...

    @overload
    def __getitem__(self: TFlatData, item: slice) -> TFlatData: ...

    def __getitem__(self: TFlatData, item: Union[int, slice]) -> Union[T_co, TFlatData]:
        if isinstance(item, (int, np.integer)):
            dataset_idx, idx = self._get_idx(int(item))
            return self._datasets[dataset_idx][idx]
        else:
            slice_indices = slice_alike_object_to_indices(
                slice_alike_object=item, max_length=len(self)
            )

            return self.subset(indices=slice_indices)

    def __len__(self) -> int:
        if len(self._cumulative_sizes) == 0:
            return 0
        elif self._indices is not None:
            return len(self._indices)
        return self._cumulative_sizes[-1]

    def __add__(self: TFlatData, other: TFlatData) -> TFlatData:
        return self.concat(other)

    def __radd__(self: TFlatData, other: TFlatData) -> TFlatData:
        return other.concat(self)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return _flatdata_repr(self)

    def _tree_depth(self):
        """Return the depth of the tree of datasets.
        Use only to debug performance issues.
        """
        return _flatdata_depth(self)

class LazyRange(LazyIndices):
    """Avoid 'eagerification' step for ranges."""

    def __init__(self, start: SupportsInt, end: SupportsInt, offset: SupportsInt = 0):
        self._start = int(start)
        self._end = int(end)
        self._offset = int(offset)
        self._known_length = self._end - self._start
        self._eager_list = self

    def _to_eager(self):
        # LazyRange is eager already
        pass

    def __iter__(self):
        for i in range(self._start, self._end):
            yield self._offset + i

    def __getitem__(self, item):
        assert item >= self._start and item < self._end, "LazyRange: index out of range"
        return self._start + self._offset + item

    def __add__(self, other):
        # TODO: could be optimized to merge contiguous ranges
        return LazyIndices(self, other)

    def __len__(self):
        return self._end - self._start

