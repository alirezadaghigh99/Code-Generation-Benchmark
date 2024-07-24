class SizeBasedChunker(Chunker):
    """A Chunker that will split data into Chunks based on the preferred number of observations per Chunk.

    Notes
    -----
    - Chunks are adjacent, not overlapping
    - There may be "incomplete" chunks, as the remainder of observations after dividing by `chunk_size`
      will form a chunk of their own.

    Examples
    --------
    Chunk using monthly periods and providing a column name

    >>> from nannyml.chunk import SizeBasedChunker
    >>> df = pd.read_parquet('/path/to/my/data.pq')
    >>> chunker = SizeBasedChunker(chunk_size=2000, incomplete='drop')
    >>> chunks = chunker.split(data=df)

    """

    def __init__(self, chunk_size: int, incomplete: str = 'keep', timestamp_column_name: Optional[str] = None):
        """Create a new SizeBasedChunker.

        Parameters
        ----------
        chunk_size: int
            The preferred size of the resulting Chunks, i.e. the number of observations in each Chunk.
        incomplete: str, default='keep'
            Choose how to handle any leftover observations that don't make up a full Chunk.
            The following options are available:

            - ``'drop'``: drop the leftover observations

            - ``'keep'``: keep the incomplete Chunk (containing less than ``chunk_size`` observations)

            - ``'append'``: append leftover observations to the last complete Chunk (overfilling it)

            Defaults to ``'keep'``.

        Returns
        -------
        chunker: a size-based instance used to split data into Chunks of a constant size.

        """
        super().__init__(timestamp_column_name)

        # TODO wording
        if not isinstance(chunk_size, (int, np.int64)):
            raise InvalidArgumentsException(
                f"given chunk_size is of type {type(chunk_size)} but should be an int."
                f"Please provide an integer as a chunk size"
            )

        # TODO wording
        if chunk_size <= 0:
            raise InvalidArgumentsException(
                f"given chunk_size {chunk_size} is less then or equal to zero."
                f"The chunk size should always be larger then zero"
            )

        self.chunk_size = chunk_size
        self.incomplete = incomplete

    def _split(self, data: pd.DataFrame) -> List[Chunk]:
        def _create_chunk(index: int, data: pd.DataFrame, chunk_size: int) -> Chunk:
            chunk_data = data.iloc[index : index + chunk_size]
            chunk = Chunk(
                key=f'[{index}:{index + chunk_size - 1}]',
                data=chunk_data,
                start_index=index,
                end_index=index + chunk_size - 1,
            )
            if self.timestamp_column_name:
                chunk.start_datetime = pd.to_datetime(chunk.data[self.timestamp_column_name].min())
                chunk.end_datetime = pd.to_datetime(chunk.data[self.timestamp_column_name].max())
            return chunk

        chunks = [
            _create_chunk(index=i, data=data, chunk_size=self.chunk_size)
            for i in range(0, data.shape[0], self.chunk_size)
            if i + self.chunk_size - 1 < len(data)
        ]

        # deal with unassigned observations
        if data.shape[0] % self.chunk_size != 0:
            incomplete_chunk = _create_chunk(
                index=self.chunk_size * (data.shape[0] // self.chunk_size),
                data=data,
                chunk_size=(data.shape[0] % self.chunk_size),
            )
            if self.incomplete == 'keep':
                chunks += [incomplete_chunk]
            elif self.incomplete == 'append':
                chunks[-1] = chunks[-1].merge(incomplete_chunk)
            elif self.incomplete == 'drop':
                pass
            else:
                raise InvalidArgumentsException(
                    f"unknown value '{self.incomplete}' for 'incomplete'. "
                    f"Value should be one of ['drop', 'keep', 'append']"
                )

        return chunks

class CountBasedChunker(Chunker):
    """A Chunker that will split data into chunks based on the preferred number of total chunks.

    Notes
    -----
    - Chunks are adjacent, not overlapping
    - There may be "incomplete" chunks, as the remainder of observations after dividing by `chunk_size`
      will form a chunk of their own.

    Examples
    --------
    >>> from nannyml.chunk import CountBasedChunker
    >>> df = pd.read_parquet('/path/to/my/data.pq')
    >>> chunker = CountBasedChunker(chunk_number=100)
    >>> chunks = chunker.split(data=df)

    """

    def __init__(self, chunk_number: int, incomplete: str = 'keep', timestamp_column_name: Optional[str] = None):
        """Creates a new CountBasedChunker.

        It will calculate the amount of observations per chunk based on the given chunk count.
        It then continues to split the data into chunks just like a SizeBasedChunker does.

        Parameters
        ----------
        chunk_number: int
            The amount of chunks to split the data in.
        incomplete: str, default='keep'
            Choose how to handle any leftover observations that don't make up a full Chunk.
            The following options are available:

            - ``'drop'``: drop the leftover observations

            - ``'keep'``: keep the incomplete Chunk (containing less than ``chunk_size`` observations)

            - ``'append'``: append leftover observations to the last complete Chunk (overfilling it)

            Defaults to ``'keep'``.

        Returns
        -------
        chunker: CountBasedChunker

        """
        super().__init__(timestamp_column_name)
        self.incomplete = incomplete

        # TODO wording
        if not isinstance(chunk_number, int):
            raise InvalidArgumentsException(
                f"given chunk_number is of type {type(chunk_number)} but should be an int."
                f"Please provide an integer as a chunk count"
            )

        # TODO wording
        if chunk_number <= 0:
            raise InvalidArgumentsException(
                f"given chunk_number {chunk_number} is less then or equal to zero."
                f"The chunk number should always be larger then zero"
            )

        self.chunk_number = chunk_number

    def _split(self, data: pd.DataFrame) -> List[Chunk]:
        if data.shape[0] == 0:
            return []

        chunk_size = data.shape[0] // self.chunk_number
        chunks = SizeBasedChunker(
            chunk_size=chunk_size, incomplete=self.incomplete, timestamp_column_name=self.timestamp_column_name
        ).split(data=data)

        return chunks

class Chunk:
    """A subset of data that acts as a logical unit during calculations."""

    def __init__(
        self,
        key: str,
        data: pd.DataFrame,
        start_datetime: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        start_index: int = -1,
        end_index: int = -1,
        period: Optional[str] = None,
    ):
        """Creates a new chunk.

        Parameters
        ----------
        key : str, required.
            A value describing what data is wrapped in this chunk.
        data : DataFrame, required
            The data to be contained within the chunk.
        start_datetime: datetime
            The starting point in time for this chunk.
        end_datetime: datetime
            The end point in time for this chunk.
        period : string, optional
            The 'period' this chunk belongs to, for example 'reference' or 'analysis'.
        """
        self.key = key
        self.data = data
        self.period = period

        self.is_transition: bool = False

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.start_index: int = start_index
        self.end_index: int = end_index
        self.chunk_index: int = -1

    def __repr__(self):
        """Returns textual summary of a chunk.

        Returns
        -------
        chunk_str: str

        """
        return (
            f'Chunk[key={self.key}, data=pd.DataFrame[[{self.data.shape[0]}x{self.data.shape[1]}]], '
            f'period={self.period}, is_transition={self.is_transition},'
            f'start_datetime={self.start_datetime}, end_datetime={self.end_datetime},'
            f'start_index={self.start_index}, end_index={self.end_index}]'
        )

    def __len__(self):
        """Returns the number of rows held within this chunk.

        Returns
        -------
        length: int
            Number of rows in the `data` property of the chunk.

        """
        return self.data.shape[0]

    def __lt__(self, other: Chunk):
        if self.start_datetime and self.end_datetime and other.start_datetime and other.end_datetime:
            return self.end_datetime < other.start_datetime
        else:
            return self.end_index < other.start_index

    def merge(self, other: Chunk):
        """Merge two chunks together into a single one"""
        if self < other:
            first, second = self, other
        else:
            first, second = other, self

        result = copy.deepcopy(first)
        result.data = pd.concat([first.data, second.data])
        result.end_datetime = second.end_datetime
        result.key = f'[{first.start_index}:{second.end_index}]'
        return result

class DefaultChunker(Chunker):
    """Splits data into about 10 chunks.

    Examples
    --------
    >>> from nannyml.chunk import DefaultChunker
    >>> df = pd.read_parquet('/path/to/my/data.pq')
    >>> chunker = DefaultChunker()
    >>> chunks = chunker.split(data=df)
    """

    DEFAULT_CHUNK_COUNT = 10

    def __init__(self, timestamp_column_name: Optional[str] = None):
        """Creates a new DefaultChunker."""
        super(DefaultChunker, self).__init__(timestamp_column_name)

    def _split(self, data: pd.DataFrame) -> List[Chunk]:
        if data.shape[0] == 0:
            return []

        chunk_size = data.shape[0] // self.DEFAULT_CHUNK_COUNT
        chunks = SizeBasedChunker(chunk_size=chunk_size, timestamp_column_name=self.timestamp_column_name).split(
            data=data
        )
        return chunks

class PeriodBasedChunker(Chunker):
    """A Chunker that will split data into Chunks based on a date column in the data.

    Examples
    --------
    Chunk using monthly periods and providing a column name

    >>> from nannyml.chunk import PeriodBasedChunker
    >>> df = pd.read_parquet('/path/to/my/data.pq')
    >>> chunker = PeriodBasedChunker(timestamp_column_name='observation_date', offset='M')
    >>> chunks = chunker.split(data=df)

    Or chunk using weekly periods

    >>> from nannyml.chunk import PeriodBasedChunker
    >>> df = pd.read_parquet('/path/to/my/data.pq')
    >>> chunker = PeriodBasedChunker(timestamp_column_name=df['observation_date'], offset='W', minimum_chunk_size=50)
    >>> chunks = chunker.split(data=df)

    """

    def __init__(
        self,
        timestamp_column_name: str,
        offset: str = 'W',
    ):
        """Creates a new PeriodBasedChunker.

        Parameters
        ----------
        offset: a frequency string representing a pandas.tseries.offsets.DateOffset
            The offset determines how the time-based grouping will occur. A list of possible values
            is to be found at https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases.
        drop: bool, default=False
            Drops the timestamp column from the chunk data if True.

        Returns
        -------
        chunker: a PeriodBasedChunker instance used to split data into time-based Chunks.
        """
        super().__init__(timestamp_column_name)

        self.offset = offset

    def _split(self, data: pd.DataFrame) -> List[Chunk]:
        chunks = []
        try:
            grouped_data = data.groupby(pd.to_datetime(data[self.timestamp_column_name]).dt.to_period(self.offset))

            k: Period
            for k in grouped_data.groups.keys():
                chunk = Chunk(
                    key=str(k), data=grouped_data.get_group(k), start_datetime=k.start_time, end_datetime=k.end_time
                )
                chunks.append(chunk)
        except KeyError:
            raise ChunkerException(f"could not find date_column '{self.timestamp_column_name}' in given data")

        except ParserError:
            raise ChunkerException(
                f"could not parse date_column '{self.timestamp_column_name}' values as dates."
                f"Please verify if you've specified the correct date column."
            )

        return chunks

