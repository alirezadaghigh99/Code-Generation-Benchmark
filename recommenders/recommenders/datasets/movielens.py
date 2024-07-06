def get_df(
        cls,
        size: int = 3,
        seed: int = 100,
        keep_first_n_cols: Optional[int] = None,
        keep_title_col: bool = False,
        keep_genre_col: bool = False,
    ) -> pd.DataFrame:
        """Return fake movielens dataset as a Pandas Dataframe with specified rows.

        Args:
            size (int): number of rows to generate
            seed (int, optional): seeding the pseudo-number generation. Defaults to 100.
            keep_first_n_cols (int, optional): keep the first n default movielens columns.
            keep_title_col (bool): remove the title column if False. Defaults to True.
            keep_genre_col (bool): remove the genre column if False. Defaults to True.

        Returns:
            pandas.DataFrame: a mock dataset
        """
        schema = cls.to_schema()
        if keep_first_n_cols is not None:
            if keep_first_n_cols < 1 or keep_first_n_cols > len(DEFAULT_HEADER):
                raise ValueError(
                    f"Invalid value for 'keep_first_n_cols': {keep_first_n_cols}. Valid range: [1-{len(DEFAULT_HEADER)}]"
                )
            schema = schema.remove_columns(DEFAULT_HEADER[keep_first_n_cols:])
        if not keep_title_col:
            schema = schema.remove_columns([DEFAULT_TITLE_COL])
        if not keep_genre_col:
            schema = schema.remove_columns([DEFAULT_GENRE_COL])

        random.seed(seed)
        schema.checks = [pa.Check.unique_columns([DEFAULT_USER_COL, DEFAULT_ITEM_COL])]
        return schema.example(size=size)

def load_pandas_df(
    size="100k",
    header=None,
    local_cache_path=None,
    title_col=None,
    genres_col=None,
    year_col=None,
):
    """Loads the MovieLens dataset as pd.DataFrame.

    Download the dataset from https://files.grouplens.org/datasets/movielens, unzip, and load.
    To load movie information only, you can use load_item_df function.

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m", "mock100").
        header (list or tuple or None): Rating dataset header.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored and data is rendered using the 'DEFAULT_HEADER' instead.
        local_cache_path (str): Path (directory or a zip file) to cache the downloaded zip file.
            If None, all the intermediate files will be stored in a temporary directory and removed after use.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored.
        title_col (str): Movie title column name. If None, the column will not be loaded.
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, the column will not be loaded.
        year_col (str): Movie release year column name. If None, the column will not be loaded.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored.

    Returns:
        pandas.DataFrame: Movie rating dataset.


    **Examples**

    .. code-block:: python

        # To load just user-id, item-id, and ratings from MovieLens-1M dataset,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating'))

        # To load rating's timestamp together,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'))

        # To load movie's title, genres, and released year info along with the ratings data,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'),
            title_col='Title',
            genres_col='Genres',
            year_col='Year'
        )
    """
    size = size.lower()
    if size not in DATA_FORMAT and size not in MOCK_DATA_FORMAT:
        raise ValueError(f"Size: {size}. " + ERROR_MOVIE_LENS_SIZE)

    if header is None:
        header = DEFAULT_HEADER
    elif len(header) < 2:
        raise ValueError(ERROR_HEADER)
    elif len(header) > 4:
        warnings.warn(WARNING_MOVIE_LENS_HEADER)
        header = header[:4]

    if size in MOCK_DATA_FORMAT:
        # generate fake data
        return MockMovielensSchema.get_df(
            keep_first_n_cols=len(header),
            keep_title_col=(title_col is not None),
            keep_genre_col=(genres_col is not None),
            **MOCK_DATA_FORMAT[
                size
            ],  # supply the rest of the kwarg with the dictionary
        )

    movie_col = header[1]

    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "ml-{}.zip".format(size))
        datapath, item_datapath = _maybe_download_and_extract(size, filepath)

        # Load movie features such as title, genres, and release year
        item_df = _load_item_df(
            size, item_datapath, movie_col, title_col, genres_col, year_col
        )

        # Load rating data
        df = pd.read_csv(
            datapath,
            sep=DATA_FORMAT[size].separator,
            engine="python",
            names=header,
            usecols=[*range(len(header))],
            header=0 if DATA_FORMAT[size].has_header else None,
        )

        # Convert 'rating' type to float
        if len(header) > 2:
            df[header[2]] = df[header[2]].astype(float)

        # Merge rating df w/ item_df
        if item_df is not None:
            df = df.merge(item_df, on=header[1])

    return df

def get_spark_df(
        cls,
        spark,
        size: int = 3,
        seed: int = 100,
        keep_title_col: bool = False,
        keep_genre_col: bool = False,
        tmp_path: Optional[str] = None,
    ):
        """Return fake movielens dataset as a Spark Dataframe with specified rows

        Args:
            spark (SparkSession): spark session to load the dataframe into
            size (int): number of rows to generate
            seed (int): seeding the pseudo-number generation. Defaults to 100.
            keep_title_col (bool): remove the title column if False. Defaults to False.
            keep_genre_col (bool): remove the genre column if False. Defaults to False.
            tmp_path (str, optional): path to store files for serialization purpose
                when transferring data from python to java.
                If None, a temporal path is used instead

        Returns:
            pyspark.sql.DataFrame: a mock dataset
        """
        pandas_df = cls.get_df(
            size=size, seed=seed, keep_title_col=True, keep_genre_col=True
        )

        # generate temp folder
        with download_path(tmp_path) as tmp_folder:
            filepath = os.path.join(tmp_folder, f"mock_movielens_{size}.csv")
            # serialize the pandas.df as a csv to avoid the expensive java <-> python communication
            pandas_df.to_csv(filepath, header=False, index=False)
            spark_df = spark.read.csv(
                filepath, schema=cls._get_spark_deserialization_schema()
            )
            # Cache and force trigger action since data-file might be removed.
            spark_df.cache()
            spark_df.count()

        if not keep_title_col:
            spark_df = spark_df.drop(DEFAULT_TITLE_COL)
        if not keep_genre_col:
            spark_df = spark_df.drop(DEFAULT_GENRE_COL)
        return spark_df

