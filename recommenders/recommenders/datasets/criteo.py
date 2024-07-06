def load_pandas_df(size="sample", local_cache_path=None, header=DEFAULT_HEADER):
    """Loads the Criteo DAC dataset as `pandas.DataFrame`. This function download, untar, and load the dataset.

    The dataset consists of a portion of Criteo’s traffic over a period
    of 24 days. Each row corresponds to a display ad served by Criteo and the first
    column indicates whether this ad has been clicked or not.

    There are 13 features taking integer values (mostly count features) and 26
    categorical features. The values of the categorical features have been hashed
    onto 32 bits for anonymization purposes.

    The schema is:

    .. code-block:: python

        <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>

    More details (need to accept user terms to see the information):
    http://labs.criteo.com/2013/12/download-terabyte-click-logs/

    Args:
        size (str): Dataset size. It can be "sample" or "full".
        local_cache_path (str): Path where to cache the tar.gz file locally
        header (list): Dataset header names.

    Returns:
        pandas.DataFrame: Criteo DAC sample dataset.
    """
    with download_path(local_cache_path) as path:
        filepath = download_criteo(size, path)
        filepath = extract_criteo(size, filepath)
        df = pd.read_csv(filepath, sep="\t", header=None, names=header)
    return df

def load_spark_df(
    spark,
    size="sample",
    header=DEFAULT_HEADER,
    local_cache_path=None,
    dbfs_datapath="dbfs:/FileStore/dac",
    dbutils=None,
):
    """Loads the Criteo DAC dataset as `pySpark.DataFrame`.

    The dataset consists of a portion of Criteo’s traffic over a period
    of 24 days. Each row corresponds to a display ad served by Criteo and the first
    column is indicates whether this ad has been clicked or not.

    There are 13 features taking integer values (mostly count features) and 26
    categorical features. The values of the categorical features have been hashed
    onto 32 bits for anonymization purposes.

    The schema is:

    .. code-block:: python

        <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>

    More details (need to accept user terms to see the information):
    http://labs.criteo.com/2013/12/download-terabyte-click-logs/

    Args:
        spark (pySpark.SparkSession): Spark session.
        size (str): Dataset size. It can be "sample" or "full".
        local_cache_path (str): Path where to cache the tar.gz file locally.
        header (list): Dataset header names.
        dbfs_datapath (str): Where to store the extracted files on Databricks.
        dbutils (Databricks.dbutils): Databricks utility object.

    Returns:
        pyspark.sql.DataFrame: Criteo DAC training dataset.
    """
    with download_path(local_cache_path) as path:
        filepath = download_criteo(size, path)
        filepath = extract_criteo(size, filepath)

        if is_databricks():
            try:
                # Driver node's file path
                node_path = "file:" + filepath
                # needs to be on dbfs to load
                dbutils.fs.cp(node_path, dbfs_datapath, recurse=True)
                path = dbfs_datapath
            except Exception:
                raise ValueError(
                    "To use on a Databricks notebook, dbutils object should be passed as an argument"
                )
        else:
            path = filepath

        schema = get_spark_schema(header)
        df = spark.read.csv(path, schema=schema, sep="\t", header=False)
        df.cache().count()  # trigger execution to overcome spark's lazy evaluation
    return df

def download_criteo(size="sample", work_directory="."):
    """Download criteo dataset as a compressed file.

    Args:
        size (str): Size of criteo dataset. It can be "full" or "sample".
        work_directory (str): Working directory.

    Returns:
        str: Path of the downloaded file.

    """
    url = CRITEO_URL[size]
    return maybe_download(url, work_directory=work_directory)

