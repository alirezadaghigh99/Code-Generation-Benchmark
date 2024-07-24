class PickleFileWriter(FileWriter):
    """Writes ``Results`` to disk (local/remote/cloud) as a pickle file.

    A :class:`~nannyml.io.file_writer.FileWriter` implementation that pickles a `Result` object and writes the
    resulting bytestream to local or cloud storage.
    """

    def __init__(
        self,
        path: str,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
        write_args: Optional[Dict[str, Any]] = None,
    ):
        """

        Parameters
        ----------
        path : str
            The directory in which to output the generated pickle file. The name of the pickle file will equal
            the fully qualified result class name with a `pkl` extension, e.g. `nannyml.drift.univariate.result.pkl`
        credentials : Dict[str, Any] default=None
            Used to provide credential information following specific ``fsspec`` implementations.
        fs_args : default=None
            Specific arguments passed along to the ``fsspec`` filesystem initializer.

        Examples
        --------
        >>> writer = PickleFileWriter(
        ...    path='s3://my-output-bucket/output',
        ...    credentials={'aws_access_key_id': 'access_key_id', 'aws_secret_access_key': 'secret_access_key'}
        ... )
        >>> writer.write(result)
        """
        super().__init__(path, write_args, credentials, fs_args)

    @log_usage(UsageEvent.WRITE_PICKLE)
    def _write(self, result: Result, **write_args):
        if 'filename' not in write_args:
            raise InvalidArgumentsException("missing required parameter 'filename'")
        filename = write_args.pop('filename')

        write_path = Path(get_filepath_str(self.filepath, self._protocol)) / filename
        bytez = pickle.dumps(result)
        _write_bytes_to_filesystem(bytez, write_path, self._fs)

