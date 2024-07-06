def establish_db_conn(config):
    """Establishes the database connection.

    If ``fiftyone.config.database_uri`` is defined, then we connect to that
    URI. Otherwise, a :class:`fiftyone.core.service.DatabaseService` is
    created.

    Args:
        config: a :class:`fiftyone.core.config.FiftyOneConfig`

    Raises:
        ConnectionError: if a connection to ``mongod`` could not be established
        FiftyOneConfigError: if ``fiftyone.config.database_uri`` is not
            defined and ``mongod`` could not be found
        ServiceExecutableNotFound: if
            :class:`fiftyone.core.service.DatabaseService` startup was
            attempted, but ``mongod`` was not found in :mod:`fiftyone.db.bin`
        RuntimeError: if the ``mongod`` found does not meet FiftyOne's
            requirements, or validation could not occur
    """
    global _client
    global _db_service
    global _connection_kwargs

    established_port = os.environ.get("FIFTYONE_PRIVATE_DATABASE_PORT", None)
    if established_port is not None:
        _connection_kwargs["port"] = int(established_port)
    if config.database_uri is not None:
        _connection_kwargs["host"] = config.database_uri
    elif _db_service is None:
        if os.environ.get("FIFTYONE_DISABLE_SERVICES", False):
            return

        try:
            _db_service = fos.DatabaseService()
            port = _db_service.port
            _connection_kwargs["port"] = port
            os.environ["FIFTYONE_PRIVATE_DATABASE_PORT"] = str(port)

        except fos.ServiceExecutableNotFound:
            raise FiftyOneConfigError(
                "MongoDB could not be installed on your system. Please "
                "define a `database_uri` in your "
                "`fiftyone.core.config.FiftyOneConfig` to connect to your"
                "own MongoDB instance or cluster "
            )

    _client = pymongo.MongoClient(
        **_connection_kwargs, appname=foc.DATABASE_APPNAME
    )
    _validate_db_version(config, _client)

    # Register cleanup method
    atexit.register(_delete_non_persistent_datasets_if_allowed)

    connect(config.database_name, **_connection_kwargs)

    db_config = get_db_config()
    if foc.CLIENT_TYPE != db_config.type:
        raise ConnectionError(
            "Cannot connect to database type '%s' with client type '%s'"
            % (db_config.type, foc.CLIENT_TYPE)
        )

    if os.environ.get("FIFTYONE_DISABLE_SERVICES", "0") != "1":
        fom.migrate_database_if_necessary(config=db_config)

def get_db_conn():
    """Returns a connection to the database.

    Returns:
        a ``pymongo.database.Database``
    """
    _connect()
    db = _client[fo.config.database_name]
    return _apply_options(db)

