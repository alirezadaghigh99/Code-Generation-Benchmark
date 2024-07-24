class MMLogger(Logger, ManagerMixin):
    """Formatted logger used to record messages.

    ``MMLogger`` can create formatted logger to log message with different
    log levels and get instance in the same way as ``ManagerMixin``.
    ``MMLogger`` has the following features:

    - Distributed log storage, ``MMLogger`` can choose whether to save log of
      different ranks according to `log_file`.
    - Message with different log levels will have different colors and format
      when displayed on terminal.

    Note:
        - The `name` of logger and the ``instance_name`` of ``MMLogger`` could
          be different. We can only get ``MMLogger`` instance by
          ``MMLogger.get_instance`` but not ``logging.getLogger``. This feature
          ensures ``MMLogger`` will not be incluenced by third-party logging
          config.
        - Different from ``logging.Logger``, ``MMLogger`` will not log warning
          or error message without ``Handler``.

    Examples:
        >>> logger = MMLogger.get_instance(name='MMLogger',
        >>>                                logger_name='Logger')
        >>> # Although logger has name attribute just like `logging.Logger`
        >>> # We cannot get logger instance by `logging.getLogger`.
        >>> assert logger.name == 'Logger'
        >>> assert logger.instance_name = 'MMLogger'
        >>> assert id(logger) != id(logging.getLogger('Logger'))
        >>> # Get logger that do not store logs.
        >>> logger1 = MMLogger.get_instance('logger1')
        >>> # Get logger only save rank0 logs.
        >>> logger2 = MMLogger.get_instance('logger2', log_file='out.log')
        >>> # Get logger only save multiple ranks logs.
        >>> logger3 = MMLogger.get_instance('logger3', log_file='out.log',
        >>>                                 distributed=True)

    Args:
        name (str): Global instance name.
        logger_name (str): ``name`` attribute of ``Logging.Logger`` instance.
            If `logger_name` is not defined, defaults to 'mmengine'.
        log_file (str, optional): The log filename. If specified, a
            ``FileHandler`` will be added to the logger. Defaults to None.
        log_level (str): The log level of the handler. Defaults to
            'INFO'. If log level is 'DEBUG', distributed logs will be saved
            during distributed training.
        file_mode (str): The file mode used to open log file. Defaults to 'w'.
        distributed (bool): Whether to save distributed logs, Defaults to
            false.
        file_handler_cfg (dict, optional): Configuration of file handler.
            Defaults to None. If ``file_handler_cfg`` is not specified,
            ``logging.FileHandler`` will be used by default. If it is
            specified, the ``type`` key should be set. It can be
            ``RotatingFileHandler``, ``TimedRotatingFileHandler``,
            ``WatchedFileHandler`` or other file handlers, and the remaining
            fields will be used to build the handler.

            Examples:
                >>> file_handler_cfg = dict(
                >>>    type='TimedRotatingFileHandler',
                >>>    when='MIDNIGHT',
                >>>    interval=1,
                >>>    backupCount=365)

            `New in version 0.9.0.`
    """

    def __init__(self,
                 name: str,
                 logger_name='mmengine',
                 log_file: Optional[str] = None,
                 log_level: Union[int, str] = 'INFO',
                 file_mode: str = 'w',
                 distributed=False,
                 file_handler_cfg: Optional[dict] = None):
        Logger.__init__(self, logger_name)
        ManagerMixin.__init__(self, name)
        # Get rank in DDP mode.
        if isinstance(log_level, str):
            log_level = logging._nameToLevel[log_level]
        global_rank = _get_rank()
        device_id = _get_device_id()

        # Config stream_handler. If `rank != 0`. stream_handler can only
        # export ERROR logs.
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        # `StreamHandler` record month, day, hour, minute, and second
        # timestamp.
        stream_handler.setFormatter(
            MMFormatter(color=True, datefmt='%m/%d %H:%M:%S'))
        # Only rank0 `StreamHandler` will log messages below error level.
        if global_rank == 0:
            stream_handler.setLevel(log_level)
        else:
            stream_handler.setLevel(logging.ERROR)
        stream_handler.addFilter(FilterDuplicateWarning(logger_name))
        self.handlers.append(stream_handler)

        if log_file is not None:
            world_size = _get_world_size()
            is_distributed = (log_level <= logging.DEBUG
                              or distributed) and world_size > 1
            if is_distributed:
                filename, suffix = osp.splitext(osp.basename(log_file))
                hostname = _get_host_info()
                if hostname:
                    filename = (f'{filename}_{hostname}_device{device_id}_'
                                f'rank{global_rank}{suffix}')
                else:
                    # Omit hostname if it is empty
                    filename = (f'{filename}_device{device_id}_'
                                f'rank{global_rank}{suffix}')
                log_file = osp.join(osp.dirname(log_file), filename)
            # Save multi-ranks logs if distributed is True. The logs of rank0
            # will always be saved.
            if global_rank == 0 or is_distributed:
                if file_handler_cfg is not None:
                    assert 'type' in file_handler_cfg
                    file_handler_type = file_handler_cfg.pop('type')
                    file_handlers_map = _get_logging_file_handlers()
                    if file_handler_type in file_handlers_map:
                        file_handler_cls = file_handlers_map[file_handler_type]
                        file_handler_cfg.setdefault('filename', log_file)
                        file_handler = file_handler_cls(**file_handler_cfg)
                    else:
                        raise ValueError('`logging.handlers` does not '
                                         f'contain {file_handler_type}')
                else:
                    # Here, the default behavior of the official
                    # logger is 'a'. Thus, we provide an interface to
                    # change the file mode to the default behavior.
                    # `FileHandler` is not supported to have colors,
                    # otherwise it will appear garbled.
                    file_handler = logging.FileHandler(log_file, file_mode)

                # `StreamHandler` record year, month, day hour, minute,
                # and second timestamp. file_handler will only record logs
                # without color to avoid garbled code saved in files.
                file_handler.setFormatter(
                    MMFormatter(color=False, datefmt='%Y/%m/%d %H:%M:%S'))
                file_handler.setLevel(log_level)
                file_handler.addFilter(FilterDuplicateWarning(logger_name))
                self.handlers.append(file_handler)
        self._log_file = log_file

    @property
    def log_file(self):
        return self._log_file

    @classmethod
    def get_current_instance(cls) -> 'MMLogger':
        """Get latest created ``MMLogger`` instance.

        :obj:`MMLogger` can call :meth:`get_current_instance` before any
        instance has been created, and return a logger with the instance name
        "mmengine".

        Returns:
            MMLogger: Configured logger instance.
        """
        if not cls._instance_dict:
            cls.get_instance('mmengine')
        return super().get_current_instance()

    def callHandlers(self, record: LogRecord) -> None:
        """Pass a record to all relevant handlers.

        Override ``callHandlers`` method in ``logging.Logger`` to avoid
        multiple warning messages in DDP mode. Loop through all handlers of
        the logger instance and its parents in the logger hierarchy. If no
        handler was found, the record will not be output.

        Args:
            record (LogRecord): A ``LogRecord`` instance contains logged
                message.
        """
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def setLevel(self, level):
        """Set the logging level of this logger.

        If ``logging.Logger.selLevel`` is called, all ``logging.Logger``
        instances managed by ``logging.Manager`` will clear the cache. Since
        ``MMLogger`` is not managed by ``logging.Manager`` anymore,
        ``MMLogger`` should override this method to clear caches of all
        ``MMLogger`` instance which is managed by :obj:`ManagerMixin`.

        level must be an int or a str.
        """
        self.level = logging._checkLevel(level)
        _accquire_lock()
        # The same logic as `logging.Manager._clear_cache`.
        for logger in MMLogger._instance_dict.values():
            logger._cache.clear()
        _release_lock()

