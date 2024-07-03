    def register(cls, key: str) -> Callable:
        def inner_wrapper(wrapped_class: Type[Calibrator]) -> Type[Calibrator]:
            if key in cls._registry:
                warnings.warn(f"re-registering calibrator with key '{key}'")

            cls._registry[key] = wrapped_class
            return wrapped_class

        return inner_wrapper    def create(cls, key: str = 'isotonic', **kwargs):
        """Creates a new Calibrator given a key value and optional keyword args.

        If the provided key equals ``None``, then a new instance of the default Calibrator (IsotonicCalibrator)
        will be returned.

        If a non-existent key is provided an ``InvalidArgumentsException`` is raised.

        Parameters
        ----------
        key : str, default='isotonic'
            The key used to retrieve a Calibrator. When providing a key that is already in the index, the value
            will be overwritten.
        kwargs : dict
            Optional keyword arguments that will be passed along to the function associated with the key.
            It can then use these arguments during the creation of a new Calibrator instance.

        Returns
        -------
        calibrator: Calibrator
            A new instance of a specific Calibrator subclass.

        Examples
        --------
        >>> calibrator = CalibratorFactory.create('isotonic', kwargs={'foo': 'bar'})
        """
        if key not in cls._registry:
            raise InvalidArgumentsException(
                f"calibrator '{key}' unknown. " f"Please provide one of the following: {cls._registry.keys()}"
            )

        calibrator_class = cls._registry.get(key)
        assert calibrator_class

        return calibrator_class(**kwargs)