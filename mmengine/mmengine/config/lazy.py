def build(self) -> Any:
        """Return imported object.

        Returns:
            Any: Imported object
        """
        if isinstance(self._module, str):
            try:
                module = importlib.import_module(self._module)
            except Exception as e:
                raise type(e)(f'Failed to import {self._module} '
                              f'in {self.location} for {e}')

            if self._imported is not None:
                if hasattr(module, self._imported):
                    module = getattr(module, self._imported)
                else:
                    raise ImportError(
                        f'Failed to import {self._imported} '
                        f'from {self._module} in {self.location}')

            return module
        else:
            # import xxx.xxx
            # import xxx.yyy
            # import xxx.zzz
            # return imported xxx
            try:
                for module in self._module:
                    importlib.import_module(module)  # type: ignore
                module_name = self._module[0].split('.')[0]
                return importlib.import_module(module_name)
            except Exception as e:
                raise type(e)(f'Failed to import {self.module} '
                              f'in {self.location} for {e}')

class LazyObject:
    """LazyObject is used to lazily initialize the imported module during
    parsing the configuration file.

    During parsing process, the syntax like:

    Examples:
        >>> import torch.nn as nn
        >>> from mmdet.models import RetinaNet
        >>> import mmcls.models
        >>> import mmcls.datasets
        >>> import mmcls

    Will be parsed as:

    Examples:
        >>> # import torch.nn as nn
        >>> nn = lazyObject('torch.nn')
        >>> # from mmdet.models import RetinaNet
        >>> RetinaNet = lazyObject('mmdet.models', 'RetinaNet')
        >>> # import mmcls.models; import mmcls.datasets; import mmcls
        >>> mmcls = lazyObject(['mmcls', 'mmcls.datasets', 'mmcls.models'])

    ``LazyObject`` records all module information and will be further
    referenced by the configuration file.

    Args:
        module (str or list or tuple): The module name to be imported.
        imported (str, optional): The imported module name. Defaults to None.
        location (str, optional): The filename and line number of the imported
            module statement happened.
    """

    def __init__(self,
                 module: Union[str, list, tuple],
                 imported: Optional[str] = None,
                 location: Optional[str] = None):
        if not isinstance(module, str) and not is_seq_of(module, str):
            raise TypeError('module should be `str`, `list`, or `tuple`'
                            f'but got {type(module)}, this might be '
                            'a bug of MMEngine, please report it to '
                            'https://github.com/open-mmlab/mmengine/issues')
        self._module: Union[str, list, tuple] = module

        if not isinstance(imported, str) and imported is not None:
            raise TypeError('imported should be `str` or None, but got '
                            f'{type(imported)}, this might be '
                            'a bug of MMEngine, please report it to '
                            'https://github.com/open-mmlab/mmengine/issues')
        self._imported = imported
        self.location = location

    def build(self) -> Any:
        """Return imported object.

        Returns:
            Any: Imported object
        """
        if isinstance(self._module, str):
            try:
                module = importlib.import_module(self._module)
            except Exception as e:
                raise type(e)(f'Failed to import {self._module} '
                              f'in {self.location} for {e}')

            if self._imported is not None:
                if hasattr(module, self._imported):
                    module = getattr(module, self._imported)
                else:
                    raise ImportError(
                        f'Failed to import {self._imported} '
                        f'from {self._module} in {self.location}')

            return module
        else:
            # import xxx.xxx
            # import xxx.yyy
            # import xxx.zzz
            # return imported xxx
            try:
                for module in self._module:
                    importlib.import_module(module)  # type: ignore
                module_name = self._module[0].split('.')[0]
                return importlib.import_module(module_name)
            except Exception as e:
                raise type(e)(f'Failed to import {self.module} '
                              f'in {self.location} for {e}')

    @property
    def module(self):
        if isinstance(self._module, str):
            return self._module
        return self._module[0].split('.')[0]

    def __call__(self, *args, **kwargs):
        raise RuntimeError()

    def __deepcopy__(self, memo):
        return LazyObject(self._module, self._imported, self.location)

    def __getattr__(self, name):
        # Cannot locate the line number of the getting attribute.
        # Therefore only record the filename.
        if self.location is not None:
            location = self.location.split(', line')[0]
        else:
            location = self.location
        return LazyAttr(name, self, location)

    def __str__(self) -> str:
        if self._imported is not None:
            return self._imported
        return self.module

    __repr__ = __str__

    # `pickle.dump` will try to get the `__getstate__` and `__setstate__`
    # methods of the dumped object. If these two methods are not defined,
    # LazyObject will return a `__getstate__` LazyObject` or `__setstate__`
    # LazyObject.
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

