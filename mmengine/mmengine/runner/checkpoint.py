def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

def save_checkpoint(checkpoint,
                    filename,
                    file_client_args=None,
                    backend_args=None):
    """Save checkpoint to file.

    Args:
        checkpoint (dict): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            `backend_args` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.
    """
    if file_client_args is not None:
        print_log(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead',
            logger='current',
            level=logging.WARNING)
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set '
                'at the same time.')

    if filename.startswith('pavi://'):
        if file_client_args is not None or backend_args is not None:
            raise ValueError(
                '"file_client_args" or "backend_args" should be "None" if '
                'filename starts with "pavi://"')
        try:
            from pavi import exception, modelcloud
        except ImportError:
            raise ImportError(
                'Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except exception.NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        file_client = FileClient.infer_client(file_client_args, filename)
        if file_client_args is None:
            file_backend = get_file_backend(
                filename, backend_args=backend_args)
        else:
            file_backend = file_client

        with io.BytesIO() as f:
            torch.save(checkpoint, f)
            file_backend.put(f.getvalue(), filename)

