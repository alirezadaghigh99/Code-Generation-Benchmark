class AimVisBackend(BaseVisBackend):
    """Aim visualization backend class.

    Examples:
        >>> from mmengine.visualization import AimVisBackend
        >>> import numpy as np
        >>> aim_vis_backend = AimVisBackend()
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> aim_vis_backend.add_image('img', img)
        >>> aim_vis_backend.add_scalar('mAP', 0.6)
        >>> aim_vis_backend.add_scalars({'loss': 0.1, 'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> aim_vis_backend.add_config(cfg)

    Note:
        1. `New in version 0.9.0.`
        2. Refer to
           `Github issue <https://github.com/aimhubio/aim/issues/2064>`_ ,
           Aim is not unable to be install on Windows for now.

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer.
        init_kwargs (dict, optional): Aim initialization parameters. See
            `Aim <https://aimstack.readthedocs.io/en/latest/refs/sdk.html>`_
            for details. Defaults to None.
    """

    def __init__(self,
                 save_dir: Optional[str] = None,
                 init_kwargs: Optional[dict] = None):
        super().__init__(save_dir)  # type:ignore
        self._init_kwargs = init_kwargs

    def _init_env(self):
        """Setup env for Aim."""
        try:
            from aim import Run
        except ImportError:
            raise ImportError('Please run "pip install aim" to install aim')

        from datetime import datetime

        if self._save_dir is not None:
            path_list = os.path.normpath(self._save_dir).split(os.sep)
            exp_name = f'{path_list[-2]}_{path_list[-1]}'
        else:
            exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        if self._init_kwargs is None:
            self._init_kwargs = {}
        self._init_kwargs.setdefault('experiment', exp_name)
        self._aim_run = Run(**self._init_kwargs)

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return Aim object."""
        return self._aim_run

    @force_init_env
    def add_config(self, config, **kwargs) -> None:
        """Record the config to Aim.

        Args:
            config (Config): The Config object
        """
        if isinstance(config, Config):
            config = config.to_dict()
        self._aim_run['hparams'] = config

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        from aim import Image
        self._aim_run.track(name=name, value=Image(image), step=step)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to Aim.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        self._aim_run.track(name=name, value=value, step=step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to wandb.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        for key, value in scalar_dict.items():
            self._aim_run.track(name=key, value=value, step=step)

    def close(self) -> None:
        """Close the Aim."""
        if not hasattr(self, '_aim_run'):
            return

        self._aim_run.close()

class MLflowVisBackend(BaseVisBackend):
    """MLflow visualization backend class.

    It can write images, config, scalars, etc. to a
    mlflow file.

    Examples:
        >>> from mmengine.visualization import MLflowVisBackend
        >>> from mmengine import Config
        >>> import numpy as np
        >>> vis_backend = MLflowVisBackend(save_dir='temp_dir')
        >>> img = np.random.randint(0, 256, size=(10, 10, 3))
        >>> vis_backend.add_image('img.png', img)
        >>> vis_backend.add_scalar('mAP', 0.6)
        >>> vis_backend.add_scalars({'loss': 0.1,'acc':0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> vis_backend.add_config(cfg)

    Args:
        save_dir (str): The root directory to save the files
            produced by the backend.
        exp_name (str, optional): The experiment name. Defaults to None.
        run_name (str, optional): The run name. Defaults to None.
        tags (dict, optional): The tags to be added to the experiment.
            Defaults to None.
        params (dict, optional): The params to be added to the experiment.
            Defaults to None.
        tracking_uri (str, optional): The tracking uri. Defaults to None.
        artifact_suffix (Tuple[str] or str, optional): The artifact suffix.
            Defaults to ('.json', '.log', '.py', 'yaml').
        tracked_config_keys (dict, optional): The top level keys of config that
            will be added to the experiment. If it is None, which means all
            the config will be added. Defaults to None.
            `New in version 0.7.4.`
        artifact_location (str, optional): The location to store run artifacts.
            If None, the server picks an appropriate default.
            Defaults to None.
            `New in version 0.10.4.`
    """

    def __init__(self,
                 save_dir: str,
                 exp_name: Optional[str] = None,
                 run_name: Optional[str] = None,
                 tags: Optional[dict] = None,
                 params: Optional[dict] = None,
                 tracking_uri: Optional[str] = None,
                 artifact_suffix: SUFFIX_TYPE = ('.json', '.log', '.py',
                                                 'yaml'),
                 tracked_config_keys: Optional[dict] = None,
                 artifact_location: Optional[str] = None):
        super().__init__(save_dir)
        self._exp_name = exp_name
        self._run_name = run_name
        self._tags = tags
        self._params = params
        self._tracking_uri = tracking_uri
        self._artifact_suffix = artifact_suffix
        self._tracked_config_keys = tracked_config_keys
        self._artifact_location = artifact_location

    def _init_env(self):
        """Setup env for MLflow."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore

        try:
            import mlflow
        except ImportError:
            raise ImportError(
                'Please run "pip install mlflow" to install mlflow'
            )  # type: ignore
        self._mlflow = mlflow

        # when mlflow is imported, a default logger is created.
        # at this time, the default logger's stream is None
        # so the stream is reopened only when the stream is None
        # or the stream is closed
        logger = MMLogger.get_current_instance()
        for handler in logger.handlers:
            if handler.stream is None or handler.stream.closed:
                handler.stream = open(handler.baseFilename, 'a')

        if self._tracking_uri is not None:
            logger.warning(
                'Please make sure that the mlflow server is running.')
            self._mlflow.set_tracking_uri(self._tracking_uri)
        else:
            if os.name == 'nt':
                file_url = f'file:\\{os.path.abspath(self._save_dir)}'
            else:
                file_url = f'file://{os.path.abspath(self._save_dir)}'
            self._mlflow.set_tracking_uri(file_url)

        self._exp_name = self._exp_name or 'Default'

        if self._mlflow.get_experiment_by_name(self._exp_name) is None:
            self._mlflow.create_experiment(
                self._exp_name, artifact_location=self._artifact_location)

        self._mlflow.set_experiment(self._exp_name)

        if self._run_name is not None:
            self._mlflow.set_tag('mlflow.runName', self._run_name)
        if self._tags is not None:
            self._mlflow.set_tags(self._tags)
        if self._params is not None:
            self._mlflow.log_params(self._params)

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return MLflow object."""
        return self._mlflow

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to mlflow.

        Args:
            config (Config): The Config object
        """
        self.cfg = config
        if self._tracked_config_keys is None:
            self._mlflow.log_params(self._flatten(self.cfg.to_dict()))
        else:
            tracked_cfg = dict()
            for k in self._tracked_config_keys:
                tracked_cfg[k] = self.cfg[k]
            self._mlflow.log_params(self._flatten(tracked_cfg))
        self._mlflow.log_text(self.cfg.pretty_text, 'config.py')

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to mlflow.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Global step value to record. Default to 0.
        """
        self._mlflow.log_image(image, name)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to mlflow.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        self._mlflow.log_metric(name, value, step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to mlflow.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        assert isinstance(scalar_dict, dict)
        assert 'step' not in scalar_dict, 'Please set it directly ' \
                                          'through the step parameter'
        self._mlflow.log_metrics(scalar_dict, step)

    def close(self) -> None:
        """Close the mlflow."""
        if not hasattr(self, '_mlflow'):
            return

        file_paths = dict()
        for filename in scandir(self.cfg.work_dir, self._artifact_suffix,
                                True):
            file_path = osp.join(self.cfg.work_dir, filename)
            relative_path = os.path.relpath(file_path, self.cfg.work_dir)
            dir_path = os.path.dirname(relative_path)
            file_paths[file_path] = dir_path

        for file_path, dir_path in file_paths.items():
            self._mlflow.log_artifact(file_path, dir_path)

        self._mlflow.end_run()

    def _flatten(self, d, parent_key='', sep='.') -> dict:
        """Flatten the dict."""
        items = dict()
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.update(self._flatten(v, new_key, sep=sep))
            elif isinstance(v, list):
                if any(isinstance(x, dict) for x in v):
                    for i, x in enumerate(v):
                        items.update(
                            self._flatten(x, new_key + sep + str(i), sep=sep))
                else:
                    items[new_key] = v
            else:
                items[new_key] = v
        return items

