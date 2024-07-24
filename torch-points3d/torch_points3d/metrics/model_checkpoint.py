class ModelCheckpoint(object):
    """Create a checkpoint for a given model

    Argumemnts:
        - load_dir: directory where to load the checkpoint from (if exists)
        - check_name: Name of the checkpoint (without the .pt extension)
        - selection_stage: Stage that is used for selecting the best model
        - run_config: Config of the run. In resume mode, this gets discarded
        - resume: Resume a previous training - this creates optimizers
        - strict: If strict and checkpoint is empty then it raises a ValueError. Being in resume mode forces strict
    """

    def __init__(
        self,
        load_dir: str,
        check_name: str,
        selection_stage: str,
        run_config: DictConfig = DictConfig({}),
        resume=False,
        strict=False,
    ):
        # Conversion of run_config to save a dictionary and not a pickle of omegaconf
        rc = OmegaConf.to_container(copy.deepcopy(run_config))
        self._checkpoint = Checkpoint.load(load_dir, check_name, run_config=rc, strict=strict, resume=resume)
        self._resume = resume
        self._selection_stage = selection_stage

    def create_model(self, dataset, weight_name=Checkpoint._LATEST):
        if not self.is_empty:
            run_config = copy.deepcopy(self._checkpoint.run_config)
            model = instantiate_model(OmegaConf.create(run_config), dataset)
            if hasattr(self._checkpoint, "model_props"):
                for k, v in self._checkpoint.model_props.items():
                    setattr(model, k, v)
                delattr(self._checkpoint, "model_props")
            self._initialize_model(model, weight_name)
            return model
        else:
            raise ValueError("Checkpoint is empty")

    @property
    def start_epoch(self):
        if self._resume:
            return self.get_starting_epoch()
        else:
            return 1

    @property
    def run_config(self):
        return OmegaConf.create(self._checkpoint.run_config)

    @property
    def data_config(self):
        return OmegaConf.create(self._checkpoint.run_config).data

    @property
    def selection_stage(self):
        return self._selection_stage

    @selection_stage.setter
    def selection_stage(self, value):
        self._selection_stage = value

    @property
    def is_empty(self):
        return self._checkpoint.is_empty

    @property
    def checkpoint_path(self):
        return self._checkpoint.path

    @property
    def dataset_properties(self) -> Dict:
        return self._checkpoint.dataset_properties

    @dataset_properties.setter
    def dataset_properties(self, dataset_properties: Union[Dict[str, Any], Dict]):
        self._checkpoint.dataset_properties = dataset_properties

    def get_starting_epoch(self):
        return len(self._checkpoint.stats["train"]) + 1

    def _initialize_model(self, model: model_interface.CheckpointInterface, weight_name):
        if not self._checkpoint.is_empty:
            state_dict = self._checkpoint.get_state_dict(weight_name)
            model.load_state_dict(state_dict, strict=False)
            self._checkpoint.load_optim_sched(model, load_state=self._resume)

    def find_func_from_metric_name(self, metric_name, default_metrics_func):
        for token_name, func in default_metrics_func.items():
            if token_name in metric_name:
                return func
        raise Exception(
            'The metric name {} doesn t have a func to measure which one is best in {}. Example: For best_train_iou, {{"iou":max}}'.format(
                metric_name, default_metrics_func
            )
        )

    def save_best_models_under_current_metrics(
        self, model: model_interface.CheckpointInterface, metrics_holder: dict, metric_func_dict: dict, **kwargs
    ):
        """[This function is responsible to save checkpoint under the current metrics and their associated DEFAULT_METRICS_FUNC]
        Arguments:
            model {[CheckpointInterface]} -- [Model]
            metrics_holder {[Dict]} -- [Need to contain stage, epoch, current_metrics]
        """
        metrics = metrics_holder["current_metrics"]
        stage = metrics_holder["stage"]
        epoch = metrics_holder["epoch"]

        stats = self._checkpoint.stats
        state_dict = copy.deepcopy(model.state_dict())

        current_stat = {}
        current_stat["epoch"] = epoch

        models_to_save = self._checkpoint.models
        if stage not in stats:
            stats[stage] = []

        if stage == "train":
            models_to_save[Checkpoint._LATEST] = state_dict
        else:
            if len(stats[stage]) > 0:
                latest_stats = stats[stage][-1]

                msg = ""
                improved_metric = 0

                for metric_name, current_metric_value in metrics.items():
                    current_stat[metric_name] = current_metric_value

                    metric_func = self.find_func_from_metric_name(metric_name, metric_func_dict)
                    best_metric_from_stats = latest_stats.get("best_{}".format(metric_name), current_metric_value)
                    best_value = metric_func(best_metric_from_stats, current_metric_value)
                    current_stat["best_{}".format(metric_name)] = best_value

                    # This new value seems to be better under metric_func
                    if (self._selection_stage == stage) and (
                        current_metric_value == best_value
                    ):  # Update the model weights
                        models_to_save["best_{}".format(metric_name)] = state_dict

                        msg += "{}: {} -> {}, ".format(metric_name, best_metric_from_stats, best_value)
                        improved_metric += 1

                if improved_metric > 0:
                    colored_print(COLORS.VAL_COLOR, msg[:-2])
            else:
                # stats[stage] is empty.
                for metric_name, metric_value in metrics.items():
                    current_stat[metric_name] = metric_value
                    current_stat["best_{}".format(metric_name)] = metric_value
                    models_to_save["best_{}".format(metric_name)] = state_dict

        kwargs["model_props"] = {
            "num_epochs": model.num_epochs,  # type: ignore
            "num_batches": model.num_batches,  # type: ignore
            "num_samples": model.num_samples,  # type: ignore
        }

        self._checkpoint.stats[stage].append(current_stat)
        self._checkpoint.save_objects(models_to_save, stage, current_stat, model.optimizer, model.schedulers, **kwargs)

    def validate(self, dataset_config):
        """A checkpoint is considered as valid if it can recreate the model from
        a dataset config only"""
        if dataset_config is not None:
            for k, v in dataset_config.items():
                self.data_config[k] = v
        try:
            instantiate_model(OmegaConf.create(self.run_config), self.data_config)
        except:
            return False

        return True

