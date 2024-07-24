class Detectron2GoRunner(D2GoDataAPIMixIn, BaseRunner):
    def register(self, cfg):
        super().register(cfg)
        self.original_cfg = cfg.clone()
        inject_coco_datasets(cfg)
        register_dynamic_datasets(cfg)
        update_cfg_if_using_adhoc_dataset(cfg)

    @classmethod
    def get_default_cfg(cls):
        return get_detectron2go_runner_default_cfg(CfgNode())

    # temporary API
    def _build_model(self, cfg, eval_only=False):
        # build_model might modify the cfg, thus clone
        cfg = cfg.clone()

        model = build_d2go_model(cfg).model
        ema.may_build_model_ema(cfg, model)

        if cfg.QUANTIZATION.QAT.ENABLED:
            # Disable fake_quant and observer so that the model will be trained normally
            # before QAT being turned on (controlled by QUANTIZATION.QAT.START_ITER).
            if hasattr(model, "get_rand_input"):
                imsize = cfg.INPUT.MAX_SIZE_TRAIN
                rand_input = model.get_rand_input(imsize)
                example_inputs = (rand_input, {})
                model = setup_qat_model(
                    cfg,
                    model,
                    enable_fake_quant=eval_only,
                    enable_observer=True,
                )
                model(*example_inputs)
            else:
                imsize = cfg.INPUT.MAX_SIZE_TRAIN
                model = setup_qat_model(
                    cfg,
                    model,
                    enable_fake_quant=eval_only,
                    enable_observer=False,
                )

        if cfg.MODEL.FROZEN_LAYER_REG_EXP:
            set_requires_grad(model, cfg.MODEL.FROZEN_LAYER_REG_EXP, False)
            model = freeze_matched_bn(model, cfg.MODEL.FROZEN_LAYER_REG_EXP)

        if eval_only:
            checkpointer = self.build_checkpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            model.eval()

            if cfg.MODEL_EMA.ENABLED and cfg.MODEL_EMA.USE_EMA_WEIGHTS_FOR_EVAL_ONLY:
                ema.apply_model_ema(model)

        return model

    def build_model(self, cfg, eval_only=False):
        # Attach memory profiler to GPU OOM events
        if cfg.get("MEMORY_PROFILER", CfgNode()).get("ENABLED", False):
            attach_oom_logger(
                cfg.OUTPUT_DIR, trace_max_entries=cfg.MEMORY_PROFILER.TRACE_MAX_ENTRIES
            )

        model = self._build_model(cfg, eval_only)
        model = prepare_fb_model(cfg, model)

        # Note: the _visualize_model API is experimental
        if comm.is_main_process():
            if hasattr(model, "_visualize_model"):
                logger.info("Adding model visualization ...")
                tbx_writer = self.get_tbx_writer(cfg)
                model._visualize_model(tbx_writer)

        return model

    def build_checkpointer(self, cfg, model, save_dir, **kwargs):
        kwargs.update(ema.may_get_ema_checkpointer(cfg, model))
        checkpointer = FSDPCheckpointer(model, save_dir=save_dir, **kwargs)
        return checkpointer

    def build_optimizer(self, cfg, model):
        return build_optimizer_mapper(cfg, model)

    def build_lr_scheduler(self, cfg, optimizer):
        return d2_build_lr_scheduler(cfg, optimizer)

    def _create_evaluators(
        self,
        cfg,
        dataset_name,
        output_folder,
        train_iter,
        model_tag,
        model=None,
    ):
        evaluator = self.get_evaluator(cfg, dataset_name, output_folder=output_folder)

        if not isinstance(evaluator, DatasetEvaluators):
            evaluator = DatasetEvaluators([evaluator])
        if comm.is_main_process():
            # Add evaluator for visualization only to rank 0
            tbx_writer = self.get_tbx_writer(cfg)
            logger.info("Adding visualization evaluator ...")
            mapper = self.get_mapper(cfg, is_train=False)
            vis_eval_type = self.get_visualization_evaluator()
            if vis_eval_type is not None:
                evaluator._evaluators.append(
                    vis_eval_type(
                        cfg,
                        tbx_writer,
                        mapper,
                        dataset_name,
                        train_iter=train_iter,
                        tag_postfix=model_tag,
                    )
                )
        return evaluator

    # experimental API
    @classmethod
    def _get_inference_callbacks(cls):
        return {
            "on_start": lambda: None,
            "on_end": lambda: None,
            "before_inference": lambda: None,
            "after_inference": lambda: None,
        }

    def _do_test(self, cfg, model, train_iter=None, model_tag="default"):
        """train_iter: Current iteration of the model, None means final iteration"""
        assert len(cfg.DATASETS.TEST)
        assert cfg.OUTPUT_DIR

        is_final = (train_iter is None) or (train_iter == cfg.SOLVER.MAX_ITER - 1)

        logger.info(
            f"Running evaluation for model tag {model_tag} at iter {train_iter}..."
        )

        def _get_inference_dir_name(base_dir, inference_type, dataset_name):
            return os.path.join(
                base_dir,
                inference_type,
                model_tag,
                str(train_iter) if train_iter is not None else "final",
                dataset_name,
            )

        attach_profilers(cfg, model)

        results = OrderedDict()
        results[model_tag] = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            # Evaluator will create output folder, no need to create here
            output_folder = _get_inference_dir_name(
                cfg.OUTPUT_DIR, "inference", dataset_name
            )

            # NOTE: creating evaluator after dataset is loaded as there might be dependency.  # noqa
            data_loader = self.build_detection_test_loader(cfg, dataset_name)

            evaluator = self._create_evaluators(
                cfg,
                dataset_name,
                output_folder,
                train_iter,
                model_tag,
                (
                    model.module
                    if isinstance(model, nn.parallel.DistributedDataParallel)
                    else model
                ),
            )

            inference_callbacks = self._get_inference_callbacks()
            results_per_dataset = inference_on_dataset(
                model, data_loader, evaluator, callbacks=inference_callbacks
            )

            if comm.is_main_process():
                results[model_tag][dataset_name] = results_per_dataset
                if is_final:
                    print_csv_format(results_per_dataset)

            if is_final and cfg.TEST.AUG.ENABLED:
                # In the end of training, run an evaluation with TTA
                # Only support some R-CNN models.
                output_folder = _get_inference_dir_name(
                    cfg.OUTPUT_DIR, "inference_TTA", dataset_name
                )

                logger.info("Running inference with test-time augmentation ...")
                data_loader = self.build_detection_test_loader(
                    cfg, dataset_name, mapper=lambda x: x
                )
                evaluator = self.get_evaluator(
                    cfg, dataset_name, output_folder=output_folder
                )
                inference_on_dataset(
                    GeneralizedRCNNWithTTA(cfg, model), data_loader, evaluator
                )

        if is_final and cfg.TEST.EXPECTED_RESULTS and comm.is_main_process():
            assert len(results) == 1, "Results verification only supports one dataset!"
            verify_results(cfg, results[model_tag][cfg.DATASETS.TEST[0]])

        # write results to tensorboard
        if comm.is_main_process() and results:
            from detectron2.evaluation.testing import flatten_results_dict

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                tbx_writer = self.get_tbx_writer(cfg)
                tbx_writer._writer.add_scalar("eval_{}".format(k), v, train_iter)

        if comm.is_main_process():
            tbx_writer = self.get_tbx_writer(cfg)
            tbx_writer._writer.flush()
        return results

    def do_test(self, cfg, model, train_iter=None):
        """do_test does not load the weights of the model.
        If you want to use it outside the regular training routine,
        you will have to load the weights through a checkpointer.
        """
        results = OrderedDict()
        with maybe_subsample_n_images(cfg) as new_cfg:
            # default model
            cur_results = self._do_test(
                new_cfg, model, train_iter=train_iter, model_tag="default"
            )
            results.update(cur_results)

            # model with ema weights
            if cfg.MODEL_EMA.ENABLED and not isinstance(model, PredictorWrapper):
                logger.info("Run evaluation with EMA.")
                with ema.apply_model_ema_and_restore(model):
                    cur_results = self._do_test(
                        new_cfg, model, train_iter=train_iter, model_tag="ema"
                    )
                    results.update(cur_results)

        return results

    def _get_trainer_hooks(
        self, cfg, model, optimizer, scheduler, periodic_checkpointer, trainer
    ):
        return [
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model) if cfg.MODEL_EMA.ENABLED else None,
            self._create_data_loader_hook(cfg),
            self._create_after_step_hook(
                cfg, model, optimizer, scheduler, periodic_checkpointer
            ),
            create_preemption_hook(cfg, periodic_checkpointer, self._control_pg),
            hooks.EvalHook(
                cfg.TEST.EVAL_PERIOD,
                lambda: self.do_test(cfg, model, train_iter=trainer.iter),
                eval_after_train=False,  # done by a separate do_test call in tools/train_net.py
            ),
            compute_kmeans_anchors_hook(self, cfg),
            self._create_qat_hook(cfg) if cfg.QUANTIZATION.QAT.ENABLED else None,
        ]

    def do_train(self, cfg, model, resume):
        with get_monitoring_service():
            # Note that flops at the beginning of training is often inaccurate,
            # if a model has input-dependent logic
            attach_profilers(cfg, model)

            if cfg.NUMA_BINDING is True:
                import numa

                num_gpus_per_node = comm.get_local_size()
                num_sockets = numa.get_max_node() + 1
                socket_id = torch.cuda.current_device() // (
                    max(num_gpus_per_node // num_sockets, 1)
                )
                node_mask = set([socket_id])
                numa.bind(node_mask)

            optimizer = self.build_optimizer(cfg, model)
            scheduler = self.build_lr_scheduler(cfg, optimizer)

            checkpointer = self.build_checkpointer(
                cfg,
                model,
                save_dir=cfg.OUTPUT_DIR,
                load_ckpt_to_gpu=cfg.LOAD_CKPT_TO_GPU,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
            start_iter = (
                checkpoint.get("iteration", -1)
                if resume and checkpointer.has_checkpoint()
                else -1
            )
            del checkpoint
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
            start_iter += 1

            if "EARLY_STOPPING_FRACTION" in cfg.SOLVER:
                assert (
                    cfg.SOLVER.EARLY_STOPPING_FRACTION >= 0
                ), f"Early stopping fraction must be non-negative, but is {cfg.SOLVER.EARLY_STOPPING_FRACTION}"
                assert (
                    cfg.SOLVER.EARLY_STOPPING_FRACTION <= 1
                ), f"Early stopping fraction must not be larger than 1, but is {cfg.SOLVER.EARLY_STOPPING_FRACTION}"
                max_iter = int(cfg.SOLVER.MAX_ITER * cfg.SOLVER.EARLY_STOPPING_FRACTION)
            else:
                max_iter = cfg.SOLVER.MAX_ITER

            periodic_checkpointer = PeriodicCheckpointer(
                checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
            )

            data_loader = self.build_detection_train_loader(cfg)

            def _get_model_with_abnormal_checker(model):
                if not cfg.ABNORMAL_CHECKER.ENABLED:
                    return model

                tbx_writer = self.get_tbx_writer(cfg)
                writers = get_writers(cfg, tbx_writer)
                checker = AbnormalLossChecker(start_iter, writers)
                ret = AbnormalLossCheckerWrapper(model, checker)
                return ret

            if cfg.SOLVER.AMP.ENABLED:
                trainer = AMPTrainer(
                    _get_model_with_abnormal_checker(model),
                    data_loader,
                    optimizer,
                    gather_metric_period=cfg.GATHER_METRIC_PERIOD,
                    zero_grad_before_forward=cfg.ZERO_GRAD_BEFORE_FORWARD,
                    grad_scaler=get_grad_scaler(cfg),
                    precision=parse_precision_from_string(
                        cfg.SOLVER.AMP.PRECISION, lightning=False
                    ),
                    log_grad_scaler=cfg.SOLVER.AMP.LOG_GRAD_SCALER,
                    async_write_metrics=cfg.ASYNC_WRITE_METRICS,
                )
            else:
                trainer = SimpleTrainer(
                    _get_model_with_abnormal_checker(model),
                    data_loader,
                    optimizer,
                    gather_metric_period=cfg.GATHER_METRIC_PERIOD,
                    zero_grad_before_forward=cfg.ZERO_GRAD_BEFORE_FORWARD,
                    async_write_metrics=cfg.ASYNC_WRITE_METRICS,
                )

            if cfg.SOLVER.AMP.ENABLED and torch.cuda.is_available():
                # Allow to use the TensorFloat32 (TF32) tensor cores, available on A100 GPUs.
                # For more details https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere.
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            elif cfg.SOLVER.DETERMINISTIC:
                torch.set_float32_matmul_precision("highest")
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

            trainer_hooks = self._get_trainer_hooks(
                cfg, model, optimizer, scheduler, periodic_checkpointer, trainer
            )

            if comm.is_main_process():
                assert (
                    cfg.GATHER_METRIC_PERIOD <= cfg.WRITER_PERIOD
                    and cfg.WRITER_PERIOD % cfg.GATHER_METRIC_PERIOD == 0
                ), "WRITER_PERIOD needs to be divisible by GATHER_METRIC_PERIOD"
                tbx_writer = self.get_tbx_writer(cfg)
                writers = [
                    CommonMetricPrinter(max_iter, window_size=cfg.WRITER_PERIOD),
                    JSONWriter(
                        os.path.join(cfg.OUTPUT_DIR, "metrics.json"),
                        window_size=cfg.WRITER_PERIOD,
                    ),
                    tbx_writer,
                ]
                trainer_hooks.append(hooks.PeriodicWriter(writers, cfg.WRITER_PERIOD))
            update_hooks_from_registry(trainer_hooks, cfg)
            trainer.register_hooks(trainer_hooks)
            trainer.train(start_iter, max_iter)

            if hasattr(self, "original_cfg"):
                table = get_cfg_diff_table(cfg, self.original_cfg)
                logger.info(
                    "GeneralizeRCNN Runner ignoring training config change: \n" + table
                )
                trained_cfg = self.original_cfg.clone()
            else:
                trained_cfg = cfg.clone()
            with temp_defrost(trained_cfg):
                trained_cfg.MODEL.WEIGHTS = checkpointer.get_checkpoint_file()
            return {"model_final": trained_cfg}

    @staticmethod
    def get_evaluator(cfg, dataset_name, output_folder):
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            # D2 is in the process of reducing the use of cfg.
            dataset_evaluators = COCOEvaluator(
                dataset_name,
                output_dir=output_folder,
                kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS,
                max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
            )
        elif evaluator_type in ["rotated_coco"]:
            dataset_evaluators = DatasetEvaluators(
                [RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)]
            )
        elif evaluator_type in ["lvis"]:
            dataset_evaluators = LVISEvaluator(
                dataset_name,
                output_dir=output_folder,
                max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
            )
        else:
            dataset_evaluators = D2Trainer.build_evaluator(
                cfg, dataset_name, output_folder
            )
        if not isinstance(dataset_evaluators, DatasetEvaluators):
            dataset_evaluators = DatasetEvaluators([dataset_evaluators])
        return dataset_evaluators

    @staticmethod
    def final_model_name():
        return "model_final"

    def _create_after_step_hook(
        self, cfg, model, optimizer, scheduler, periodic_checkpointer
    ):
        """
        Create a hook that performs some pre-defined tasks used in this script
        (evaluation, LR scheduling, checkpointing).
        """

        def after_step_callback(trainer):
            trainer.storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
            )
            if trainer.iter < cfg.SOLVER.MAX_ITER - 1:
                # Since scheduler.step() is called after the backward at each iteration,
                # this will cause "where = 1.0" in the scheduler after the last interation,
                # which will trigger "IndexError: list index out of range" in StepParamScheduler.
                # See test_warmup_stepwithfixedgamma in vision/fair/detectron2/tests:test_scheduler for an example
                scheduler.step()
            # Note: when precise BN is enabled, some checkpoints will have more precise
            # statistics than others, if they are saved immediately after eval.
            # Note: FSDP requires all ranks to execute saving/loading logic
            if comm.is_main_process() or is_distributed_checkpoint(
                periodic_checkpointer.checkpointer
            ):
                periodic_checkpointer.step(trainer.iter)

        return hooks.CallbackHook(after_step=after_step_callback)

    def _create_data_loader_hook(self, cfg):
        """
        Create a hook for manipulating data loader
        """
        return None

    def _create_qat_hook(self, cfg) -> Optional[QATHook]:
        """
        Create a hook to start QAT (during training) and/or change the phase of QAT.
        """
        if not cfg.QUANTIZATION.QAT.ENABLED:
            return None

        return QATHook(cfg, self.build_detection_train_loader)

class BaseRunner(object):
    def __init__(self):
        identifier = f"D2Go.Runner.{self.__class__.__name__}"
        torch._C._log_api_usage_once(identifier)
        # initialize the control pg for stuff like checkpoint and preemption handling
        logger.info("Initializing control pg")
        self._control_pg: Optional[dist.ProcessGroup] = None
        if dist.is_initialized():
            logger.info("Create gloo CPU control pg")
            self._control_pg = dist.new_group(
                backend=dist.Backend.GLOO,
                timeout=CONTROL_PG_TIMEOUT,
            )

    def _initialize(self, cfg):
        """Runner should be initialized in the sub-process in ddp setting"""
        if getattr(self, "_has_initialized", False):
            logger.warning("Runner has already been initialized, skip initialization.")
            return
        self._has_initialized = True
        self.register(cfg)

    def register(self, cfg):
        """
        Override `register` in order to run customized code before other things like:
            - registering datasets.
            - registering model using Registry.
        """
        pass

    def cleanup(self) -> None:
        """
        Override `cleanup` to add custom clean ups such as:
            - de-register datasets.
            - free up global variables.
        """
        pass

    @classmethod
    def create_shared_context(cls, cfg) -> D2GoSharedContext:
        """
        Override `create_shared_context` in order to run customized code to create distributed shared context that can be accessed by all workers
        """
        pass

    @classmethod
    def get_default_cfg(cls):
        return get_base_runner_default_cfg(CfgNode())

    def build_model(self, cfg, eval_only=False) -> nn.Module:
        # cfg may need to be reused to build trace model again, thus clone
        model = build_d2go_model(cfg.clone()).model

        if eval_only:
            checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            model.eval()

        return model

    def do_test(self, *args, **kwargs):
        raise NotImplementedError()

    def do_train(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def build_detection_test_loader(cls, *args, **kwargs):
        return d2_build_detection_test_loader(*args, **kwargs)

    @classmethod
    def build_detection_train_loader(cls, *args, **kwargs):
        return d2_build_detection_train_loader(*args, **kwargs)

