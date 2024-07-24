class Experiment(Configurable):  # pyre-ignore: 13
    """
    This class is at the top level of Implicitron's config hierarchy. Its
    members are high-level components necessary for training an implicit rende-
    ring network.

    Members:
        data_source: An object that produces datasets and dataloaders.
        model_factory: An object that produces an implicit rendering model as
            well as its corresponding Stats object.
        optimizer_factory: An object that produces the optimizer and lr
            scheduler.
        training_loop: An object that runs training given the outputs produced
            by the data_source, model_factory and optimizer_factory.
        seed: A random seed to ensure reproducibility.
        detect_anomaly: Whether torch.autograd should detect anomalies. Useful
            for debugging, but might slow down the training.
        exp_dir: Root experimentation directory. Checkpoints and training stats
            will be saved here.
    """

    data_source: DataSourceBase
    data_source_class_type: str = "ImplicitronDataSource"
    model_factory: ModelFactoryBase
    model_factory_class_type: str = "ImplicitronModelFactory"
    optimizer_factory: OptimizerFactoryBase
    optimizer_factory_class_type: str = "ImplicitronOptimizerFactory"
    training_loop: TrainingLoopBase
    training_loop_class_type: str = "ImplicitronTrainingLoop"

    seed: int = 42
    detect_anomaly: bool = False
    exp_dir: str = "./data/default_experiment/"

    hydra: dict = field(
        default_factory=lambda: {
            "run": {"dir": "."},  # Make hydra not change the working dir.
            "output_subdir": None,  # disable storing the .hydra logs
            "mode": _RUN,
        }
    )

    def __post_init__(self):
        seed_all_random_engines(
            self.seed
        )  # Set all random engine seeds for reproducibility

        run_auto_creation(self)

    def run(self) -> None:
        # Initialize the accelerator if desired.
        if no_accelerate:
            accelerator = None
            device = torch.device("cuda:0")
        else:
            accelerator = Accelerator(device_placement=False)
            logger.info(accelerator.state)
            device = accelerator.device

        logger.info(f"Running experiment on device: {device}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # set the debug mode
        if self.detect_anomaly:
            logger.info("Anomaly detection!")
        torch.autograd.set_detect_anomaly(self.detect_anomaly)

        # Initialize the datasets and dataloaders.
        datasets, dataloaders = self.data_source.get_datasets_and_dataloaders()

        # Init the model and the corresponding Stats object.
        model = self.model_factory(
            accelerator=accelerator,
            exp_dir=self.exp_dir,
        )

        stats = self.training_loop.load_stats(
            log_vars=model.log_vars,
            exp_dir=self.exp_dir,
            resume=self.model_factory.resume,
            resume_epoch=self.model_factory.resume_epoch,  # pyre-ignore [16]
        )
        start_epoch = stats.epoch + 1

        model.to(device)

        # Init the optimizer and LR scheduler.
        optimizer, scheduler = self.optimizer_factory(
            accelerator=accelerator,
            exp_dir=self.exp_dir,
            last_epoch=start_epoch,
            model=model,
            resume=self.model_factory.resume,
            resume_epoch=self.model_factory.resume_epoch,
        )

        # Wrap all modules in the distributed library
        # Note: we don't pass the scheduler to prepare as it
        # doesn't need to be stepped at each optimizer step
        train_loader = dataloaders.train
        val_loader = dataloaders.val
        test_loader = dataloaders.test
        if accelerator is not None:
            (
                model,
                optimizer,
                train_loader,
                val_loader,
            ) = accelerator.prepare(model, optimizer, train_loader, val_loader)

        # Enter the main training loop.
        self.training_loop.run(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            # pyre-ignore[6]
            train_dataset=datasets.train,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            device=device,
            exp_dir=self.exp_dir,
            stats=stats,
            seed=self.seed,
        )

