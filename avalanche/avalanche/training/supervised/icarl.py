class ICaRL(SupervisedTemplate):
    """iCaRL Strategy.

    This strategy does not use task identities.
    """

    def __init__(
        self,
        *,
        feature_extractor: Module,
        classifier: Module,
        optimizer: Optimizer,
        memory_size: int,
        buffer_transform,
        fixed_memory: bool,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
    ):
        """Init.

        :param feature_extractor: The feature extractor.
        :param classifier: The differentiable classifier that takes as input
            the output of the feature extractor.
        :param optimizer: The optimizer to use.
        :param memory_size: The nuber of patterns saved in the memory.
        :param buffer_transform: transform applied on buffer elements already
            modified by test_transform (if specified) before being used for
            replay
        :param fixed_memory: If True a memory of size memory_size is
            allocated and partitioned between samples from the observed
            experiences. If False every time a new class is observed
            memory_size samples of that class are added to the memory.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """
        model = TrainEvalModel(
            feature_extractor,
            train_classifier=classifier,
            eval_classifier=NCMClassifier(normalize=True),
        )

        criterion = ICaRLLossPlugin()  # iCaRL requires this specific loss (#966)
        icarl = _ICaRLPlugin(
            memory_size,
            buffer_transform,
            fixed_memory,
        )

        if plugins is None:
            plugins = [icarl]
        else:
            plugins += [icarl]

        if isinstance(criterion, SupervisedPlugin):
            plugins += [criterion]

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

