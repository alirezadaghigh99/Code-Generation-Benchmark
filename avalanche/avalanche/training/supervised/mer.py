class MER(SupervisedMetaLearningTemplate):
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType = CrossEntropyLoss(),
        mem_size=200,
        batch_size_mem=10,
        n_inner_steps=5,
        beta=0.1,
        gamma=0.1,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
        **kwargs
    ):
        """Implementation of Look-ahead MAML (LaMAML) algorithm in Avalanche
            using Higher library for applying fast updates.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: maximum size of the buffer.
        :param batch_size_mem: number of samples to retrieve from buffer
            for each sample.
        :param n_inner_steps: number of inner updates per sample.
        :param beta: coefficient for within-batch Reptile update.
        :param gamma: coefficient for within-task Reptile update.

        """
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
            peval_mode=peval_mode,
            **kwargs
        )

        self.buffer = MERBuffer(
            mem_size=mem_size,
            batch_size_mem=batch_size_mem,
            device=self.device,
        )
        self.n_inner_steps = n_inner_steps
        self.beta = beta
        self.gamma = gamma

    def _before_inner_updates(self, **kwargs):
        self.w_bef = deepcopy(self.model.state_dict())
        super()._before_inner_updates(**kwargs)

    def _inner_updates(self, **kwargs):
        for inner_itr in range(self.n_inner_steps):
            x, y, t = self.mb_x, self.mb_y, self.mb_task_id
            x, y, t = self.buffer.get_batch(x, y, t)

            # Inner updates
            w_bef_t = deepcopy(self.model.state_dict())
            for idx in range(x.shape[0]):
                x_b = x[idx].unsqueeze(0)
                y_b = y[idx].unsqueeze(0)
                t_b = t[idx].unsqueeze(0)
                self.model.zero_grad()
                pred = avalanche_forward(self.model, x_b, t_b)
                loss = self._criterion(pred, y_b)
                loss.backward()
                self.optimizer.step()

            # Within-batch Reptile update
            w_aft_t = self.model.state_dict()
            load_dict = {}
            for name, param in self.model.named_parameters():
                load_dict[name] = w_bef_t[name] + (
                    (w_aft_t[name] - w_bef_t[name]) * self.beta
                )

            self.model.load_state_dict(load_dict, strict=False)

    def _outer_update(self, **kwargs):
        w_aft = self.model.state_dict()

        load_dict = {}
        for name, param in self.model.named_parameters():
            load_dict[name] = self.w_bef[name] + (
                (w_aft[name] - self.w_bef[name]) * self.gamma
            )

        self.model.load_state_dict(load_dict, strict=False)

        with torch.no_grad():
            pred = self.forward()
            self.loss = self._criterion(pred, self.mb_y)

    def _after_training_exp(self, **kwargs):
        self.buffer.update(self)
        super()._after_training_exp(**kwargs)

