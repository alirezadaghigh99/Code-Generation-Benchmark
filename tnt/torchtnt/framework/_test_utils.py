def generate_random_dataloader(
    num_samples: int, input_dim: int, batch_size: int
) -> DataLoader:
    return DataLoader(
        generate_random_dataset(num_samples, input_dim),
        batch_size=batch_size,
    )

def get_dummy_train_state(dataloader: Optional[Iterable[object]] = None) -> State:
    return State(
        entry_point=EntryPoint.TRAIN,
        train_state=PhaseState(
            dataloader=dataloader or [1, 2, 3, 4],
            max_epochs=1,
            max_steps=1,
            max_steps_per_epoch=1,
        ),
        timer=None,
    )

class DummyTrainUnit(TrainUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module, loss_fn, & optimizer
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

    def train_step(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss, outputs

class DummyAutoUnit(AutoUnit[Batch]):
    def compute_loss(self, state: State, data: Batch) -> Tuple[torch.Tensor, object]:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: torch.nn.Module
    ) -> Tuple[torch.optim.Optimizer, TLRScheduler]:
        my_optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            my_optimizer, gamma=0.9
        )
        return my_optimizer, my_lr_scheduler

class DummyPredictUnit(PredictUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module
        self.module = nn.Linear(input_dim, 2)

    def predict_step(self, state: State, data: Batch) -> torch.Tensor:
        inputs, targets = data

        outputs = self.module(inputs)
        return outputs

class DummyEvalUnit(EvalUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module & loss_fn
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def eval_step(self, state: State, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs

class DummyFitUnit(TrainUnit[Batch], EvalUnit[Batch]):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # initialize module, loss_fn, & optimizer
        self.module = nn.Linear(input_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

    def train_step(
        self, state: State, data: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss, outputs

    def eval_step(self, state: State, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = data

        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs

