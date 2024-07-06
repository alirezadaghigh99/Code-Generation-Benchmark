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

