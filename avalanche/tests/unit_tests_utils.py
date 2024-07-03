def get_fast_benchmark(
    use_task_labels=False,
    shuffle=True,
    n_samples_per_class=100,
    n_classes=10,
    n_features=6,
    seed=None,
    train_transform=None,
    eval_transform=None,
):
    train, test = dummy_classification_datasets(
        n_classes, n_features, n_samples_per_class, seed
    )
    my_nc_benchmark = nc_benchmark(
        train,
        test,
        5,
        task_labels=use_task_labels,
        shuffle=shuffle,
        train_transform=train_transform,
        eval_transform=eval_transform,
        seed=seed,
    )
    return my_nc_benchmarkdef dummy_image_dataset():
    """Returns a PyTorch image dataset of 10 classes."""
    global image_data

    if image_data is None:
        image_data = MNIST(
            root=default_dataset_location("mnist"),
            train=True,
            download=True,
        )
    return image_data