def get_dataset(mode='classification', featurizer='GraphConv', num_tasks=2):
    data_points = 20
    if mode == 'classification':
        tasks, all_dataset, transformers = load_bace_classification(featurizer)
    else:
        tasks, all_dataset, transformers = load_delaney(featurizer)

    train, valid, test = all_dataset
    for _ in range(1, num_tasks):
        tasks.append("random_task")
    w = np.ones(shape=(data_points, len(tasks)))

    if mode == 'classification':
        y = np.random.randint(0, 2, size=(data_points, len(tasks)))
        metric = Metric(roc_auc_score, np.mean, mode="classification")
    else:
        y = np.random.normal(size=(data_points, len(tasks)))
        metric = Metric(mean_absolute_error, mode="regression")

    ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

    return tasks, ds, transformers, metric

