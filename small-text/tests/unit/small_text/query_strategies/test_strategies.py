def query_random_data(strategy, clf=None, num_samples=100, n=10, use_embeddings=False, embedding_dim=100):

    x = np.random.rand(num_samples, 10)
    kwargs = dict()

    if use_embeddings:
        kwargs['embeddings'] = np.random.rand(SamplingStrategiesTests.DEFAULT_NUM_SAMPLES,
                                              embedding_dim)

    indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
    indices_unlabeled = np.array([i for i in range(x.shape[0])
                                  if i not in set(indices_labeled)])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    return strategy.query(clf,
                          x,
                          indices_unlabeled,
                          indices_labeled,
                          y,
                          n=n,
                          **kwargs)

