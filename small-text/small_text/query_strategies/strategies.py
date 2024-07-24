class ContrastiveActiveLearning(EmbeddingBasedQueryStrategy):
    """Contrastive Active Learning [MVB+21]_ selects instances whose k-nearest neighbours
    exhibit the largest mean Kullback-Leibler divergence."""

    def __init__(self, k=10, embed_kwargs=dict(), normalize=True, batch_size=100, pbar='tqdm'):
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbours whose KL divergence is considered.
        embed_kwargs : dict
            Embedding keyword args which are passed to `clf.embed()`.
        normalize : bool, default=True
            Embeddings will be L2 normalized if `True`, otherwise they remain unchanged.
        batch_size : int, default=100
            Batch size which is used to process the embeddings.
        pbar : 'tqdm' or None, default='tqdm'
            Displays a progress bar if 'tqdm' is passed.
        """
        self.embed_kwargs = embed_kwargs
        self.normalize = normalize
        self.k = k
        self.batch_size = batch_size
        self.pbar = pbar

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10, pbar='tqdm',
              embeddings=None, embed_kwargs=dict()):

        return super().query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n,
                             embed_kwargs=self.embed_kwargs, pbar=self.pbar)

    def sample(self, _clf, dataset, indices_unlabeled, _indices_labeled, _y, n, embeddings,
               embeddings_proba=None):
        from sklearn.neighbors import NearestNeighbors

        if embeddings_proba is None:
            raise ValueError('Error: embeddings_proba is None. '
                             'This strategy requires a classifier whose embed() method '
                             'supports the return_proba kwarg.')

        if self.normalize:
            embeddings = normalize(embeddings, axis=1)

        nn = NearestNeighbors(n_neighbors=n)
        nn.fit(embeddings)

        return self._contrastive_active_learning(dataset, embeddings, embeddings_proba,
                                                 indices_unlabeled, nn, n)

    def _contrastive_active_learning(self, dataset, embeddings, embeddings_proba,
                                     indices_unlabeled, nn, n):
        from scipy.special import rel_entr

        scores = []

        embeddings_unlabelled_proba = embeddings_proba[indices_unlabeled]
        embeddings_unlabeled = embeddings[indices_unlabeled]

        num_batches = int(np.ceil(len(dataset) / self.batch_size))
        offset = 0
        for batch_idx in np.array_split(np.arange(indices_unlabeled.shape[0]), num_batches,
                                        axis=0):

            nn_indices = nn.kneighbors(embeddings_unlabeled[batch_idx],
                                       n_neighbors=self.k,
                                       return_distance=False)

            kl_divs = np.apply_along_axis(lambda v: np.mean([
                rel_entr(embeddings_proba[i], embeddings_unlabelled_proba[v])
                for i in nn_indices[v - offset]]),
                0,
                batch_idx[None, :])

            scores.extend(kl_divs.tolist())
            offset += batch_idx.shape[0]

        scores = np.array(scores)
        indices = np.argpartition(-scores, n)[:n]

        return indices

    def __str__(self):
        return f'ContrastiveActiveLearning(k={self.k}, ' \
               f'embed_kwargs={str(self.embed_kwargs)}, ' \
               f'normalize={self.normalize})'

class RandomSampling(QueryStrategy):
    """Randomly selects instances."""

    def query(self, clf, _dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)
        return np.random.choice(indices_unlabeled, size=n, replace=False)

    def __str__(self):
        return 'RandomSampling()'

class EmbeddingKMeans(EmbeddingBasedQueryStrategy):
    """This is a generalized version of BERT-K-Means [YLB20]_, which is applicable to any kind
    of dense embedding, regardless of the classifier.
    """

    def __init__(self, normalize=True):
        """
        Parameters
        ----------
        normalize : bool, default=True
            Embeddings will be L2 normalized if `True`, otherwise they remain unchanged.
        """
        self.normalize = normalize

    def sample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n, embeddings,
               embeddings_proba=None):
        """Samples from the given embeddings.

        Parameters
        ----------
        clf : small_text.classifiers.Classifier
            A classifier.
        dataset : Dataset
            A dataset.
        indices_unlabeled : ndarray
            Indices (relative to `x`) for the unlabeled data.
        indices_labeled : ndarray
            Indices (relative to `x`) for the labeled data.
        y : ndarray or list of int
            List of labels where each label maps by index position to `indices_labeled`.
        dataset : ndarray
            Instances for which the score should be computed.
        embeddings : ndarray
            Embeddings for each sample in x.

        Returns
        -------
        indices : ndarray
            A numpy array of selected indices (relative to `indices_unlabeled`).
        """
        from sklearn.cluster import KMeans

        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)

        km = KMeans(n_clusters=n)
        km.fit(embeddings[indices_unlabeled])

        indices = self._get_nearest_to_centers(km.cluster_centers_,
                                               embeddings[indices_unlabeled],
                                               normalized=self.normalize)

        # fall back to an iterative version if one or more vectors are most similar
        # to multiple cluster centers
        if np.unique(indices).shape[0] < n:
            indices = self._get_nearest_to_centers_iterative(km.cluster_centers_,
                                                             embeddings[indices_unlabeled],
                                                             normalized=self.normalize)

        return indices

    @staticmethod
    def _get_nearest_to_centers(centers, vectors, normalized=True):
        sim = EmbeddingKMeans._similarity(centers, vectors, normalized)
        return sim.argmax(axis=1)

    @staticmethod
    def _similarity(centers, vectors, normalized):
        sim = np.matmul(centers, vectors.T)

        if not normalized:
            sim = sim / np.dot(np.linalg.norm(centers, axis=1)[:, np.newaxis],
                               np.linalg.norm(vectors, axis=1)[np.newaxis, :])
        return sim

    @staticmethod
    def _get_nearest_to_centers_iterative(cluster_centers, vectors, normalized=True):
        indices = np.empty(cluster_centers.shape[0], dtype=int)

        for i in range(cluster_centers.shape[0]):
            sim = EmbeddingKMeans._similarity(cluster_centers[None, i], vectors, normalized)
            sim[0, indices[0:i]] = -np.inf
            indices[i] = sim.argmax()

        return indices

    def __str__(self):
        return f'EmbeddingKMeans(normalize={self.normalize})'

class BreakingTies(ConfidenceBasedQueryStrategy):
    """Selects instances which have a small margin between their most likely and second
    most likely predicted class [LUO05]_.
    """

    def __init__(self):
        super().__init__(lower_is_better=True)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        return np.apply_along_axis(lambda x: self._best_versus_second_best(x), 1, proba)

    @staticmethod
    def _best_versus_second_best(proba):
        ind = np.argsort(proba)
        return proba[ind[-1]] - proba[ind[-2]]

    def __str__(self):
        return 'BreakingTies()'

class RandomSampling(QueryStrategy):
    """Randomly selects instances."""

    def query(self, clf, _dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)
        return np.random.choice(indices_unlabeled, size=n, replace=False)

    def __str__(self):
        return 'RandomSampling()'

class LeastConfidence(ConfidenceBasedQueryStrategy):
    """Selects instances with the least prediction confidence (regarding the most likely class)
    [LG94]_."""

    def __init__(self):
        super().__init__(lower_is_better=True)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        return np.amax(proba, axis=1)

    def __str__(self):
        return 'LeastConfidence()'

class SubsamplingQueryStrategy(QueryStrategy):
    """A decorator that first subsamples randomly from the unlabeled pool and then applies
    the `base_query_strategy` on the sampled subset.
    """
    def __init__(self, base_query_strategy, subsample_size=4096):
        """
        Parameters
        ----------
        base_query_strategy : QueryStrategy
            Base query strategy to which the querying is being delegated after subsampling.
        subsample_size : int, default=4096
            Size of the subsampled set.
        """
        self.base_query_strategy = base_query_strategy
        self.subsample_size = subsample_size

        self.subsampled_indices_ = None

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        if self.subsample_size > indices_unlabeled.shape[0]:
            return self.base_query_strategy.query(clf, dataset, indices_unlabeled, indices_labeled,
                                                  y, n=n)

        return self._subsample(clf, dataset, indices_unlabeled, indices_labeled, y, n)

    def _subsample(self, clf, dataset, indices_unlabeled, indices_labeled, y, n):

        subsampled_indices = np.random.choice(indices_unlabeled,
                                              self.subsample_size,
                                              replace=False)

        subset = dataset[np.concatenate([subsampled_indices, indices_labeled])]
        subset_indices_unlabeled = np.arange(self.subsample_size)
        subset_indices_labeled = np.arange(self.subsample_size,
                                           self.subsample_size + indices_labeled.shape[0])

        indices = self.base_query_strategy.query(clf,
                                                 subset,
                                                 subset_indices_unlabeled,
                                                 subset_indices_labeled,
                                                 y,
                                                 n=n)

        self.subsampled_indices_ = indices

        return np.array([subsampled_indices[i] for i in indices])

    @property
    def scores_(self):
        if hasattr(self.base_query_strategy, 'scores_'):
            return self.base_query_strategy.scores_[:self.subsample_size]
        return None

    def __str__(self):
        return f'SubsamplingQueryStrategy(base_query_strategy={self.base_query_strategy}, ' \
               f'subsample_size={self.subsample_size})'

class SEALS(QueryStrategy):
    """Similarity Search for Efficient Active Learning and Search of Rare Concepts (SEALS)
    improves the computational efficiency of active learning by presenting a reduced subset
    of the unlabeled pool to a base strategy [CCK+22]_.

    This method is to be applied in conjunction with a base query strategy. SEALS selects a
    subset of the unlabeled pool by selecting the `k` nearest neighbours of the current labeled
    pool.

    If the size of the unlabeled pool falls below the given `k`, this implementation will
    not select a subset anymore and will just delegate to the base strategy instead.

    .. note ::
       This strategy requires the optional dependency `hnswlib`.
    """
    def __init__(self, base_query_strategy, k=100, hnsw_kwargs=dict(), embed_kwargs=dict(),
                 normalize=True):
        """
        base_query_strategy : small_text.query_strategy.QueryStrategy
            A base query strategy which operates on the subset that is selected by SEALS.
        k : int, default=100
            Number of nearest neighbors that will be selected.
        hnsw_kwargs : dict(), default=dict()
            Kwargs which will be passed to the underlying hnsw index.
            Check the `hnswlib github repository <https://github.com/nmslib/hnswlib>`_ on details
            for the parameters `space`, `ef_construction`, `ef`, and `M`.
        embed_kwargs : dict, default=dict()
            Kwargs that will be passed to the embed() method.
        normalize : bool, default=True
            Embeddings will be L2 normalized if `True`, otherwise they remain unchanged.
        """
        check_optional_dependency('hnswlib')

        self.base_query_strategy = base_query_strategy
        self.k = k
        self.hnsw_kwargs = hnsw_kwargs
        self.embed_kwargs = embed_kwargs
        self.normalize = normalize

        self.nn = None

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10, pbar='tqdm'):

        if self.k > indices_unlabeled.shape[0]:
            return self.base_query_strategy.query(clf, dataset, indices_unlabeled, indices_labeled,
                                                  y, n=n)

        indices_subset = self.get_subset_indices(clf,
                                                 dataset,
                                                 indices_unlabeled,
                                                 indices_labeled,
                                                 pbar=pbar)
        return self.base_query_strategy.query(clf, dataset, indices_subset, indices_labeled, y, n=n)

    def get_subset_indices(self, clf, dataset, indices_unlabeled, indices_labeled, pbar='tqdm'):
        if self.nn is None:
            self.embeddings = clf.embed(dataset, pbar=pbar)
            if self.normalize:
                self.embeddings = normalize(self.embeddings, axis=1)

            self.nn = self.initialize_index(self.embeddings, indices_unlabeled, self.hnsw_kwargs)
            self.indices_unlabeled = set(indices_unlabeled)
        else:
            recently_removed_elements = self.indices_unlabeled - set(indices_unlabeled)
            for el in recently_removed_elements:
                self.nn.mark_deleted(el)
            self.indices_unlabeled = set(indices_unlabeled)

        indices_nn, _ = self.nn.knn_query(self.embeddings[indices_labeled], k=self.k)
        indices_nn = np.unique(indices_nn.astype(int).flatten())

        return indices_nn

    @staticmethod
    def initialize_index(embeddings, indices_unlabeled, hnsw_kwargs):
        import hnswlib

        space = hnsw_kwargs.get('space', 'l2')
        ef_construction = hnsw_kwargs.get('ef_construction', 200)
        m = hnsw_kwargs.get('M', 64)
        ef = hnsw_kwargs.get('ef', 200)

        index = hnswlib.Index(space=space, dim=embeddings.shape[1])
        index.init_index(max_elements=embeddings.shape[0],
                         ef_construction=ef_construction,
                         M=m)
        index.add_items(embeddings[indices_unlabeled], indices_unlabeled)
        index.set_ef(ef)

        return index

    def __str__(self):
        return f'SEALS(base_query_strategy={str(self.base_query_strategy)}, ' \
               f'k={self.k}, embed_kwargs={self.embed_kwargs}, normalize={self.normalize})'

