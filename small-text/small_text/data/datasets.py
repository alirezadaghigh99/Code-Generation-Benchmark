class SklearnDataset(_InferLabelsMixin, Dataset):
    """A dataset representations which is usable in combination with scikit-learn classifiers.
    """

    def __init__(self, x, y, target_labels=None):
        """
        Parameters
        ----------
        x : numpy.ndarray or scipy.sparse.csr_matrix
            Dense or sparse feature matrix.
        y : numpy.ndarray[int] or scipy.sparse.csr_matrix
            List of labels where each label belongs to the features of the respective row.
        target_labels : numpy.ndarray[int] or None, default=None
            List of possible labels. Will be inferred from `y` if `None` is passed."""
        check_dataset_and_labels(x, y)
        check_target_labels(target_labels)

        self._x = x
        self._y = y

        self.multi_label = is_multi_label(self._y)

        if target_labels is not None:
            self.track_target_labels = False
            self.target_labels = target_labels
        else:
            self.track_target_labels = True
            self._infer_target_labels()

    @property
    def x(self):
        """Returns the features.

        Returns
        -------
        x : numpy.ndarray or scipy.sparse.csr_matrix
            Dense or sparse feature matrix.
        """
        return self._x

    @x.setter
    def x(self, x):
        check_dataset_and_labels(x, self._y)
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        check_dataset_and_labels(self.x, y)
        self._y = y

        if self.track_target_labels:
            self.target_labels = get_updated_target_labels(self.is_multi_label, y, self.target_labels)
        else:
            max_label_id = np.max(y)
            max_target_labels_id = self.target_labels.max()
            if max_label_id > max_target_labels_id:
                raise ValueError(f'Error while assigning new labels to dataset: '
                                 f'Encountered label with id {max_label_id} which is outside of '
                                 f'the configured set of target labels (whose maximum label is '
                                 f'is {max_target_labels_id}) [track_target_labels=False]')

    @property
    def is_multi_label(self):
        return self.multi_label

    @property
    def target_labels(self):
        """Returns a list of possible labels.

        Returns
        -------
        target_labels : numpy.ndarray
            List of possible labels.
        """
        return self._target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        encountered_labels = get_flattened_unique_labels(self)
        if np.setdiff1d(encountered_labels, target_labels).shape[0] > 0:
            raise ValueError('Cannot remove existing labels from target_labels as long as they '
                             'still exists in the data. Create a new dataset instead.')
        self._target_labels = target_labels

    def clone(self):
        if isinstance(self._x, csr_matrix):
            x = self._x.copy()
        else:
            x = np.copy(self._x)

        if isinstance(self._y, csr_matrix):
            y = self._y.copy()
        else:
            y = np.copy(self._y)

        if self.track_target_labels:
            target_labels = None
        else:
            target_labels = np.copy(self._target_labels)

        return SklearnDataset(x, y, target_labels=target_labels)

    @classmethod
    @experimental
    def from_arrays(cls, texts, y, vectorizer, target_labels=None, train=True):
        """Constructs a new SklearnDataset from the given text and label arrays.

        Parameters
        ----------
        texts : list of str or np.ndarray[str]
            List of text documents.
        y : np.ndarray[int] or scipy.sparse.csr_matrix
            List of labels where each label belongs to the features of the respective row.
            Depending on the type of `y` the resulting dataset will be single-label (`np.ndarray`)
            or multi-label (`scipy.sparse.csr_matrix`).
        vectorizer : object
            A scikit-learn vectorizer which is used to construct the feature matrix.
        target_labels : numpy.ndarray[int] or None, default=None
            List of possible labels. Will be directly passed to the datset constructor.
        train : bool
            If `True` fits the vectorizer and transforms the data, otherwise just transforms the
            data.

        Returns
        -------
        dataset : SklearnDataset
            A dataset constructed from the given texts and labels.


        .. seealso::
           `scikit-learn docs: Vectorizer API reference
           <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
           #sklearn.feature_extraction.text.TfidfVectorizer>`__

        .. warning::
           This functionality is still experimental and may be subject to change.

        .. versionadded:: 1.1.0
        """
        if train:
            x = vectorizer.fit_transform(texts)
        else:
            x = vectorizer.transform(texts)

        return SklearnDataset(x, y, target_labels=target_labels)

    def __getitem__(self, item):
        return SklearnDatasetView(self, item)

    def __len__(self):
        return self._x.shape[0]

class TextDataset(_InferLabelsMixin, Dataset):
    """A dataset representation consisting of raw text data.
    """

    def __init__(self, x, y, target_labels=None):
        """
        Parameters
        ----------
        x : list of str
            List of texts.
        y : numpy.ndarray[int] or scipy.sparse.csr_matrix
            List of labels where each label belongs to the features of the respective row.
        target_labels : numpy.ndarray[int] or None, default=None
            List of possible labels. Will be inferred from `y` if `None` is passed."""
        check_dataset_and_labels(x, y)
        check_target_labels(target_labels)

        if isinstance(x, np.ndarray):
            x = x.tolist()

        self._x = x
        self._y = y

        self.multi_label = is_multi_label(self._y)

        if target_labels is not None:
            self.track_target_labels = False
            self.target_labels = target_labels
        else:
            self.track_target_labels = True
            self._infer_target_labels()

    @property
    def x(self):
        """Returns the features.

        Returns
        -------
        x : list of str
            List of texts.
        """
        return self._x

    @x.setter
    def x(self, x):
        check_dataset_and_labels(x, self._y)
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        check_dataset_and_labels(self.x, y)
        self._y = y

        if self.track_target_labels:
            self.target_labels = get_updated_target_labels(self.is_multi_label, y, self.target_labels)
        else:
            max_label_id = np.max(y)
            max_target_labels_id = self.target_labels.max()
            if max_label_id > max_target_labels_id:
                raise ValueError(f'Error while assigning new labels to dataset: '
                                 f'Encountered label with id {max_label_id} which is outside of '
                                 f'the configured set of target labels (whose maximum label is '
                                 f'is {max_target_labels_id}) [track_target_labels=False]')

    @property
    def is_multi_label(self):
        return self.multi_label

    @property
    def target_labels(self):
        """Returns a list of possible labels.

        Returns
        -------
        target_labels : numpy.ndarray
            List of possible labels.
        """
        return self._target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        encountered_labels = get_flattened_unique_labels(self)
        if np.setdiff1d(encountered_labels, target_labels).shape[0] > 0:
            raise ValueError('Cannot remove existing labels from target_labels as long as they '
                             'still exists in the data. Create a new dataset instead.')
        self._target_labels = target_labels

    def clone(self):
        x = copy(self._x)

        if isinstance(self._y, csr_matrix):
            y = self._y.copy()
        else:
            y = np.copy(self._y)

        if self.track_target_labels:
            target_labels = None
        else:
            target_labels = np.copy(self._target_labels)

        return TextDataset(x, y, target_labels=target_labels)

    @classmethod
    @experimental
    def from_arrays(cls, texts, y, target_labels=None):
        """Constructs a new TextDataset from the given text and label arrays.

        Parameters
        ----------
        texts : list of str
            List of text documents.
        y : np.ndarray[int] or scipy.sparse.csr_matrix
            List of labels where each label belongs to the features of the respective row.
            Depending on the type of `y` the resulting dataset will be single-label (`np.ndarray`)
            or multi-label (`scipy.sparse.csr_matrix`).
        target_labels : numpy.ndarray[int] or None, default=None
            List of possible labels. Will be directly passed to the datset constructor.

        Returns
        -------
        dataset : SklearnDataset
            A dataset constructed from the given texts and labels.


        .. seealso::
           `scikit-learn docs: Vectorizer API reference
           <https://scikit-learn.org/stable/modules/classes.html
           #module-sklearn.feature_extraction.text>`_

        .. warning::
           This functionality is still experimental and may be subject to change.

        .. versionadded:: 1.2.0
        """

        return TextDataset(texts, y, target_labels=target_labels)

    def __getitem__(self, item):
        return TextDatasetView(self, item)

    def __len__(self):
        return len(self._x)

