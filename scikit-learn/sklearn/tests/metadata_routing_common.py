def assert_request_equal(request, dictionary):
    for method, requests in dictionary.items():
        mmr = getattr(request, method)
        assert mmr.requests == requests

    empty_methods = [method for method in SIMPLE_METHODS if method not in dictionary]
    for method in empty_methods:
        assert not len(getattr(request, method).requests)

def assert_request_is_empty(metadata_request, exclude=None):
    """Check if a metadata request dict is empty.

    One can exclude a method or a list of methods from the check using the
    ``exclude`` parameter. If metadata_request is a MetadataRouter, then
    ``exclude`` can be of the form ``{"object" : [method, ...]}``.
    """
    if isinstance(metadata_request, MetadataRouter):
        for name, route_mapping in metadata_request:
            if exclude is not None and name in exclude:
                _exclude = exclude[name]
            else:
                _exclude = None
            assert_request_is_empty(route_mapping.router, exclude=_exclude)
        return

    exclude = [] if exclude is None else exclude
    for method in SIMPLE_METHODS:
        if method in exclude:
            continue
        mmr = getattr(metadata_request, method)
        props = [
            prop
            for prop, alias in mmr.requests.items()
            if isinstance(alias, str) or alias is not None
        ]
        assert not props

class ConsumingClassifier(ClassifierMixin, BaseEstimator):
    """A classifier consuming metadata.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.

    alpha : float, default=0
        This parameter is only used to test the ``*SearchCV`` objects, and
        doesn't do anything.
    """

    def __init__(self, registry=None, alpha=0.0):
        self.alpha = alpha
        self.registry = registry

    def partial_fit(
        self, X, y, classes=None, sample_weight="default", metadata="default"
    ):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        _check_partial_fit_first_call(self, classes)
        return self

    def fit(self, X, y, sample_weight="default", metadata="default"):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )

        self.classes_ = np.unique(y)
        return self

    def predict(self, X, sample_weight="default", metadata="default"):
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        y_score = np.empty(shape=(len(X),), dtype="int8")
        y_score[len(X) // 2 :] = 0
        y_score[: len(X) // 2] = 1
        return y_score

    def predict_proba(self, X, sample_weight="default", metadata="default"):
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        y_proba = np.empty(shape=(len(X), 2))
        y_proba[: len(X) // 2, :] = np.asarray([1.0, 0.0])
        y_proba[len(X) // 2 :, :] = np.asarray([0.0, 1.0])
        return y_proba

    def predict_log_proba(self, X, sample_weight="default", metadata="default"):
        pass  # pragma: no cover

        # uncomment when needed
        # record_metadata_not_default(
        #     self, sample_weight=sample_weight, metadata=metadata
        # )
        # return np.zeros(shape=(len(X), 2))

    def decision_function(self, X, sample_weight="default", metadata="default"):
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        y_score = np.empty(shape=(len(X),))
        y_score[len(X) // 2 :] = 0
        y_score[: len(X) // 2] = 1
        return y_score

class _Registry(list):
    # This list is used to get a reference to the sub-estimators, which are not
    # necessarily stored on the metaestimator. We need to override __deepcopy__
    # because the sub-estimators are probably cloned, which would result in a
    # new copy of the list, but we need copy and deep copy both to return the
    # same instance.
    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

class MetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """A meta-regressor which is only a router."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        params = process_routing(self, "fit", **fit_params)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)

    def get_metadata_routing(self):
        router = MetadataRouter(owner=self.__class__.__name__).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

class ConsumingRegressor(RegressorMixin, BaseEstimator):
    """A regressor consuming metadata.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.
    """

    def __init__(self, registry=None):
        self.registry = registry

    def partial_fit(self, X, y, sample_weight="default", metadata="default"):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return self

    def fit(self, X, y, sample_weight="default", metadata="default"):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return self

    def predict(self, X, y=None, sample_weight="default", metadata="default"):
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return np.zeros(shape=(len(X),))

    def score(self, X, y, sample_weight="default", metadata="default"):
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return 1

class WeightedMetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """A meta-regressor which is also a consumer."""

    def __init__(self, estimator, registry=None):
        self.estimator = estimator
        self.registry = registry

    def fit(self, X, y, sample_weight=None, **fit_params):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata(self, sample_weight=sample_weight)
        params = process_routing(self, "fit", sample_weight=sample_weight, **fit_params)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)
        return self

    def predict(self, X, **predict_params):
        params = process_routing(self, "predict", **predict_params)
        return self.estimator_.predict(X, **params.estimator.predict)

    def get_metadata_routing(self):
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="predict", callee="predict"),
            )
        )
        return router

class WeightedMetaClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """A meta-estimator which also consumes sample_weight itself in ``fit``."""

    def __init__(self, estimator, registry=None):
        self.estimator = estimator
        self.registry = registry

    def fit(self, X, y, sample_weight=None, **kwargs):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata(self, sample_weight=sample_weight)
        params = process_routing(self, "fit", sample_weight=sample_weight, **kwargs)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)
        return self

    def get_metadata_routing(self):
        router = (
            MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router

class ConsumingTransformer(TransformerMixin, BaseEstimator):
    """A transformer which accepts metadata on fit and transform.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.
    """

    def __init__(self, registry=None):
        self.registry = registry

    def fit(self, X, y=None, sample_weight="default", metadata="default"):
        if self.registry is not None:
            self.registry.append(self)

        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return self

    def transform(self, X, sample_weight="default", metadata="default"):
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return X + 1

    def fit_transform(self, X, y, sample_weight="default", metadata="default"):
        # implementing ``fit_transform`` is necessary since
        # ``TransformerMixin.fit_transform`` doesn't route any metadata to
        # ``transform``, while here we want ``transform`` to receive
        # ``sample_weight`` and ``metadata``.
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return self.fit(X, y, sample_weight=sample_weight, metadata=metadata).transform(
            X, sample_weight=sample_weight, metadata=metadata
        )

    def inverse_transform(self, X, sample_weight=None, metadata=None):
        record_metadata_not_default(
            self, sample_weight=sample_weight, metadata=metadata
        )
        return X - 1

