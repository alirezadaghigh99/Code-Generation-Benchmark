def _get_visual_block(estimator):
    """Generate information about how to display an estimator."""
    if hasattr(estimator, "_sk_visual_block_"):
        try:
            return estimator._sk_visual_block_()
        except Exception:
            return _VisualBlock(
                "single",
                estimator,
                names=estimator.__class__.__name__,
                name_details=str(estimator),
            )

    if isinstance(estimator, str):
        return _VisualBlock(
            "single", estimator, names=estimator, name_details=estimator
        )
    elif estimator is None:
        return _VisualBlock("single", estimator, names="None", name_details="None")

    # check if estimator looks like a meta estimator (wraps estimators)
    if hasattr(estimator, "get_params") and not isclass(estimator):
        estimators = [
            (key, est)
            for key, est in estimator.get_params(deep=False).items()
            if hasattr(est, "get_params") and hasattr(est, "fit") and not isclass(est)
        ]
        if estimators:
            return _VisualBlock(
                "parallel",
                [est for _, est in estimators],
                names=[f"{key}: {est.__class__.__name__}" for key, est in estimators],
                name_details=[str(est) for _, est in estimators],
            )

    return _VisualBlock(
        "single",
        estimator,
        names=estimator.__class__.__name__,
        name_details=str(estimator),
    )

class _HTMLDocumentationLinkMixin:
    """Mixin class allowing to generate a link to the API documentation.

    This mixin relies on three attributes:
    - `_doc_link_module`: it corresponds to the root module (e.g. `sklearn`). Using this
      mixin, the default value is `sklearn`.
    - `_doc_link_template`: it corresponds to the template used to generate the
      link to the API documentation. Using this mixin, the default value is
      `"https://scikit-learn.org/{version_url}/modules/generated/
      {estimator_module}.{estimator_name}.html"`.
    - `_doc_link_url_param_generator`: it corresponds to a function that generates the
      parameters to be used in the template when the estimator module and name are not
      sufficient.

    The method :meth:`_get_doc_link` generates the link to the API documentation for a
    given estimator.

    This useful provides all the necessary states for
    :func:`sklearn.utils.estimator_html_repr` to generate a link to the API
    documentation for the estimator HTML diagram.

    Examples
    --------
    If the default values for `_doc_link_module`, `_doc_link_template` are not suitable,
    then you can override them:
    >>> from sklearn.base import BaseEstimator
    >>> estimator = BaseEstimator()
    >>> estimator._doc_link_template = "https://website.com/{single_param}.html"
    >>> def url_param_generator(estimator):
    ...     return {"single_param": estimator.__class__.__name__}
    >>> estimator._doc_link_url_param_generator = url_param_generator
    >>> estimator._get_doc_link()
    'https://website.com/BaseEstimator.html'
    """

    _doc_link_module = "sklearn"
    _doc_link_url_param_generator = None

    @property
    def _doc_link_template(self):
        sklearn_version = parse_version(__version__)
        if sklearn_version.dev is None:
            version_url = f"{sklearn_version.major}.{sklearn_version.minor}"
        else:
            version_url = "dev"
        return getattr(
            self,
            "__doc_link_template",
            (
                f"https://scikit-learn.org/{version_url}/modules/generated/"
                "{estimator_module}.{estimator_name}.html"
            ),
        )

    @_doc_link_template.setter
    def _doc_link_template(self, value):
        setattr(self, "__doc_link_template", value)

    def _get_doc_link(self):
        """Generates a link to the API documentation for a given estimator.

        This method generates the link to the estimator's documentation page
        by using the template defined by the attribute `_doc_link_template`.

        Returns
        -------
        url : str
            The URL to the API documentation for this estimator. If the estimator does
            not belong to module `_doc_link_module`, the empty string (i.e. `""`) is
            returned.
        """
        if self.__class__.__module__.split(".")[0] != self._doc_link_module:
            return ""

        if self._doc_link_url_param_generator is None:
            estimator_name = self.__class__.__name__
            # Construct the estimator's module name, up to the first private submodule.
            # This works because in scikit-learn all public estimators are exposed at
            # that level, even if they actually live in a private sub-module.
            estimator_module = ".".join(
                itertools.takewhile(
                    lambda part: not part.startswith("_"),
                    self.__class__.__module__.split("."),
                )
            )
            return self._doc_link_template.format(
                estimator_module=estimator_module, estimator_name=estimator_name
            )
        return self._doc_link_template.format(
            **self._doc_link_url_param_generator(self)
        )

