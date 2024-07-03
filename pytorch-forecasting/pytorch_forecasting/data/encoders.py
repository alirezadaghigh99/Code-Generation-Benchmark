class GroupNormalizer(TorchNormalizer):
    """
    Normalizer that scales by groups.

    For each group a scaler is fitted and applied. This scaler can be used as target normalizer or
    also to normalize any other variable.
    """

    def __init__(
        self,
        method: str = "standard",
        groups: List[str] = [],
        center: bool = True,
        scale_by_group: bool = False,
        transformation: Union[str, Tuple[Callable, Callable]] = None,
        method_kwargs: Dict[str, Any] = {},
    ):
        """
        Group normalizer to normalize a given entry by groups. Can be used as target normalizer.

        Args:
            method (str, optional): method to rescale series. Either "standard" (standard scaling) or "robust"
                (scale using quantiles 0.25-0.75). Defaults to "standard".
            method_kwargs (Dict[str, Any], optional): Dictionary of method specific arguments as listed below
                * "robust" method: "upper", "lower", "center" quantiles defaulting to 0.75, 0.25 and 0.5
            groups (List[str], optional): Group names to normalize by. Defaults to [].
            center (bool, optional): If to center the output to zero. Defaults to True.
            scale_by_group (bool, optional): If to scale the output by group, i.e. norm is calculated as
                ``(group1_norm * group2_norm * ...) ^ (1 / n_groups)``. Defaults to False.
            transformation (Union[str, Tuple[Callable, Callable]] optional): Transform values before
                applying normalizer. Available options are

                * None (default): No transformation of values
                * log: Estimate in log-space leading to a multiplicative model
                * log1p: Estimate in log-space but add 1 to values before transforming for stability
                    (e.g. if many small values <<1 are present).
                    Note, that inverse transform is still only `torch.exp()` and not `torch.expm1()`.
                * logit: Apply logit transformation on values that are between 0 and 1
                * count: Apply softplus to output (inverse transformation) and x + 1 to input
                    (transformation)
                * softplus: Apply softplus to output (inverse transformation) and inverse softplus to input
                    (transformation)
                * relu: Apply max(0, x) to output
                * Dict[str, Callable] of PyTorch functions that transforms and inversely transforms values.
                  ``forward`` and ``reverse`` entries are required. ``inverse`` transformation is optional and
                  should be defined if ``reverse`` is not the inverse of the forward transformation. ``inverse_torch``
                  can be defined to provide a torch distribution transform for inverse transformations.

        """
        self.groups = groups
        self.scale_by_group = scale_by_group
        super().__init__(method=method, center=center, transformation=transformation, method_kwargs=method_kwargs)

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """
        Determine scales for each group

        Args:
            y (pd.Series): input data
            X (pd.DataFrame): dataframe with columns for each group defined in ``groups`` parameter.

        Returns:
            self
        """
        y = self.preprocess(y)
        eps = np.finfo(np.float16).eps
        if len(self.groups) == 0:
            assert not self.scale_by_group, "No groups are defined, i.e. `scale_by_group=[]`"
            if self.method == "standard":
                self.norm_ = {"center": np.mean(y), "scale": np.std(y) + eps}  # center and scale
            else:
                quantiles = np.quantile(
                    y,
                    [
                        self.method_kwargs.get("lower", 0.25),
                        self.method_kwargs.get("center", 0.5),
                        self.method_kwargs.get("upper", 0.75),
                    ],
                )
                self.norm_ = {
                    "center": quantiles[1],
                    "scale": (quantiles[2] - quantiles[0]) / 2.0 + eps,
                }  # center and scale
            if not self.center:
                self.norm_["scale"] = self.norm_["center"] + eps
                self.norm_["center"] = 0.0

        elif self.scale_by_group:
            if self.method == "standard":
                self.norm_ = {
                    g: X[[g]]
                    .assign(y=y)
                    .groupby(g, observed=True)
                    .agg(center=("y", "mean"), scale=("y", "std"))
                    .assign(center=lambda x: x["center"], scale=lambda x: x.scale + eps)
                    for g in self.groups
                }
            else:
                self.norm_ = {
                    g: X[[g]]
                    .assign(y=y)
                    .groupby(g, observed=True)
                    .y.quantile(
                        [
                            self.method_kwargs.get("lower", 0.25),
                            self.method_kwargs.get("center", 0.5),
                            self.method_kwargs.get("upper", 0.75),
                        ]
                    )
                    .unstack(-1)
                    .assign(
                        center=lambda x: x[self.method_kwargs.get("center", 0.5)],
                        scale=lambda x: (
                            x[self.method_kwargs.get("upper", 0.75)] - x[self.method_kwargs.get("lower", 0.25)]
                        )
                        / 2.0
                        + eps,
                    )[["center", "scale"]]
                    for g in self.groups
                }
            # calculate missings
            if not self.center:  # swap center and scale

                def swap_parameters(norm):
                    norm["scale"] = norm["center"] + eps
                    norm["center"] = 0.0
                    return norm

                self.norm_ = {g: swap_parameters(norm) for g, norm in self.norm_.items()}
            self.missing_ = {group: scales.median().to_dict() for group, scales in self.norm_.items()}

        else:
            if self.method == "standard":
                self.norm_ = (
                    X[self.groups]
                    .assign(y=y)
                    .groupby(self.groups, observed=True)
                    .agg(center=("y", "mean"), scale=("y", "std"))
                    .assign(center=lambda x: x["center"], scale=lambda x: x.scale + eps)
                )
            else:
                self.norm_ = (
                    X[self.groups]
                    .assign(y=y)
                    .groupby(self.groups, observed=True)
                    .y.quantile(
                        [
                            self.method_kwargs.get("lower", 0.25),
                            self.method_kwargs.get("center", 0.5),
                            self.method_kwargs.get("upper", 0.75),
                        ]
                    )
                    .unstack(-1)
                    .assign(
                        center=lambda x: x[self.method_kwargs.get("center", 0.5)],
                        scale=lambda x: (
                            x[self.method_kwargs.get("upper", 0.75)] - x[self.method_kwargs.get("lower", 0.25)]
                        )
                        / 2.0
                        + eps,
                    )[["center", "scale"]]
                )
            if not self.center:  # swap center and scale
                self.norm_["scale"] = self.norm_["center"] + eps
                self.norm_["center"] = 0.0
            self.missing_ = self.norm_.median().to_dict()

        if (
            (self.scale_by_group and any([(self.norm_[group]["scale"] < 1e-7).any() for group in self.groups]))
            or (not self.scale_by_group and isinstance(self.norm_["scale"], float) and self.norm_["scale"] < 1e-7)
            or (
                not self.scale_by_group
                and not isinstance(self.norm_["scale"], float)
                and (self.norm_["scale"] < 1e-7).any()
            )
        ):
            warnings.warn(
                "scale is below 1e-7 - consider not centering "
                "the data or using data with higher variance for numerical stability",
                UserWarning,
            )

        return self

    @property
    def names(self) -> List[str]:
        """
        Names of determined scales.

        Returns:
            List[str]: list of names
        """
        return ["center", "scale"]

    def fit_transform(
        self, y: pd.Series, X: pd.DataFrame, return_norm: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Fit normalizer and scale input data.

        Args:
            y (pd.Series): data to scale
            X (pd.DataFrame): dataframe with ``groups`` columns
            return_norm (bool, optional): If to return . Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Scaled data, if ``return_norm=True``, returns also scales
                as second element
        """
        return self.fit(y, X).transform(y, X, return_norm=return_norm)

    def inverse_transform(self, y: pd.Series, X: pd.DataFrame):
        """
        Rescaling data to original scale - not implemented - call class with target scale instead.
        """
        raise NotImplementedError()

    def transform(
        self, y: pd.Series, X: pd.DataFrame = None, return_norm: bool = False, target_scale: torch.Tensor = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Scale input data.

        Args:
            y (pd.Series): data to scale
            X (pd.DataFrame): dataframe with ``groups`` columns
            return_norm (bool, optional): If to return . Defaults to False.
            target_scale (torch.Tensor): target scale to use instead of fitted center and scale

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Scaled data, if ``return_norm=True``, returns also scales
                as second element
        """
        # # check if arguments are wrong way round
        if isinstance(y, pd.DataFrame) and not isinstance(X, pd.DataFrame):
            raise ValueError("X and y is in wrong positions")
        if target_scale is None:
            assert X is not None, "either target_scale or X has to be passed"
            target_scale = self.get_norm(X)
        return super().transform(y, return_norm=return_norm, target_scale=target_scale)

    def get_parameters(self, groups: Union[torch.Tensor, list, tuple], group_names: List[str] = None) -> np.ndarray:
        """
        Get fitted scaling parameters for a given group.

        Args:
            groups (Union[torch.Tensor, list, tuple]): group ids for which to get parameters
            group_names (List[str], optional): Names of groups corresponding to positions
                in ``groups``. Defaults to None, i.e. the instance attribute ``groups``.

        Returns:
            np.ndarray: parameters used for scaling
        """
        if isinstance(groups, torch.Tensor):
            groups = groups.tolist()
        if isinstance(groups, list):
            groups = tuple(groups)
        if group_names is None:
            group_names = self.groups
        else:
            # filter group names
            group_names = [name for name in group_names if name in self.groups]
        assert len(group_names) == len(self.groups), "Passed groups and fitted do not match"

        if len(self.groups) == 0:
            params = np.array([self.norm_["center"], self.norm_["scale"]])
        elif self.scale_by_group:
            norm = np.array([1.0, 1.0])
            for group, group_name in zip(groups, group_names):
                try:
                    norm = norm * self.norm_[group_name].loc[group].to_numpy()
                except KeyError:
                    norm = norm * np.asarray([self.missing_[group_name][name] for name in self.names])
            norm = np.power(norm, 1.0 / len(self.groups))
            params = norm
        else:
            try:
                params = self.norm_.loc[groups].to_numpy()
            except (KeyError, TypeError):
                params = np.asarray([self.missing_[name] for name in self.names])
        return params

    def get_norm(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get scaling parameters for multiple groups.

        Args:
            X (pd.DataFrame): dataframe with ``groups`` columns

        Returns:
            pd.DataFrame: dataframe with scaling parameterswhere each row corresponds to the input dataframe
        """
        if len(self.groups) == 0:
            norm = np.asarray([self.norm_["center"], self.norm_["scale"]]).reshape(1, -1)
        elif self.scale_by_group:
            norm = [
                np.prod(
                    [
                        X[group_name]
                        .map(self.norm_[group_name][name])
                        .fillna(self.missing_[group_name][name])
                        .to_numpy()
                        for group_name in self.groups
                    ],
                    axis=0,
                )
                for name in self.names
            ]
            norm = np.power(np.stack(norm, axis=1), 1.0 / len(self.groups))
        else:
            norm = X[self.groups].set_index(self.groups).join(self.norm_).fillna(self.missing_).to_numpy()
        return norm