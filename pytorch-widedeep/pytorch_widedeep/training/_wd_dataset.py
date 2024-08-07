class WideDeepDataset(Dataset):
    r"""
    Defines the Dataset object to load WideDeep data to the model

    Parameters
    ----------
    X_wide: np.ndarray
        wide input
    X_tab: np.ndarray
        deeptabular input
    X_text: np.ndarray or List[np.ndarray]
        deeptext input
    X_img: np.ndarray or List[np.ndarray]
        deepimage input
    target: np.ndarray
        target array
    transforms: Optional[Transforms | Compose]
        torchvision Compose object. See models/_multiple_transforms.py
    with_lds: bool
        Boolean indicating if Label Distribution Smoothing will be applied to
        the dataset

    Other Parameters
    ----------------
    **kwargs
        Label Distribution Smoothing parameters:
            lds_kernel: Literal['gaussian', 'triang', 'laplace'] = 'gaussian'
                choice of kernel for Label Distribution Smoothing
            lds_ks: int = 5
                LDS kernel window size
            lds_sigma: float = 2
                standard deviation of ['gaussian','laplace'] kernel for LDS
            lds_granularity: int = 100,
                number of bins in the histogram used in LDS to count occurence of sample values
            lds_reweight: bool
                option to reweight bin frequency counts in LDS
            lds_y_max: Optional[float] = None
                option to restrict LDS bins by upper label limit
            lds_y_min: Optional[float] = None
                option to restrict LDS bins by lower label limit
    """

    def __init__(
        self,
        X_wide: Optional[np.ndarray] = None,
        X_tab: Optional[np.ndarray] = None,
        X_text: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        X_img: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        target: Optional[np.ndarray] = None,
        transforms: Optional[Union[Transforms, Compose]] = None,
        is_training: bool = True,
        with_lds: bool = False,
        **kwargs,
    ):
        super(WideDeepDataset, self).__init__()
        self.X_wide = X_wide
        self.X_tab = X_tab
        self.X_text = X_text
        self.X_img = X_img
        self.transforms = transforms
        if self.transforms:
            if isinstance(self.transforms, Compose):
                self.transforms_names = [
                    tr.__class__.__name__ for tr in self.transforms.transforms
                ]
            else:
                self.transforms_names = [self.transforms.__class__.__name__]
        else:
            self.transforms_names = []
        self.Y = target

        # LDS
        self.is_training = is_training
        self.with_lds = with_lds
        if self.Y is not None and self.is_training:
            # this is a hack to avoid having to run separate for loops during
            # training whether we use lds or not
            if self.with_lds:
                self.weights = self._compute_lds_weights(**kwargs)
            else:
                self.weights = np.zeros_like(self.Y, dtype="float32")

    def __getitem__(self, idx: int):  # noqa: C901
        x = Bunch()
        if self.X_wide is not None:
            x.wide = self.X_wide[idx]
        if self.X_tab is not None:
            x.deeptabular = self.X_tab[idx]
        if self.X_text is not None:
            if isinstance(self.X_text, list):
                x.deeptext = [self.X_text[i][idx] for i in range(len(self.X_text))]
            else:
                x.deeptext = self.X_text[idx]
        if self.X_img is not None:
            if isinstance(self.X_img, list):
                x.deepimage = [
                    self._prepare_images(self.X_img[i], idx)
                    for i in range(len(self.X_img))
                ]
            else:
                x.deepimage = self._prepare_images(self.X_img, idx)
        if self.Y is None:
            return x
        else:
            y = self.Y[idx]
            if self.is_training:
                return x, y, self.weights[idx]
            else:
                return x, y

    def _compute_lds_weights(
        self,
        lds_granularity: int = 100,
        lds_reweight: bool = False,
        lds_kernel: Literal["gaussian", "triang", "laplace"] = "gaussian",
        lds_ks: int = 5,
        lds_sigma: float = 2,
        lds_y_min: Optional[float] = None,
        lds_y_max: Optional[float] = None,
    ) -> np.ndarray:
        """Assign weight to each sample by following procedure:
        1.      creating histogram from label values with nuber of bins = granularity
        2[opt]. reweighting label frequencies by sqrt
        3[opt]. smoothing label frequencies by convolution of kernel function window with frequencies list
        4.      inverting values by n_samples / (n_classes * np.bincount(y)), see:
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html
        5.      assigning weight to each sample from closest bin value
        """

        assert self.Y is not None, "No target array provided"
        y_max = max(self.Y) if lds_y_max is None else lds_y_max
        y_min = min(self.Y) if lds_y_min is None else lds_y_min
        bin_edges = np.linspace(y_min, y_max, num=lds_granularity, endpoint=True)
        value_dict = dict(zip(bin_edges[:-1], np.histogram(self.Y, bin_edges)[0]))

        if lds_reweight:
            value_dict = dict(
                zip(value_dict.keys(), np.sqrt(list(value_dict.values())))
            )

        if lds_kernel is not None:
            lds_kernel_window = get_kernel_window(lds_kernel, lds_ks, lds_sigma)
            smoothed_values = convolve1d(
                list(value_dict.values()), weights=lds_kernel_window, mode="constant"
            )

            # to avoid 0 division
            numerator = sum(smoothed_values)
            denominator = (len(smoothed_values) * smoothed_values).astype("float")
            denominator[denominator == 0] = np.inf

            weigths = numerator / denominator
            # weigths = sum(smoothed_values) / (len(smoothed_values) * smoothed_values)
        else:
            values = list(value_dict.values())
            weigths = sum(values) / (len(values) * values)  # type: ignore[operator]
        value_dict = dict(zip(value_dict.keys(), weigths))

        left_bin_edges = find_bin(bin_edges, self.Y)
        weights = np.array(
            [value_dict[edge] for edge in left_bin_edges], dtype="float32"
        )

        return weights

    def _prepare_images(self, imgs: np.ndarray, idx: int):
        # if an image dataset is used, make sure is in the right format to
        # be ingested by the conv layers
        xdi = imgs[idx]
        # if int must be uint8
        if "int" in str(xdi.dtype) and "uint8" != str(xdi.dtype):
            xdi = xdi.astype("uint8")
        # if float must be float32
        if "float" in str(xdi.dtype) and "float32" != str(xdi.dtype):
            xdi = xdi.astype("float32")
        # if there are no transforms, or these do not include ToTensor(),
        # then we need to  replicate what Tensor() does -> transpose axis
        # and normalize if necessary
        if not self.transforms or "ToTensor" not in self.transforms_names:
            if xdi.ndim == 2:
                xdi = xdi[:, :, None]
            xdi = xdi.transpose(2, 0, 1)
            if "int" in str(xdi.dtype):
                xdi = (xdi / xdi.max()).astype("float32")
        # if ToTensor() is included, simply apply transforms
        if "ToTensor" in self.transforms_names:
            xdi = self.transforms(xdi)
        # else apply transforms on the result of calling torch.tensor on
        # xdi after all the previous manipulation
        elif self.transforms:
            xdi = self.transforms(torch.tensor(xdi))
        return xdi

    def __len__(self):
        if self.X_wide is not None:
            return len(self.X_wide)
        if self.X_tab is not None:
            return len(self.X_tab)
        if self.X_text is not None:
            if isinstance(self.X_text, list):
                return len(self.X_text[0])
            else:
                return len(self.X_text)
        if self.X_img is not None:
            if isinstance(self.X_img, list):
                return len(self.X_img[0])
            else:
                return len(self.X_img)

