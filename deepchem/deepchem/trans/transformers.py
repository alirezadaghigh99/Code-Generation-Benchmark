class DataTransforms(object):
    """Applies different data transforms to images.

    This utility class facilitates various image transformations that may be of
    use for handling image datasets.

    Note
    ----
    This class requires PIL to be installed.
    """

    def __init__(self, Image):
        self.Image = Image

    def scale(self, h, w):
        """Scales the image

        Parameters
        ----------
        h: int
            Height of the images
        w: int
            Width of the images

        Returns
        -------
        np.ndarray
            The scaled image.
        """
        from PIL import Image
        return Image.fromarray(self.Image).resize((h, w))

    def flip(self, direction="lr"):
        """Flips the image

        Parameters
        ----------
        direction: str
            "lr" denotes left-right flip and "ud" denotes up-down flip.

        Returns
        -------
        np.ndarray
            The flipped image.
        """
        if direction == "lr":
            return np.fliplr(self.Image)
        elif direction == "ud":
            return np.flipud(self.Image)
        else:
            raise ValueError(
                "Invalid flip command : Enter either lr (for left to right flip) or ud (for up to down flip)"
            )

    def rotate(self, angle=0):
        """Rotates the image

        Parameters
        ----------
        angle: float (default = 0 i.e no rotation)
            Denotes angle by which the image should be rotated (in Degrees)

        Returns
        -------
        np.ndarray
            The rotated image.
        """
        return scipy.ndimage.rotate(self.Image, angle)

    def gaussian_blur(self, sigma=0.2):
        """Adds gaussian noise to the image

        Parameters
        ----------
        sigma: float
            Std dev. of the gaussian distribution

        Returns
        -------
        np.ndarray
            The image added gaussian noise.
        """
        return scipy.ndimage.gaussian_filter(self.Image, sigma)

    def center_crop(self, x_crop, y_crop):
        """Crops the image from the center

        Parameters
        ----------
        x_crop: int
            the total number of pixels to remove in the horizontal direction, evenly split between the left and right sides
        y_crop: int
            the total number of pixels to remove in the vertical direction, evenly split between the top and bottom sides

        Returns
        -------
        np.ndarray
            The center cropped image.
        """
        y = self.Image.shape[0]
        x = self.Image.shape[1]
        x_start = x // 2 - (x_crop // 2)
        y_start = y // 2 - (y_crop // 2)
        return self.Image[y_start:y_start + y_crop, x_start:x_start + x_crop]

    def crop(self, left, top, right, bottom):
        """Crops the image and returns the specified rectangular region from an image

        Parameters
        ----------
        left: int
            the number of pixels to exclude from the left of the image
        top: int
            the number of pixels to exclude from the top of the image
        right: int
            the number of pixels to exclude from the right of the image
        bottom: int
            the number of pixels to exclude from the bottom of the image

        Returns
        -------
        np.ndarray
            The cropped image.
        """
        y = self.Image.shape[0]
        x = self.Image.shape[1]
        return self.Image[top:y - bottom, left:x - right]

    def convert2gray(self):
        """Converts the image to grayscale. The coefficients correspond to the Y' component of the Y'UV color system.

        Returns
        -------
        np.ndarray
            The grayscale image.
        """
        return np.dot(self.Image[..., :3], [0.2989, 0.5870, 0.1140])

    def shift(self, width, height, mode='constant', order=3):
        """Shifts the image

        Parameters
        ----------
        width: float
            Amount of width shift (positive values shift image right )
        height: float
            Amount of height shift(positive values shift image lower)
        mode: str
            Points outside the boundaries of the input are filled according to the
            given mode: (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default is
            ‘constant’
        order: int
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

        Returns
        -------
        np.ndarray
            The shifted image.
        """
        if len(self.Image.shape) == 2:
            return scipy.ndimage.shift(self.Image, [height, width],
                                       order=order,
                                       mode=mode)
        if len(self.Image.shape == 3):
            return scipy.ndimage.shift(self.Image, [height, width, 0],
                                       order=order,
                                       mode=mode)

    def gaussian_noise(self, mean=0, std=25.5):
        """Adds gaussian noise to the image

        Parameters
        ----------
        mean: float
            Mean of gaussian.
        std: float
            Standard deviation of gaussian.

        Returns
        -------
        np.ndarray
            The image added gaussian noise.
        """

        x = self.Image
        x = x + np.random.normal(loc=mean, scale=std, size=self.Image.shape)
        return x

    def salt_pepper_noise(self, prob=0.05, salt=255, pepper=0):
        """Adds salt and pepper noise to the image

        Parameters
        ----------
        prob: float
            probability of the noise.
        salt: float
            value of salt noise.
        pepper: float
            value of pepper noise.

        Returns
        -------
        np.ndarray
            The image added salt and pepper noise.
        """

        noise = np.random.random(size=self.Image.shape)
        x = self.Image
        x[noise < (prob / 2)] = pepper
        x[noise > (1 - prob / 2)] = salt
        return x

    def median_filter(self, size):
        """ Calculates a multidimensional median filter

        Parameters
        ----------
        size: int
            The kernel size in pixels.

        Returns
        -------
        np.ndarray
            The median filtered image.
        """
        from PIL import Image, ImageFilter
        image = Image.fromarray(self.Image)
        image = image.filter(ImageFilter.MedianFilter(size=size))
        return np.array(image)

class RxnSplitTransformer(Transformer):
    """Splits the reaction SMILES input into the source and target strings
    required for machine translation tasks.

    The input is expected to be in the form reactant>reagent>product. The source
    string would be reactants>reagents and the target string would be the products.

    The transformer can also separate the reagents from the reactants for a mixed
    training mode. During mixed training, the source string is transformed from
    reactants>reagent to reactants.reagent> . This can be toggled (default True)
    by setting the value of sep_reagent while calling the transformer.

    Examples
    --------
    >>> # When mixed training is toggled.
    >>> import numpy as np
    >>> from deepchem.trans.transformers import RxnSplitTransformer
    >>> reactions = np.array(["CC(C)C[Mg+].CON(C)C(=O)c1ccc(O)nc1>C1CCOC1.[Cl-]>CC(C)CC(=O)c1ccc(O)nc1","CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(N)cc3)cc21.O=CO>>CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(NC=O)cc3)cc21"], dtype=object)
    >>> trans = RxnSplitTransformer(sep_reagent=True)
    >>> split_reactions = trans.transform_array(X=reactions, y=np.array([]), w=np.array([]), ids=np.array([]))
    >>> split_reactions
    (array([['CC(C)C[Mg+].CON(C)C(=O)c1ccc(O)nc1>C1CCOC1.[Cl-]',
            'CC(C)CC(=O)c1ccc(O)nc1'],
           ['CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(N)cc3)cc21.O=CO>',
            'CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(NC=O)cc3)cc21']], dtype='<U51'), array([], dtype=float64), array([], dtype=float64), array([], dtype=float64))

    When mixed training is disabled, you get the following outputs:

    >>> trans_disable = RxnSplitTransformer(sep_reagent=False)
    >>> split_reactions = trans_disable.transform_array(X=reactions, y=np.array([]), w=np.array([]), ids=np.array([]))
    >>> split_reactions
    (array([['CC(C)C[Mg+].CON(C)C(=O)c1ccc(O)nc1.C1CCOC1.[Cl-]>',
            'CC(C)CC(=O)c1ccc(O)nc1'],
           ['CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(N)cc3)cc21.O=CO>',
            'CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(NC=O)cc3)cc21']], dtype='<U51'), array([], dtype=float64), array([], dtype=float64), array([], dtype=float64))

    Note
    ----
    This class only transforms the feature field of a reaction dataset like USPTO.
    """

    def __init__(self,
                 sep_reagent: bool = True,
                 dataset: Optional[Dataset] = None):
        """Initializes the Reaction split Transformer.

        Parameters
        ----------
        sep_reagent: bool, optional (default True)
            To separate the reagent and reactants for training.
        dataset: dc.data.Dataset object, optional (default None)
            Dataset to be transformed.
        """

        self.sep_reagent = sep_reagent
        super(RxnSplitTransformer, self).__init__(transform_X=True,
                                                  dataset=dataset)

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w, ids) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features(the reactions)
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of weights.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """

        reactant = list(map(lambda x: x.split('>')[0], X))
        reagent = list(map(lambda x: x.split('>')[1], X))
        product = list(map(lambda x: x.split('>')[2], X))

        if self.sep_reagent:
            source = [x + '>' + y for x, y in zip(reactant, reagent)]
        else:
            source = [
                x + '.' + y + '>' if y else x + '>' + y
                for x, y in zip(reactant, reagent)
            ]

        target = product

        X = np.column_stack((source, target))

        return (X, y, w, ids)

    def untransform(self, z):
        """Not Implemented."""
        raise NotImplementedError("Cannot untransform the source/target split.")

class Transformer(object):
    """Abstract base class for different data transformation techniques.

    A transformer is an object that applies a transformation to a given
    dataset. Think of a transformation as a mathematical operation which
    makes the source dataset more amenable to learning. For example, one
    transformer could normalize the features for a dataset (ensuring
    they have zero mean and unit standard deviation). Another
    transformer could for example threshold values in a dataset so that
    values outside a given range are truncated. Yet another transformer
    could act as a data augmentation routine, generating multiple
    different images from each source datapoint (a transformation need
    not necessarily be one to one).

    Transformers are designed to be chained, since data pipelines often
    chain multiple different transformations to a dataset. Transformers
    are also designed to be scalable and can be applied to
    large `dc.data.Dataset` objects. Not that Transformers are not
    usually thread-safe so you will have to be careful in processing
    very large datasets.

    This class is an abstract superclass that isn't meant to be directly
    instantiated. Instead, you will want to instantiate one of the
    subclasses of this class inorder to perform concrete
    transformations.
    """
    # Hack to allow for easy unpickling:
    # http://stefaanlippens.net/pickleproblem
    __module__ = os.path.splitext(os.path.basename(__file__))[0]

    def __init__(self,
                 transform_X: bool = False,
                 transform_y: bool = False,
                 transform_w: bool = False,
                 transform_ids: bool = False,
                 dataset: Optional[Dataset] = None):
        """Initializes transformation based on dataset statistics.

        Parameters
        ----------
        transform_X: bool, optional (default False)
            Whether to transform X
        transform_y: bool, optional (default False)
            Whether to transform y
        transform_w: bool, optional (default False)
            Whether to transform w
        transform_ids: bool, optional (default False)
            Whether to transform ids
        dataset: dc.data.Dataset object, optional (default None)
            Dataset to be transformed
        """
        if self.__class__.__name__ == "Transformer":
            raise ValueError(
                "Transformer is an abstract superclass and cannot be directly instantiated. You probably want to instantiate a concrete subclass instead."
            )
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.transform_w = transform_w
        self.transform_ids = transform_ids
        # Some transformation must happen
        assert transform_X or transform_y or transform_w or transform_ids

    def transform_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transform the data in a set of (X, y, w, ids) arrays.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of identifiers.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        raise NotImplementedError(
            "Each Transformer is responsible for its own transform_array method."
        )

    def untransform(self, transformed: np.ndarray) -> np.ndarray:
        """Reverses stored transformation on provided data.

        Depending on whether `transform_X` or `transform_y` or `transform_w` was
        set, this will perform different un-transformations. Note that this method
        may not always be defined since some transformations aren't 1-1.

        Parameters
        ----------
        transformed: np.ndarray
            Array which was previously transformed by this class.
        """
        raise NotImplementedError(
            "Each Transformer is responsible for its own untransform method.")

    def transform(self,
                  dataset: Dataset,
                  parallel: bool = False,
                  out_dir: Optional[str] = None,
                  **kwargs) -> Dataset:
        """Transforms all internally stored data in dataset.

        This method transforms all internal data in the provided dataset by using
        the `Dataset.transform` method. Note that this method adds X-transform,
        y-transform columns to metadata. Specified keyword arguments are passed on
        to `Dataset.transform`.

        Parameters
        ----------
        dataset: dc.data.Dataset
            Dataset object to be transformed.
        parallel: bool, optional (default False)
            if True, use multiple processes to transform the dataset in parallel.
            For large datasets, this might be faster.
        out_dir: str, optional
            If `out_dir` is specified in `kwargs` and `dataset` is a `DiskDataset`,
            the output dataset will be written to the specified directory.

        Returns
        -------
        Dataset
            A newly transformed Dataset object
        """
        # Add this case in to handle non-DiskDataset that should be written to disk
        if out_dir is not None:
            if not isinstance(dataset, dc.data.DiskDataset):
                dataset = dc.data.DiskDataset.from_numpy(
                    dataset.X, dataset.y, dataset.w, dataset.ids)
        _, y_shape, w_shape, _ = dataset.get_shape()
        if y_shape == tuple() and self.transform_y:
            raise ValueError("Cannot transform y when y_values are not present")
        if w_shape == tuple() and self.transform_w:
            raise ValueError("Cannot transform w when w_values are not present")
        return dataset.transform(self, out_dir=out_dir, parallel=parallel)

    def transform_on_array(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Transforms numpy arrays X, y, and w

        DEPRECATED. Use `transform_array` instead.

        Parameters
        ----------
        X: np.ndarray
            Array of features
        y: np.ndarray
            Array of labels
        w: np.ndarray
            Array of weights.
        ids: np.ndarray
            Array of identifiers.

        Returns
        -------
        Xtrans: np.ndarray
            Transformed array of features
        ytrans: np.ndarray
            Transformed array of labels
        wtrans: np.ndarray
            Transformed array of weights
        idstrans: np.ndarray
            Transformed array of ids
        """
        warnings.warn(
            "transform_on_array() is deprecated and has been renamed to transform_array()."
            "transform_on_array() will be removed in DeepChem 3.0",
            FutureWarning)
        X, y, w, ids = self.transform_array(X, y, w, ids)
        return X, y, w, ids

