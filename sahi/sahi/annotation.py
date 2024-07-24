class Category:
    """
    Category of the annotation.
    """

    def __init__(self, id=None, name=None):
        """
        Args:
            id: int
                ID of the object category
            name: str
                Name of the object category
        """
        if not isinstance(id, int):
            raise TypeError("id should be integer")
        if not isinstance(name, str):
            raise TypeError("name should be string")
        self.id = id
        self.name = name

    def __repr__(self):
        return f"Category: <id: {self.id}, name: {self.name}>"

class Category:
    """
    Category of the annotation.
    """

    def __init__(self, id=None, name=None):
        """
        Args:
            id: int
                ID of the object category
            name: str
                Name of the object category
        """
        if not isinstance(id, int):
            raise TypeError("id should be integer")
        if not isinstance(name, str):
            raise TypeError("name should be string")
        self.id = id
        self.name = name

    def __repr__(self):
        return f"Category: <id: {self.id}, name: {self.name}>"

class Mask:
    @classmethod
    def from_float_mask(
        cls,
        mask,
        full_shape=None,
        mask_threshold: float = 0.5,
        shift_amount: list = [0, 0],
    ):
        """
        Args:
            mask: np.ndarray of np.float elements
                Mask values between 0 and 1 (should have a shape of height*width)
            mask_threshold: float
                Value to threshold mask pixels between 0 and 1
            shift_amount: List
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: List
                Size of the full image after shifting, should be in the form of [height, width]
        """
        bool_mask = mask > mask_threshold
        return cls(
            segmentation=get_coco_segmentation_from_bool_mask(bool_mask),
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    def __init__(
        self,
        segmentation,
        full_shape=None,
        shift_amount: list = [0, 0],
    ):
        """
        Init Mask from coco segmentation representation.

        Args:
            segmentation : List[List]
                [
                    [x1, y1, x2, y2, x3, y3, ...],
                    [x1, y1, x2, y2, x3, y3, ...],
                    ...
                ]
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        # confirm full_shape is given
        if full_shape is None:
            raise ValueError("full_shape must be provided")

        self.shift_x = shift_amount[0]
        self.shift_y = shift_amount[1]

        if full_shape:
            self.full_shape_height = full_shape[0]
            self.full_shape_width = full_shape[1]
        else:
            self.full_shape_height = None
            self.full_shape_width = None

        self.segmentation = segmentation

    @classmethod
    def from_bool_mask(
        cls,
        bool_mask=None,
        full_shape=None,
        shift_amount: list = [0, 0],
    ):
        """
        Args:
            bool_mask: np.ndarray with bool elements
                2D mask of object, should have a shape of height*width
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        return cls(
            segmentation=get_coco_segmentation_from_bool_mask(bool_mask),
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @property
    def bool_mask(self):
        return get_bool_mask_from_coco_segmentation(
            self.segmentation, width=self.full_shape[1], height=self.full_shape[0]
        )

    @property
    def shape(self):
        """
        Returns mask shape as [height, width]
        """
        return [self.bool_mask.shape[0], self.bool_mask.shape[1]]

    @property
    def full_shape(self):
        """
        Returns full mask shape after shifting as [height, width]
        """
        return [self.full_shape_height, self.full_shape_width]

    @property
    def shift_amount(self):
        """
        Returns the shift amount of the mask slice as [shift_x, shift_y]
        """
        return [self.shift_x, self.shift_y]

    def get_shifted_mask(self):
        # Confirm full_shape is specified
        if (self.full_shape_height is None) or (self.full_shape_width is None):
            raise ValueError("full_shape is None")
        shifted_segmentation = []
        for s in self.segmentation:
            xs = [min(self.shift_x + s[i], self.full_shape_width) for i in range(0, len(s) - 1, 2)]
            ys = [min(self.shift_y + s[i], self.full_shape_height) for i in range(1, len(s), 2)]
            shifted_segmentation.append([j for i in zip(xs, ys) for j in i])
        return Mask(
            segmentation=shifted_segmentation,
            shift_amount=[0, 0],
            full_shape=self.full_shape,
        )

class Mask:
    @classmethod
    def from_float_mask(
        cls,
        mask,
        full_shape=None,
        mask_threshold: float = 0.5,
        shift_amount: list = [0, 0],
    ):
        """
        Args:
            mask: np.ndarray of np.float elements
                Mask values between 0 and 1 (should have a shape of height*width)
            mask_threshold: float
                Value to threshold mask pixels between 0 and 1
            shift_amount: List
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: List
                Size of the full image after shifting, should be in the form of [height, width]
        """
        bool_mask = mask > mask_threshold
        return cls(
            segmentation=get_coco_segmentation_from_bool_mask(bool_mask),
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    def __init__(
        self,
        segmentation,
        full_shape=None,
        shift_amount: list = [0, 0],
    ):
        """
        Init Mask from coco segmentation representation.

        Args:
            segmentation : List[List]
                [
                    [x1, y1, x2, y2, x3, y3, ...],
                    [x1, y1, x2, y2, x3, y3, ...],
                    ...
                ]
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        # confirm full_shape is given
        if full_shape is None:
            raise ValueError("full_shape must be provided")

        self.shift_x = shift_amount[0]
        self.shift_y = shift_amount[1]

        if full_shape:
            self.full_shape_height = full_shape[0]
            self.full_shape_width = full_shape[1]
        else:
            self.full_shape_height = None
            self.full_shape_width = None

        self.segmentation = segmentation

    @classmethod
    def from_bool_mask(
        cls,
        bool_mask=None,
        full_shape=None,
        shift_amount: list = [0, 0],
    ):
        """
        Args:
            bool_mask: np.ndarray with bool elements
                2D mask of object, should have a shape of height*width
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        return cls(
            segmentation=get_coco_segmentation_from_bool_mask(bool_mask),
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @property
    def bool_mask(self):
        return get_bool_mask_from_coco_segmentation(
            self.segmentation, width=self.full_shape[1], height=self.full_shape[0]
        )

    @property
    def shape(self):
        """
        Returns mask shape as [height, width]
        """
        return [self.bool_mask.shape[0], self.bool_mask.shape[1]]

    @property
    def full_shape(self):
        """
        Returns full mask shape after shifting as [height, width]
        """
        return [self.full_shape_height, self.full_shape_width]

    @property
    def shift_amount(self):
        """
        Returns the shift amount of the mask slice as [shift_x, shift_y]
        """
        return [self.shift_x, self.shift_y]

    def get_shifted_mask(self):
        # Confirm full_shape is specified
        if (self.full_shape_height is None) or (self.full_shape_width is None):
            raise ValueError("full_shape is None")
        shifted_segmentation = []
        for s in self.segmentation:
            xs = [min(self.shift_x + s[i], self.full_shape_width) for i in range(0, len(s) - 1, 2)]
            ys = [min(self.shift_y + s[i], self.full_shape_height) for i in range(1, len(s), 2)]
            shifted_segmentation.append([j for i in zip(xs, ys) for j in i])
        return Mask(
            segmentation=shifted_segmentation,
            shift_amount=[0, 0],
            full_shape=self.full_shape,
        )

