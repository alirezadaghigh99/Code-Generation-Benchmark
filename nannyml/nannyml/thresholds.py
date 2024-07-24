class StandardDeviationThreshold(Threshold, type="standard_deviation"):
    """A Thresholder that offsets the mean of an array by a multiple of the standard deviation of the array values.

    This thresholder will take the aggregate of an array of values, the mean by default and add or subtract an offset
    to get the upper and lower threshold values.
    This offset is calculated as a multiplier, by default 3, times the standard deviation of the given array.

    Attributes:
        std_lower_multiplier: float
        std_upper_multiplier: float

    Examples:
        >>> data = np.array(range(10))
        >>> t = ConstantThreshold(lower=None, upper=0.1)
        >>> lower, upper = t.threshold()
        >>> print(lower, upper)
        -4.116843969807043 13.116843969807043
    """

    def __init__(
        self,
        std_lower_multiplier: Optional[Union[float, int]] = 3,
        std_upper_multiplier: Optional[Union[float, int]] = 3,
        offset_from: Callable[[np.ndarray], Any] = np.nanmean,
    ):
        """Creates a new StandardDeviationThreshold instance.

        Args:
            std_lower_multiplier: float, default=3
                The number the standard deviation of the input array will be multiplied with to form the lower offset.
                This value will be subtracted from the aggregate of the input array.
                Defaults to 3.
            std_upper_multiplier: float, default=3
                The number the standard deviation of the input array will be multiplied with to form the upper offset.
                This value will be added to the aggregate of the input array.
                Defaults to 3.
            offset_from: Callable[[np.ndarray], Any], default=np.nanmean
                A function that will be applied to the input array to aggregate it into a single value.
                Adding the upper offset to this value will yield the upper threshold, subtracting the lower offset
                will yield the lower threshold.
        """

        self._validate_inputs(std_lower_multiplier, std_upper_multiplier)

        self.std_lower_multiplier = std_lower_multiplier
        self.std_upper_multiplier = std_upper_multiplier
        self.offset_from = offset_from

    def thresholds(self, data: np.ndarray, **kwargs) -> Tuple[Optional[float], Optional[float]]:
        aggregate = self.offset_from(data)
        std = np.nanstd(data)

        lower_threshold = aggregate - std * self.std_lower_multiplier if self.std_lower_multiplier is not None else None

        upper_threshold = aggregate + std * self.std_upper_multiplier if self.std_upper_multiplier is not None else None

        return lower_threshold, upper_threshold

    @staticmethod
    def _validate_inputs(
        std_lower_multiplier: Optional[Union[float, int]] = 3, std_upper_multiplier: Optional[Union[float, int]] = 3
    ):
        if (
            std_lower_multiplier is not None
            and not isinstance(std_lower_multiplier, (float, int))
            or isinstance(std_lower_multiplier, bool)
        ):
            raise InvalidArgumentsException(
                f"expected type of 'std_lower_multiplier' to be 'float', 'int' or None "
                f"but got '{type(std_lower_multiplier).__name__}'"
            )

        if std_lower_multiplier and std_lower_multiplier < 0:
            raise ThresholdException(
                f"'std_lower_multiplier' should be greater than 0 " f"but got value {std_lower_multiplier}"
            )

        if (
            std_upper_multiplier is not None
            and not isinstance(std_upper_multiplier, (float, int))
            or isinstance(std_upper_multiplier, bool)
        ):
            raise InvalidArgumentsException(
                f"expected type of 'std_upper_multiplier' to be 'float', 'int' or None "
                f"but got '{type(std_upper_multiplier).__name__}'"
            )

        if std_upper_multiplier and std_upper_multiplier < 0:
            raise ThresholdException(
                f"'std_upper_multiplier' should be greater than 0 " f"but got value {std_upper_multiplier}"
            )

class ConstantThreshold(Threshold, type="constant"):
    """A `Thresholder` implementation that returns a constant lower and or upper threshold value.

    Attributes:
        lower: Optional[float]
            The constant lower threshold value. Defaults to `None`, meaning there is no lower threshold.
        upper: Optional[float]
            The constant upper threshold value. Defaults to `None`, meaning there is no upper threshold.

    Raises:
        InvalidArgumentsException: raised when an argument was given using an incorrect type or name
        ThresholdException: raised when the ConstantThreshold could not be created using the given argument values

    Examples:
        >>> data = np.array(range(10))
        >>> t = ConstantThreshold(lower=None, upper=0.1)
        >>> lower, upper = t.threshold()
        >>> print(lower, upper)
        None 0.1
    """

    def __init__(self, lower: Optional[Union[float, int]] = None, upper: Optional[Union[float, int]] = None):
        """Creates a new ConstantThreshold instance.

        Args:
            lower: Optional[Union[float, int]], default=None
                The constant lower threshold value. Defaults to `None`, meaning there is no lower threshold.
            upper: Optional[Union[float, int]], default=None
                The constant upper threshold value. Defaults to `None`, meaning there is no upper threshold.

        Raises:
            InvalidArgumentsException: raised when an argument was given using an incorrect type or name
            ThresholdException: raised when the ConstantThreshold could not be created using the given argument values
        """
        self._validate_inputs(lower, upper)

        self.lower = lower
        self.upper = upper

    def thresholds(self, data: np.ndarray, **kwargs) -> Tuple[Optional[float], Optional[float]]:
        return self.lower, self.upper

    @staticmethod
    def _validate_inputs(lower: Optional[Union[float, int]] = None, upper: Optional[Union[float, int]] = None):
        if lower is not None and not isinstance(lower, (float, int)) or isinstance(lower, bool):
            raise InvalidArgumentsException(
                f"expected type of 'lower' to be 'float', 'int' or None " f"but got '{type(lower).__name__}'"
            )

        if upper is not None and not isinstance(upper, (float, int)) or isinstance(upper, bool):
            raise InvalidArgumentsException(
                f"expected type of 'upper' to be 'float', 'int' or None " f"but got '{type(upper).__name__}'"
            )

        # explicit None check is required due to special interpretation of the value 0.0 as False
        if lower is not None and upper is not None and lower >= upper:
            raise ThresholdException(f"lower threshold {lower} must be less than upper threshold {upper}")

