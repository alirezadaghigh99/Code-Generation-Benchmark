class Data:
    """
    Class that holds and validates datasets for Datalab.

    Internally, the data is stored as a datasets.Dataset object and the labels
    are integers (ranging from 0 to K-1, where K is the number of classes) stored
    in a numpy array.

    Parameters
    ----------
    data :
        Dataset to be audited by Datalab.
        Several formats are supported, which will internally be converted to a Dataset object.

        Supported formats:
            - datasets.Dataset
            - pandas.DataFrame
            - dict
                - keys are strings
                - values are arrays or lists of equal length
            - list
                - list of dictionaries with the same keys
            - str
                - path to a local file
                    - Text (.txt)
                    - CSV (.csv)
                    - JSON (.json)
                - or a dataset identifier on the Hugging Face Hub
            It checks if the string is a path to a file that exists locally, and if not,
            it assumes it is a dataset identifier on the Hugging Face Hub.

    label_name : Union[str, List[str]]
        Name of the label column in the dataset.

    task :
        The task associated with the dataset. This is used to determine how to
        to format the labels.

        Note:

          - If the task is a classification task, the labels
          will be mapped to integers, e.g. [0, 1, ..., K-1] where K is the number
          of classes. If the task is a regression task, the labels will not be
          mapped to integers.

          - If the task is a multilabel task, the labels will be formatted as a
            list of lists, e.g. [[0, 1], [1, 2], [0, 2]] where each sublist contains
            the labels for a single example. If the task is not a multilabel task,
            the labels will be formatted as a 1D numpy array.

    Warnings
    --------
    Optional dependencies:

    - datasets :
        Dataset, DatasetDict and load_dataset are imported from datasets.
        This is an optional dependency of cleanlab, but is required for
        :py:class:`Datalab <cleanlab.datalab.datalab.Datalab>` to work.
    """

    def __init__(
        self,
        data: "DatasetLike",
        task: Task,
        label_name: Optional[str] = None,
    ) -> None:
        self._validate_data(data)
        self._data = self._load_data(data)
        self._data_hash = hash(self._data)
        self.labels: Label
        label_class = MultiLabel if task.is_multilabel else MultiClass
        map_to_int = task.is_classification
        self.labels = label_class(data=self._data, label_name=label_name, map_to_int=map_to_int)

    def _load_data(self, data: "DatasetLike") -> Dataset:
        """Checks the type of dataset and uses the correct loader method and
        assigns the result to the data attribute."""
        dataset_factory_map: Dict[type, Callable[..., Dataset]] = {
            Dataset: lambda x: x,
            pd.DataFrame: Dataset.from_pandas,
            dict: self._load_dataset_from_dict,
            list: self._load_dataset_from_list,
            str: self._load_dataset_from_string,
        }
        if not isinstance(data, tuple(dataset_factory_map.keys())):
            raise DataFormatError(data)
        return dataset_factory_map[type(data)](data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        if isinstance(other, Data):
            # Equality checks
            hashes_are_equal = self._data_hash == other._data_hash
            labels_are_equal = self.labels == other.labels
            return all([hashes_are_equal, labels_are_equal])
        return False

    def __hash__(self) -> int:
        return self._data_hash

    @property
    def class_names(self) -> List[str]:
        return self.labels.class_names

    @property
    def has_labels(self) -> bool:
        """Check if labels are available."""
        return self.labels.is_available

    @staticmethod
    def _validate_data(data) -> None:
        if isinstance(data, datasets.DatasetDict):
            raise DatasetDictError()
        if not isinstance(data, (Dataset, pd.DataFrame, dict, list, str)):
            raise DataFormatError(data)

    @staticmethod
    def _load_dataset_from_dict(data_dict: Dict[str, Any]) -> Dataset:
        try:
            return Dataset.from_dict(data_dict)
        except Exception as error:
            raise DatasetLoadError(dict) from error

    @staticmethod
    def _load_dataset_from_list(data_list: List[Dict[str, Any]]) -> Dataset:
        try:
            return Dataset.from_list(data_list)
        except Exception as error:
            raise DatasetLoadError(list) from error

    @staticmethod
    def _load_dataset_from_string(data_string: str) -> Dataset:
        if not os.path.exists(data_string):
            try:
                dataset = datasets.load_dataset(data_string)
                return cast(Dataset, dataset)
            except Exception as error:
                raise DatasetLoadError(str) from error

        factory: Dict[str, Callable[[str], Any]] = {
            ".txt": Dataset.from_text,
            ".csv": Dataset.from_csv,
            ".json": Dataset.from_json,
        }

        extension = os.path.splitext(data_string)[1]
        if extension not in factory:
            raise DatasetLoadError(type(data_string))

        dataset = factory[extension](data_string)
        dataset_cast = cast(Dataset, dataset)
        return dataset_cast

