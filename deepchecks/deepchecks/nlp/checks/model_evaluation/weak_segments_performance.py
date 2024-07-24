class MetadataSegmentsPerformance(WeakSegmentsAbstractText):
    """Search for segments with low performance scores.

    The check is designed to help you easily identify weak spots of your model and provide a deepdive analysis into
    its performance on different segments of your data. Specifically, it is designed to help you identify the model
    weakest segments in the data distribution for further improvement and visibility purposes.

    The segments are based on the metadata - which is data that is not part of the text, but is related to it,
    such as "user_id" and "user_age". For more on metadata, see the `NLP Metadata Guide
    <https://docs.deepchecks.com/stable/nlp/usage_guides/nlp_metadata.html>`_.

    In order to achieve this, the check trains several simple tree based models which try to predict the error of the
    user provided model on the dataset. The relevant segments are detected by analyzing the different
    leafs of the trained trees.

    Parameters
    ----------
    columns : Union[Hashable, List[Hashable]] , default: None
        Columns to check, if none are given checks all columns except ignored ones.
    ignore_columns : Union[Hashable, List[Hashable]] , default: None
        Columns to ignore, if none given checks based on columns variable
    n_top_columns : Optional[int] , default: 10
        Number of columns to use for segment search. Selected at random.
    segment_minimum_size_ratio: float , default: 0.05
        Minimum size ratio for segments. Will only search for segments of
        size >= segment_minimum_size_ratio * data_size.
    max_categories_weak_segment: Optional[int] , default: None
        Maximum number of categories that can be included in a weak segment per categorical feature.
        If None, the number of categories is not limited.
    alternative_scorer : Dict[str, Union[str, Callable]] , default: None
        Scorer to use as performance measure, either function or sklearn scorer name.
        If None, a default scorer (per the model type) will be used.
    score_per_sample: Union[np.array, pd.Series, None], default: None
        Score per sample are required to detect relevant weak segments. Should follow the convention that a sample with
        a higher score mean better model performance on that sample. If provided, the check will also use provided
        score per sample as a scoring function for segments.
        if None the check calculates score per sample by via neg cross entropy for classification.
    n_samples : int , default: 5_000
        Maximum number of samples to use for this check.
    n_to_show : int , default: 3
        number of segments with the weakest performance to show.
    categorical_aggregation_threshold : float , default: 0.05
        In each categorical column, categories with frequency below threshold will be merged into "Other" category.
    multiple_segments_column : bool , default: True
        If True, will allow the same metadata column to be a segmenting column in multiple segments,
        otherwise each metadata column can appear in one segment at most.
    """

    def __init__(self,
                 columns: Union[Hashable, List[Hashable], None] = None,
                 ignore_columns: Union[Hashable, List[Hashable], None] = None,
                 n_top_columns: Optional[int] = 10,
                 segment_minimum_size_ratio: float = 0.05,
                 max_categories_weak_segment: Optional[int] = None,
                 alternative_scorer: Dict[str, Union[str, Callable]] = None,
                 score_per_sample: Union[np.ndarray, pd.Series, None] = None,
                 n_samples: int = 5_000,
                 categorical_aggregation_threshold: float = 0.05,
                 n_to_show: int = 3,
                 multiple_segments_column: bool = True,
                 **kwargs):
        super().__init__(segment_by='metadata',
                         columns=columns,
                         ignore_columns=ignore_columns,
                         n_top_features=n_top_columns,
                         segment_minimum_size_ratio=segment_minimum_size_ratio,
                         max_categories_weak_segment=max_categories_weak_segment,
                         n_samples=n_samples,
                         n_to_show=n_to_show,
                         score_per_sample=score_per_sample,
                         alternative_scorer=alternative_scorer,
                         categorical_aggregation_threshold=categorical_aggregation_threshold,
                         multiple_segments_per_feature=multiple_segments_column,
                         **kwargs)

class PropertySegmentsPerformance(WeakSegmentsAbstractText):
    """Search for segments with low performance scores.

    The check is designed to help you easily identify weak spots of your model and provide a deepdive analysis into
    its performance on different segments of your data. Specifically, it is designed to help you identify the model
    weakest segments in the data distribution for further improvement and visibility purposes.

    The segments are based on the text properties - which are features extracted from the text, such as "language" and
    "number of words". For more on properties, see the `NLP Properties Guide
    <https://docs.deepchecks.com/stable/nlp/usage_guides/nlp_properties.html>`_.

    In order to achieve this, the check trains several simple tree based models which try to predict the error of the
    user provided model on the dataset. The relevant segments are detected by analyzing the different
    leafs of the trained trees.

    Parameters
    ----------
    properties : Union[Hashable, List[Hashable]] , default: None
        Properties to check, if none are given checks all properties except ignored ones.
    ignore_properties : Union[Hashable, List[Hashable]] , default: None
        Properties to ignore, if none given checks based on properties variable
    n_top_properties : Optional[int] , default: 10
        Number of properties to use for segment search. Selected at random.
    segment_minimum_size_ratio: float , default: 0.05
        Minimum size ratio for segments. Will only search for segments of
        size >= segment_minimum_size_ratio * data_size.
    max_categories_weak_segment: Optional[int] , default: None
        Maximum number of categories that can be included in a weak segment per categorical feature.
        If None, the number of categories is not limited.
    alternative_scorer : Dict[str, Union[str, Callable]] , default: None
        Scorer to use as performance measure, either function or sklearn scorer name.
        If None, a default scorer (per the model type) will be used.
    score_per_sample: Optional[np.array, pd.Series, None], default: None
        Score per sample are required to detect relevant weak segments. Should follow the convention that a sample with
        a higher score mean better model performance on that sample. If provided, the check will also use provided
        score per sample as a scoring function for segments.
        if None the check calculates score per sample by via neg cross entropy for classification.
    n_samples : int , default: 5_000
        Maximum number of samples to use for this check.
    n_to_show : int , default: 3
        number of segments with the weakest performance to show.
    categorical_aggregation_threshold : float , default: 0.05
        In each categorical column, categories with frequency below threshold will be merged into "Other" category.
    multiple_segments_per_property : bool , default: False
        If True, will allow the same property to be a segmenting feature in multiple segments,
        otherwise each property can appear in one segment at most.
    """

    def __init__(self,
                 properties: Union[Hashable, List[Hashable], None] = None,
                 ignore_properties: Union[Hashable, List[Hashable], None] = None,
                 n_top_properties: Optional[int] = 10,
                 segment_minimum_size_ratio: float = 0.05,
                 max_categories_weak_segment: Optional[int] = None,
                 alternative_scorer: Dict[str, Union[str, Callable]] = None,
                 score_per_sample: Union[np.ndarray, pd.Series, None] = None,
                 n_samples: int = 5_000,
                 categorical_aggregation_threshold: float = 0.05,
                 n_to_show: int = 3,
                 multiple_segments_per_property: bool = False,
                 **kwargs):
        super().__init__(segment_by='properties',
                         columns=properties,
                         ignore_columns=ignore_properties,
                         n_top_features=n_top_properties,
                         segment_minimum_size_ratio=segment_minimum_size_ratio,
                         max_categories_weak_segment=max_categories_weak_segment,
                         n_samples=n_samples,
                         n_to_show=n_to_show,
                         score_per_sample=score_per_sample,
                         alternative_scorer=alternative_scorer,
                         categorical_aggregation_threshold=categorical_aggregation_threshold,
                         multiple_segments_per_feature=multiple_segments_per_property,
                         **kwargs)

