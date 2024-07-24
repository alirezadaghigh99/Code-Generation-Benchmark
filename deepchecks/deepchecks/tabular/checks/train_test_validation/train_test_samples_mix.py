class TrainTestSamplesMix(TrainTestCheck, TrainTestSamplesMixAbstract):
    """Detect samples in the test data that appear also in training data.

    Parameters
    ----------
    n_samples : int , default: 10_000_000
        number of samples to use for this check.
    n_to_show : int , default: 10
        number of samples that appear in test and training data to show.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(
        self,
        n_samples: int = 10_000_000,
        n_to_show: int = 10,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.n_to_show = n_to_show
        self.random_state = random_state

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is sample leakage ratio in %,
            displays a dataframe that shows the duplicated rows between the datasets

        Raises
        ------
        DeepchecksValueError
            If the data is not a Dataset instance
        """
        test_dataset = context.test.sample(self.n_samples, random_state=self.random_state)
        train_dataset = context.train.sample(self.n_samples, random_state=self.random_state)

        train_dataset.assert_features()
        test_dataset.assert_features()
        columns = test_dataset.features + ([test_dataset.label_name] if test_dataset.has_label() else [])

        # For pandas.groupby in python 3.6, there is problem with comparing numpy nan, so replace with placeholder
        train_df = _fillna(train_dataset.data)
        test_df = _fillna(test_dataset.data)

        train_uniques = _create_unique_frame(train_df, columns, text_prefix='Train indices: ')
        test_uniques = _create_unique_frame(test_df, columns, text_prefix='Test indices: ')

        duplicates_df, test_dup_count = _create_train_test_joined_duplicate_frame(train_uniques, test_uniques, columns)

        # Replace filler back to none
        duplicates_df = duplicates_df.applymap(lambda x: None if x == NAN_REPLACEMENT else x)
        dup_ratio = test_dup_count / test_dataset.n_samples
        user_msg = f'{format_percent(dup_ratio)} ({test_dup_count} / {test_dataset.n_samples}) \
                     of test data samples appear in train data'
        display = [user_msg, duplicates_df.head(self.n_to_show)] if context.with_display and dup_ratio else None
        result = {'ratio': dup_ratio, 'data': duplicates_df}
        return CheckResult(result, header='Train Test Samples Mix', display=display)

