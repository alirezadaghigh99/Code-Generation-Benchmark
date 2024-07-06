def load_data(data_format: str = 'Dataset', as_train_test: bool = True) -> \
        t.Union[t.Tuple, t.Union[Dataset, pd.DataFrame]]:
    """Load and returns the Adult dataset (classification).

    Parameters
    ----------
    data_format : str, default: 'Dataset'
        Represent the format of the returned value. Can be 'Dataset'|'Dataframe'
        'Dataset' will return the data as a Dataset object
        'Dataframe' will return the data as a pandas Dataframe object

    as_train_test : bool, default: True
        If True, the returned data is splitted into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        In order to get this model, call the load_fitted_model() function.
        Otherwise, returns a single object.

    Returns
    -------
    dataset : Union[deepchecks.Dataset, pd.DataFrame]
        the data object, corresponding to the data_format attribute.
    train, test : Tuple[Union[deepchecks.Dataset, pd.DataFrame],Union[deepchecks.Dataset, pd.DataFrame]
        tuple if as_train_test = True. Tuple of two objects represents the dataset splitted to train and test sets.
    """
    if not as_train_test:
        dataset = pd.read_csv(_FULL_DATA_URL, names=_FEATURES + [_target])
        dataset['income'] = dataset['income'].str.replace('.', '', regex=True)      # fix label inconsistency

        if data_format == 'Dataset':
            dataset = Dataset(dataset, label=_target, cat_features=_CAT_FEATURES)
            return dataset
        elif data_format == 'Dataframe':
            return dataset
        else:
            raise ValueError('data_format must be either "Dataset" or "Dataframe"')
    else:
        train = pd.read_csv(_TRAIN_DATA_URL, names=_FEATURES + [_target])
        test = pd.read_csv(_TEST_DATA_URL, skiprows=1, names=_FEATURES + [_target])
        test[_target] = test[_target].str[:-1]

        if data_format == 'Dataset':
            train = Dataset(train, label=_target, cat_features=_CAT_FEATURES)
            test = Dataset(test, label=_target, cat_features=_CAT_FEATURES)
            return train, test
        elif data_format == 'Dataframe':
            return train, test
        else:
            raise ValueError('data_format must be either "Dataset" or "Dataframe"')

