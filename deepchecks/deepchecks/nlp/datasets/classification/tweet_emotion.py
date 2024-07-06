def load_data(data_format: str = 'TextData', as_train_test: bool = True,
              include_properties: bool = True, include_embeddings: bool = False) -> \
        t.Union[t.Tuple, t.Union[TextData, pd.DataFrame]]:
    """Load and returns the Tweet Emotion dataset (classification).

    Parameters
    ----------
    data_format : str, default: 'TextData'
        Represent the format of the returned value. Can be 'TextData'|'DataFrame'
        'TextData' will return the data as a TextData object
        'Dataframe' will return the data as a pandas DataFrame object
    as_train_test : bool, default: True
        If True, the returned data is split into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        In order to get this model, call the load_fitted_model() function.
        Otherwise, returns a single object.
    include_properties : bool, default: True
        If True, the returned data will include the properties of the tweets. Incompatible with data_format='DataFrame'
    include_embeddings : bool, default: True
        If True, the returned data will include the embeddings of the tweets. Incompatible with data_format='DataFrame'

    Returns
    -------
    dataset : Union[TextData, pd.DataFrame]
        the data object, corresponding to the data_format attribute.
    train, test : Tuple[Union[TextData, pd.DataFrame],Union[TextData, pd.DataFrame]
        tuple if as_train_test = True. Tuple of two objects represents the dataset split to train and test sets.
    """
    if data_format.lower() not in ['textdata', 'dataframe']:
        raise ValueError('data_format must be either "TextData" or "Dataframe"')
    elif data_format.lower() == 'dataframe':
        if include_properties or include_embeddings:
            warnings.warn('include_properties and include_embeddings are incompatible with data_format="Dataframe". '
                          'loading only original text data.',
                          UserWarning)

    data = read_and_save_data(ASSETS_DIR, 'tweet_emotion_data.csv', _FULL_DATA_URL, to_numpy=False)
    if not as_train_test:
        data.drop(columns=['train_test_split'], inplace=True)
        if data_format.lower() != 'textdata':
            return data

        metadata = data.drop(columns=[_target, 'text'])
        properties = load_properties(as_train_test=False) if include_properties else None
        embeddings = load_embeddings(as_train_test=False) if include_embeddings else None

        dataset = TextData(data.text, label=data[_target], task_type='text_classification',
                           metadata=metadata, embeddings=embeddings, properties=properties,
                           categorical_metadata=_CAT_METADATA)
        return dataset

    else:
        # train has more sport and Customer Complains but less Terror and Optimism
        train = data[data['train_test_split'] == 'Train'].drop(columns=['train_test_split'])
        test = data[data['train_test_split'] == 'Test'].drop(columns=['train_test_split'])

        if data_format.lower() != 'textdata':
            return train, test

        train_metadata, test_metadata = train.drop(columns=[_target, 'text']), test.drop(columns=[_target, 'text'])
        train_properties, test_properties = load_properties(as_train_test=True) if include_properties else (None, None)
        train_embeddings, test_embeddings = load_embeddings(as_train_test=True) if include_embeddings else (None, None)

        train_ds = TextData(train.text, label=train[_target], task_type='text_classification',
                            metadata=train_metadata, embeddings=train_embeddings, properties=train_properties,
                            categorical_metadata=_CAT_METADATA)
        test_ds = TextData(test.text, label=test[_target], task_type='text_classification',
                           metadata=test_metadata, embeddings=test_embeddings, properties=test_properties,
                           categorical_metadata=_CAT_METADATA)

        return train_ds, test_ds

def load_precalculated_predictions(pred_format: str = 'predictions', as_train_test: bool = True) -> \
        t.Union[np.array, t.Tuple[np.array, np.array]]:
    """Load and return a precalculated predictions for the dataset.

    Parameters
    ----------
    pred_format : str, default: 'predictions'
        Represent the format of the returned value. Can be 'predictions' or 'probabilities'.
        'predictions' will return the predicted class for each sample.
        'probabilities' will return the predicted probabilities for each sample.
    as_train_test : bool, default: True
        If True, the returned data is split into train and test exactly like the toy model
        was trained. The first return value is the train data and the second is the test data.
        Otherwise, returns a single object.

    Returns
    -------
    predictions : np.ndarray
        The prediction of the data elements in the dataset.

    """
    all_preds = read_and_save_data(ASSETS_DIR, 'tweet_emotion_probabilities.csv', _PREDICTIONS_URL, to_numpy=True)
    if pred_format == 'predictions':
        all_preds = np.array([_LABEL_MAP[x] for x in np.argmax(all_preds, axis=1)])
    elif pred_format != 'probabilities':
        raise ValueError('pred_format must be either "predictions" or "probabilities"')

    if as_train_test:
        train_indexes, test_indexes = _get_train_test_indexes()
        return all_preds[train_indexes], all_preds[test_indexes]
    else:
        return all_preds

