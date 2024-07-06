def load_synthetic_binary_classification_dataset():
    """Loads the synthetic binary classification dataset provided for testing the NannyML package.

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference period of synthetic binary classification dataset
    analysis : pd.DataFrame
        A DataFrame containing analysis period of synthetic binary classification dataset
    analysis_tgt : pd.DataFrame
        A DataFrame containing target values for the analysis period of synthetic binary
        classification dataset

    Examples
    --------
    >>> from nannyml.datasets import load_synthetic_binary_classification_dataset
    >>> reference_df, analysis_df, analysis_targets_df = load_synthetic_binary_classification_dataset()

    """
    reference = load_csv_file_to_df('synthetic_sample_reference.csv')
    analysis = load_csv_file_to_df('synthetic_sample_analysis.csv')
    analysis_gt = load_csv_file_to_df('synthetic_sample_analysis_gt.csv')

    return reference, analysis, analysis_gt

def load_synthetic_multiclass_classification_dataset():
    """Loads the synthetic multiclass classification dataset provided for testing the NannyML package.

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference period of synthetic multiclass classification dataset
    analysis : pd.DataFrame
        A DataFrame containing analysis period of synthetic multiclass classification dataset
    analysis_tgt : pd.DataFrame
        A DataFrame containing target values for the analysis period of synthetic
        multiclass classification dataset

    Examples
    --------
    >>> from nannyml.datasets import load_synthetic_multiclass_classification_dataset
    >>> reference_df, analysis_df, analysis_targets_df = load_synthetic_multiclass_classification_dataset()

    """
    reference = load_csv_file_to_df('mc_reference.csv')
    analysis = load_csv_file_to_df('mc_analysis.csv')
    analysis_gt = load_csv_file_to_df('mc_analysis_gt.csv')

    return reference, analysis, analysis_gt

def load_synthetic_car_price_dataset():
    """Loads the synthetic car price dataset provided for testing the NannyML package on regression problems.

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference period of synthetic car price dataset
    analysis : pd.DataFrame
        A DataFrame containing analysis period of synthetic car price dataset
    analysis_tgt : pd.DataFrame
        A DataFrame containing target values for the analysis period of synthetic car price dataset

    Examples
    --------
    >>> from nannyml.datasets import load_synthetic_car_price_dataset
    >>> reference, analysis, analysis_tgt = load_synthetic_car_price_dataset()

    """

    reference = load_csv_file_to_df('regression_synthetic_reference.csv')
    analysis = load_csv_file_to_df('regression_synthetic_analysis.csv')
    analysis_tgt = load_csv_file_to_df('regression_synthetic_analysis_targets.csv')

    return reference, analysis, analysis_tgt

def load_synthetic_car_loan_data_quality_dataset():
    """Loads the synthetic car loan binary classification dataset that contains missing values
    provided for testing the NannyML package.

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference period of
        synthetic car loan binary classification dataset that contains missing values
    analysis : pd.DataFrame
        A DataFrame containing analysis period of
        synthetic car loan binary classification dataset that contains missing values
    analysis_tgt : pd.DataFrame
        A DataFrame containing target values for the analysis period of
        synthetic car loan binary classification dataset that contains missing values

    Examples
    --------
    >>> from nannyml.datasets import load_synthetic_car_loan_w_missing_dataset
    >>> reference_df, analysis_df, analysis_targets_df = load_synthetic_car_loan_w_missing_dataset()

    """
    reference = load_csv_file_to_df('synthetic_car_loan_dq_reference.csv')
    analysis = load_csv_file_to_df('synthetic_car_loan_dq_analysis.csv')
    analysis_gt = load_csv_file_to_df('synthetic_car_loan_analysis_target.csv')

    return reference, analysis, analysis_gt

def load_synthetic_car_loan_dataset():
    """Loads the synthetic car loan binary classification dataset provided for testing the NannyML package.

    Returns
    -------
    reference : pd.DataFrame
        A DataFrame containing reference period of synthetic binary classification dataset
    analysis : pd.DataFrame
        A DataFrame containing analysis period of synthetic binary classification dataset
    analysis_tgt : pd.DataFrame
        A DataFrame containing target values for the analysis period of synthetic binary
        classification dataset

    Examples
    --------
    >>> from nannyml.datasets import load_synthetic_car_loan_dataset
    >>> reference_df, analysis_df, analysis_targets_df = load_synthetic_car_loan_dataset()

    """
    reference = load_csv_file_to_df('synthetic_car_loan_reference.csv')
    analysis = load_csv_file_to_df('synthetic_car_loan_analysis.csv')
    analysis_gt = load_csv_file_to_df('synthetic_car_loan_analysis_target.csv')

    return reference, analysis, analysis_gt

