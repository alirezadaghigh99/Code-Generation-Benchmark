def calculate_builtin_properties(
        raw_text: Sequence[str],
        include_properties: Optional[List[str]] = None,
        ignore_properties: Optional[List[str]] = None,
        include_long_calculation_properties: bool = False,
        ignore_non_english_samples_for_english_properties: bool = True,
        device: Optional[str] = None,
        models_storage: Union[pathlib.Path, str, None] = None,
        batch_size: Optional[int] = 16,
        cache_models: bool = False,
        use_onnx_models: bool = True,
) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
    """Calculate properties on provided text samples.

    Parameters
    ----------
    raw_text : Sequence[str]
        The text to calculate the properties for.
    include_properties : List[str], default None
        The properties to calculate. If None, all default properties will be calculated. Cannot be used
        together with ignore_properties parameter. Available properties are:
        ['Text Length', 'Average Word Length', 'Max Word Length', '% Special Characters', '% Punctuation', 'Language',
        'Sentiment', 'Subjectivity', 'Toxicity', 'Fluency', 'Formality', 'Lexical Density', 'Unique Noun Count',
        'Reading Ease', 'Average Words Per Sentence', 'URLs Count', Unique URLs Count', 'Email Address Count',
        'Unique Email Address Count', 'Unique Syllables Count', 'Reading Time', 'Sentences Count',
        'Average Syllable Length']
        List of default properties are: ['Text Length', 'Average Word Length', 'Max Word Length',
        '% Special Characters', '% Punctuation', 'Language', 'Sentiment', 'Subjectivity', 'Toxicity', 'Fluency',
        'Formality', 'Lexical Density', 'Unique Noun Count', 'Reading Ease', 'Average Words Per Sentence']
        To calculate all the default properties, the include_properties and ignore_properties parameters should
        be None. If you pass either include_properties or ignore_properties then only the properties specified
        in the list will be calculated or ignored.
        Note that the properties ['Toxicity', 'Fluency', 'Formality', 'Language', 'Unique Noun Count'] may
        take a long time to calculate. If include_long_calculation_properties is False, these properties will be
        ignored, even if they are in the include_properties parameter.
    ignore_properties : List[str], default None
        The properties to ignore from the list of default properties. If None, no properties will be ignored and
        all the default properties will be calculated. Cannot be used together with include_properties parameter.
    include_long_calculation_properties : bool, default False
        Whether to include properties that may take a long time to calculate. If False, these properties will be
        ignored, unless they are specified in the include_properties parameter explicitly.
    ignore_non_english_samples_for_english_properties : bool, default True
        Whether to ignore samples that are not in English when calculating English properties. If False, samples
        that are not in English will be calculated as well. This parameter is ignored when calculating non-English
        properties.
        English-Only properties WILL NOT work properly on non-English samples, and this parameter should be used
        only when you are sure that all the samples are in English.
    device : Optional[str], default None
        The device to use for the calculation. If None, the default device will be used. For onnx based models it is
        recommended to set device to None for optimized performance.
    models_storage : Union[str, pathlib.Path, None], default None
        A directory to store the models.
        If not provided, models will be stored in `DEEPCHECKS_LIB_PATH/nlp/.nlp-models`.
        Also, if a folder already contains relevant resources they are not re-downloaded.
    batch_size : int, default 8
        The batch size.
    cache_models : bool, default False
        If True, will store the models in device RAM memory. This will speed up the calculation for future calls.
    use_onnx_models : bool, default True
        If True, will use onnx gpu optimized models for the calculation. Requires the optimum[onnxruntime-gpu] library
        to be installed as well as the availability of GPU.

    Returns
    -------
    Dict[str, List[float]]
        A dictionary with the property name as key and a list of the property values for each text as value.
    Dict[str, str]
        A dictionary with the property name as key and the property's type as value.
    """
    use_onnx_models = _validate_onnx_model_availability(use_onnx_models, device)
    text_properties = _select_properties(
        include_properties=include_properties,
        ignore_properties=ignore_properties,
        include_long_calculation_properties=include_long_calculation_properties
    )

    properties_types = {
        it['name']: it['output_type']
        for it in text_properties
    }
    _warn_long_compute(device, properties_types, len(raw_text), use_onnx_models)

    kwargs = dict(device=device, models_storage=models_storage)
    calculated_properties = {k: [] for k in properties_types.keys()}

    # Prepare kwargs for properties that require outside resources:
    kwargs['fasttext_model'] = get_fasttext_model(models_storage=models_storage, use_cache=cache_models)

    properties_requiring_cmudict = list(set(CMUDICT_PROPERTIES) & set(properties_types.keys()))
    if properties_requiring_cmudict:
        if not nltk_download('cmudict', quiet=True):
            _warn_if_missing_nltk_dependencies('cmudict', format_list(properties_requiring_cmudict))
            for prop in properties_requiring_cmudict:
                calculated_properties[prop] = [np.nan] * len(raw_text)
        kwargs['cmudict_dict'] = get_cmudict_dict(use_cache=cache_models)

    if 'Toxicity' in properties_types and 'toxicity_classifier' not in kwargs:
        model_name = TOXICITY_MODEL_NAME_ONNX if use_onnx_models else TOXICITY_MODEL_NAME
        kwargs['toxicity_classifier'] = get_transformer_pipeline(
            property_name='toxicity', model_name=model_name, device=device,
            models_storage=models_storage, use_cache=cache_models, use_onnx_model=use_onnx_models)

    if 'Formality' in properties_types and 'formality_classifier' not in kwargs:
        model_name = FORMALITY_MODEL_NAME_ONNX if use_onnx_models else FORMALITY_MODEL_NAME
        kwargs['formality_classifier'] = get_transformer_pipeline(
            property_name='formality', model_name=model_name, device=device,
            models_storage=models_storage, use_cache=cache_models, use_onnx_model=use_onnx_models)

    if 'Fluency' in properties_types and 'fluency_classifier' not in kwargs:
        model_name = FLUENCY_MODEL_NAME_ONNX if use_onnx_models else FLUENCY_MODEL_NAME
        kwargs['fluency_classifier'] = get_transformer_pipeline(
            property_name='fluency', model_name=model_name, device=device,
            models_storage=models_storage, use_cache=cache_models, use_onnx_model=use_onnx_models)

    # Remove language property from the list of properties to calculate as it will be calculated separately:
    text_properties = [prop for prop in text_properties if prop['name'] != 'Language']

    warning_message = (
        'Failed to calculate property {0}. '
        'Dependencies required by property are not installed. '
        'Error:\n{1}'
    )
    import_warnings = set()

    # Calculate all properties for a specific batch than continue to the next batch
    for i in tqdm(range(0, len(raw_text), batch_size)):
        batch = raw_text[i:i + batch_size]
        batch_properties = defaultdict(list)

        # filtering out empty sequences
        nan_indices = {i for i, seq in enumerate(batch) if pd.isna(seq) is True}
        filtered_sequences = [e for i, e in enumerate(batch) if i not in nan_indices]

        samples_language = _batch_wrapper(text_batch=filtered_sequences, func=language, **kwargs)
        if 'Language' in properties_types:
            batch_properties['Language'].extend(samples_language)
            calculated_properties['Language'].extend(samples_language)
        kwargs['language_property_result'] = samples_language  # Pass the language property to other properties
        kwargs['batch_size'] = batch_size

        non_english_indices = set()
        if ignore_non_english_samples_for_english_properties:
            non_english_indices = {i for i, (seq, lang) in enumerate(zip(filtered_sequences, samples_language))
                                   if lang != 'en'}

        for prop in text_properties:
            if prop['name'] in import_warnings:  # Skip properties that failed to import:
                batch_properties[prop['name']].extend([np.nan] * len(batch))
                continue

            sequences_to_use = list(filtered_sequences)
            if prop['name'] in ENGLISH_ONLY_PROPERTIES and ignore_non_english_samples_for_english_properties:
                sequences_to_use = [e for i, e in enumerate(sequences_to_use) if i not in non_english_indices]
            try:
                if prop['name'] in BATCH_PROPERTIES:
                    value = run_available_kwargs(text_batch=sequences_to_use, func=prop['method'], **kwargs)
                else:
                    value = _batch_wrapper(text_batch=sequences_to_use, func=prop['method'], **kwargs)
                batch_properties[prop['name']].extend(value)
            except ImportError as e:
                warnings.warn(warning_message.format(prop['name'], str(e)))
                batch_properties[prop['name']].extend([np.nan] * len(batch))
                import_warnings.add(prop['name'])
                continue

            # Fill in nan values for samples that were filtered out:
            result_index = 0
            for index, seq in enumerate(batch):
                if index in nan_indices or (index in non_english_indices and
                                            ignore_non_english_samples_for_english_properties and
                                            prop['name'] in ENGLISH_ONLY_PROPERTIES):
                    calculated_properties[prop['name']].append(np.nan)
                else:
                    calculated_properties[prop['name']].append(batch_properties[prop['name']][result_index])
                    result_index += 1

        # Clear property caches:
        textblob_cache.clear()
        words_cache.clear()
        sentences_cache.clear()

    if not calculated_properties:
        raise RuntimeError('Failed to calculate any of the properties.')

    properties_types = {
        k: v
        for k, v in properties_types.items()
        if k in calculated_properties
    }

    return calculated_properties, properties_types

