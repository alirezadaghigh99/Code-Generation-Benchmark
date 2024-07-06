def get_icl_task_dataloader(
    icl_task_type: str,
    dataset_uri: str,
    tokenizer: Union[transformers.PreTrainedTokenizer,
                     transformers.PreTrainedTokenizerFast],
    batch_size: int,
    has_categories: bool = False,
    hf_loading_vars: Optional[Dict] = None,
    hf_parsing_map: Optional[Dict] = None,
    destination_path: str = '',
    kwargs: Optional[Dict[str, Any]] = None,
) -> Union[DataSpec, Dict[str, DataSpec]]:
    r"""Constructs a dataloader (or dataloaders if has_categories is True)

    capable of evaluating LLMs on in-context learning language modeling tasks,
    for example LAMBADA. An example usage is below:

        .. testsetup::

            import transformers
            from composer.models import HuggingFaceModel
            from composer.trainer import Trainer
            dataset_uri = "/tmp/dataset_uri.jsonl"
            dataset = RandomTextClassificationDataset(size=16, use_keys=True)
            train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
            hf_model, tokenizer = HuggingFaceModel.hf_from_composer_checkpoint('composer-hf-checkpoint.pt')
            # At this point, hf_model is randomly initialized
            composer_model = HuggingFaceModel(hf_model, hf_tokenizer)

        Example:

        .. testcode::


            dl = get_icl_task_dataloader(
                'language_modeling',
                dataset_uri,
                tokenizer,
                batch_size=2,
                max_seq_len=2048,
                pad_tok_id=tokenizer.pad_token_id,
                num_fewshot=10,
                prompt_string='translate english to french',
                example_delimiter='\\n',
                continuation_delimiter=''
                )
            eval_evaluator = Evaluator(
                    label="lambada",
                    dataloader=dl,
                    metric_names=['InContextLearningLMAccuracy']
                )
            trainer = Trainer(
                    model=model,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_evaluator,
                    optimizers=optimizer,
                    max_duration="1ep",
                )

    Args:
        icl_task_type (str): Name of icl_task type. One of ['multiple_choice', 'schema', 'language_modeling', 'generation_task_with_answers', 'code_evaluation']
        dataset_uri (str): A local path, a remote path beginning with ``s3://`` or another backend, or a HuggingFace dataset uri prepended with ``hf://``.
            Alternate backends must be supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
            A local dataset must consist of rows of JSON data points with task dependant fields.
            The default keys expected are "context" and "answer".
        tokenizer (transformers.PreTrainedTokenizerBase): The tokenizer used to map between strings and token ids.
        batch_size (int): Size of a batch used for eval
        has_categories: (bool): If ``True``, we will search the dataset file for a category key, and partition the dataset into a separate dataloader for each category occurring in the data.
        hf_loading_vars (Dict, default = None): A dictionary containing keyword arguments to be passed into `load_dataset` if dataset is being pulled from HF.
        hf_parsing_map (Dict, default = None): A dictionary containing a mapping from HF columns to ICL dataset keys. The dictionary should be formatted {icl_key:[hf_key1, hf_key1]}.
            Column contents will be concatenated with ' ' separating them. If not included, will load the columns already present in the HF dataset.
        kwargs (Dict[str, Any], default=None): Dictionary containing a mapping
        from ICL dataset constructor's parameter names and their desired values.

    Returns:
        DataLoader: A dataloader used for performing in-context learning evaluation on the dataset provided.
    """
    if hf_loading_vars is None:
        hf_loading_vars = {}
    if hf_parsing_map is None:
        hf_parsing_map = {}
    if has_categories:
        result_dls = {}
        output_files = partition_dataset_by_category(
            dataset_uri,
            destination_path,
            hf_loading_vars,
            hf_parsing_map,
        )
        categories = sorted(output_files.keys())
        for category in categories:
            partition_uri = output_files[category]
            result_dls[category] = build_icl_dataloader(
                icl_task_type=icl_task_type,
                dataset_uri=partition_uri,
                tokenizer=tokenizer,
                batch_size=batch_size,
                destination_path=partition_uri + '_tmp',
                hf_loading_vars=hf_loading_vars,
                hf_parsing_map=hf_parsing_map,
                kwargs=kwargs,
            )
        return result_dls
    else:
        return build_icl_dataloader(
            icl_task_type=icl_task_type,
            dataset_uri=dataset_uri,
            tokenizer=tokenizer,
            batch_size=batch_size,
            hf_loading_vars=hf_loading_vars,
            hf_parsing_map=hf_parsing_map,
            destination_path=destination_path,
            kwargs=kwargs,
        )

