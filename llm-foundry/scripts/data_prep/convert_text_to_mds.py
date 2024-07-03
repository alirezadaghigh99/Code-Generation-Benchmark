def convert_text_to_mds(
    tokenizer_name: str,
    output_folder: str,
    input_folder: str,
    concat_tokens: int,
    eos_text: str,
    bos_text: str,
    no_wrap: bool,
    compression: str,
    processes: int,
    args_str: str,
    reprocess: bool,
    trust_remote_code: bool,
):
    """Convert a folder of text files to MDS format.

    Args:
        tokenizer_name (str): Name of tokenizer to use
        output_folder (str): Folder to write MDS shards to
        input_folder (str): Folder of text files to process
        concat_tokens (int): Concatenate up to this many tokens
        eos_text (str): Text to append to each example to separate concatenated samples
        bos_text (str): Text to prepend to each example to separate concatenated samples
        no_wrap: (bool): Whether to let text examples wrap across multiple training examples
        compression (str): The compression algorithm to use for MDS writing
        processes (int): The number of processes to use.
        args_str (str): String representation of the arguments
        reprocess (bool): Whether to always reprocess the given folder of text files
        trust_remote_code (bool): If true, allows custom code to be executed to load the tokenizer
    """
    is_remote_output = is_remote_path(output_folder)

    object_names = get_object_names(input_folder)
    if len(object_names) == 0:
        raise InputFolderMissingDataError(input_folder)

    # Check if the text files in the bucket have already been processed.
    if not reprocess and is_already_processed(
        output_folder,
        args_str,
        object_names,
    ):
        log.info(
            f'Input folder {input_folder} is already processed at {output_folder} and '
            +
            'reprocess is set to False. Set reprocess to True if you would like to force reprocessing.',
        )
        return

    # Use a temporary local directory if the output is remote and there are more than 1 processes
    local_output_folder = tempfile.TemporaryDirectory(
    ).name if is_remote_output else output_folder

    if os.path.isdir(output_folder) and len(os.listdir(output_folder)) > 0:
        raise OutputFolderNotEmptyError(output_folder)

    if processes > 1:
        # Download and convert the text files in parallel
        args = get_task_args(
            object_names,
            local_output_folder,
            input_folder,
            processes,
            tokenizer_name,
            concat_tokens,
            eos_text,
            bos_text,
            no_wrap,
            compression,
            trust_remote_code,
        )
        with ProcessPoolExecutor(max_workers=processes) as executor:
            list(executor.map(download_and_convert_starargs, args))

        # Merge the mds shards from each of the processes into a single folder
        merge_shard_groups(local_output_folder)
    else:
        download_and_convert(
            object_names,
            local_output_folder,
            input_folder,
            tokenizer_name,
            concat_tokens,
            eos_text,
            bos_text,
            no_wrap,
            compression,
            trust_remote_code,
        )

    # Write a done file with the args and object names
    write_done_file(local_output_folder, args_str, object_names)

    if is_remote_output:
        # Upload the local output to the remote location
        output_object_store = cast(
            ObjectStore,
            maybe_create_object_store_from_uri(output_folder),
        )
        _, _, output_folder_prefix = parse_uri(output_folder)
        files_to_upload = os.listdir(local_output_folder)

        for file in files_to_upload:
            assert not os.path.isdir(file)
            remote_path = os.path.join(output_folder_prefix, file)
            output_object_store.upload_object(
                remote_path,
                os.path.join(local_output_folder, file),
            )