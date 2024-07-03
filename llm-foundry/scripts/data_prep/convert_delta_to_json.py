def fetch_DT(args: Namespace) -> None:
    """Fetch UC Delta Table to local as jsonl."""
    log.info(f'Start .... Convert delta to json')

    obj = urllib.parse.urlparse(args.json_output_folder)
    if obj.scheme != '':
        raise ValueError(
            'Check the json_output_folder and verify it is a local path!',
        )

    if os.path.exists(args.json_output_folder):
        if not os.path.isdir(args.json_output_folder) or os.listdir(
            args.json_output_folder,
        ):
            raise RuntimeError(
                f'Output folder {args.json_output_folder} already exists and is not empty. Please remove it and retry.',
            )

    os.makedirs(args.json_output_folder, exist_ok=True)

    if not args.json_output_filename.endswith('.jsonl'):
        raise ValueError('json_output_filename needs to be a jsonl file')

    log.info(f'Directory {args.json_output_folder} created.')

    method, dbsql, sparkSession = validate_and_get_cluster_info(
        cluster_id=args.cluster_id,
        databricks_host=args.DATABRICKS_HOST,
        databricks_token=args.DATABRICKS_TOKEN,
        http_path=args.http_path,
        use_serverless=args.use_serverless,
    )

    fetch(
        method,
        args.delta_table_name,
        args.json_output_folder,
        args.batch_size,
        args.processes,
        sparkSession,
        dbsql,
    )

    if dbsql is not None:
        dbsql.close()

    # combine downloaded jsonl into one big jsonl for IFT
    iterative_combine_jsons(
        args.json_output_folder,
        os.path.join(args.json_output_folder, args.json_output_filename),
    )