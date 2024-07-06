def download(
        lang='en',
        model_dir=DEFAULT_MODEL_DIR,
        package='default',
        processors={},
        logging_level=None,
        verbose=None,
        resources_url=DEFAULT_RESOURCES_URL,
        resources_branch=None,
        resources_version=DEFAULT_RESOURCES_VERSION,
        model_url=DEFAULT_MODEL_URL,
        proxies=None,
        download_json=True
    ):
    # set global logging level
    set_logging_level(logging_level, verbose)
    # process different pipeline parameters
    lang, model_dir, package, processors = process_pipeline_parameters(
        lang, model_dir, package, processors
    )

    if download_json or not os.path.exists(os.path.join(model_dir, 'resources.json')):
        if not download_json:
            logger.warning("Asked to skip downloading resources.json, but the file does not exist.  Downloading anyway")
        download_resources_json(model_dir, resources_url, resources_branch, resources_version, resources_filepath=None, proxies=proxies)

    resources = load_resources_json(model_dir)
    if lang not in resources:
        raise UnknownLanguageError(lang)
    if 'alias' in resources[lang]:
        logger.info(f'"{lang}" is an alias for "{resources[lang]["alias"]}"')
        lang = resources[lang]['alias']
    lang_name = resources.get(lang, {}).get('lang_name', lang)
    url = expand_model_url(resources, model_url)

    # Default: download zipfile and unzip
    if package == 'default' and (processors is None or len(processors) == 0):
        logger.info(
            f'Downloading default packages for language: {lang} ({lang_name}) ...'
        )
        # want the URL to become, for example:
        # https://huggingface.co/stanfordnlp/stanza-af/resolve/v1.3.0/models/default.zip
        # so we hopefully start from
        # https://huggingface.co/stanfordnlp/stanza-{lang}/resolve/v{resources_version}/models/{filename}
        request_file(
            url.format(resources_version=resources_version, lang=lang, filename="default.zip"),
            os.path.join(model_dir, lang, f'default.zip'),
            proxies,
            md5=resources[lang]['default_md5'],
        )
        unzip(os.path.join(model_dir, lang), 'default.zip')
    # Customize: maintain download list
    else:
        download_list = maintain_processor_list(resources, lang, package, processors, allow_pretrain=True)
        download_list = add_dependencies(resources, lang, download_list)
        download_list = flatten_processor_list(download_list)
        download_models(download_list=download_list,
                        resources=resources,
                        lang=lang,
                        model_dir=model_dir,
                        resources_version=resources_version,
                        model_url=model_url,
                        proxies=proxies,
                        log_info=True)
    logger.info(f'Finished downloading models and saved to {model_dir}')

def load_resources_json(model_dir=DEFAULT_MODEL_DIR, resources_filepath=None):
    """
    Unpack the resources json file from the given model_dir
    """
    if resources_filepath is None:
        resources_filepath = os.path.join(model_dir, 'resources.json')
    if not os.path.exists(resources_filepath):
        raise ResourcesFileNotFoundError(resources_filepath)
    with open(resources_filepath) as fin:
        resources = json.load(fin)
    return resources

