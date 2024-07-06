def download_cli(cfg):
    """Download images from the Lightly platform.

    Args:
        cfg:
            The default configs are loaded from the config file.
            To overwrite them please see the section on the config file
            (.config.config.yaml).

    Command-Line Args:
        tag_name:
            Download all images from the requested tag. Use initial-tag
            to get all images from the dataset.
        token:
            User access token to the Lightly platform. If dataset_id
            and token are specified, the images and embeddings are
            uploaded to the platform.
        dataset_id:
            Identifier of the dataset on the Lightly platform. If
            dataset_id and token are specified, the images and
            embeddings are uploaded to the platform.
        input_dir:
            If input_dir and output_dir are specified, lightly will copy
            all images belonging to the tag from the input_dir to the
            output_dir.
        output_dir:
            If input_dir and output_dir are specified, lightly will copy
            all images belonging to the tag from the input_dir to the
            output_dir.

    Examples:
        >>> # download list of all files in the dataset from the Lightly platform
        >>> lightly-download token='123' dataset_id='XYZ'
        >>>
        >>> # download list of all files in tag 'my-tag' from the Lightly platform
        >>> lightly-download token='123' dataset_id='XYZ' tag_name='my-tag'
        >>>
        >>> # download all images in tag 'my-tag' from the Lightly platform
        >>> lightly-download token='123' dataset_id='XYZ' tag_name='my-tag' output_dir='my_data/'
        >>>
        >>> # copy all files in 'my-tag' to a new directory
        >>> lightly-download token='123' dataset_id='XYZ' tag_name='my-tag' input_dir='data/' output_dir='my_data/'


    """
    _download_cli(cfg)

