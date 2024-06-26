def is_model_artefacts_bucket_available() -> bool:
    return (
        AWS_ACCESS_KEY_ID is not None
        and AWS_SECRET_ACCESS_KEY is not None
        and LAMBDA
        and S3_CLIENT is not None
    )