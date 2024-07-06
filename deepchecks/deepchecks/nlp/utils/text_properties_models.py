def _get_transformer_model_and_tokenizer(
        property_name: str,
        model_name: str,
        models_storage: Union[pathlib.Path, str, None] = None,
        use_onnx_model: bool = True,
):
    """Return a transformers' model and tokenizer in cpu memory."""
    transformers = import_optional_property_dependency('transformers', property_name=property_name)

    with _log_suppressor():
        models_storage = get_create_model_storage(models_storage=models_storage)
        model_path = models_storage / model_name
        model_path_exists = model_path.exists()

        if use_onnx_model:
            onnx_runtime = import_optional_property_dependency('optimum.onnxruntime', property_name=property_name)
            classifier_cls = onnx_runtime.ORTModelForSequenceClassification
            if model_path_exists:
                model = classifier_cls.from_pretrained(model_path, provider='CUDAExecutionProvider')
            else:
                model = classifier_cls.from_pretrained(model_name, provider='CUDAExecutionProvider')
                model.save_pretrained(model_path)
        else:
            if model_path_exists:
                model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
                model.save_pretrained(model_path)
            model.eval()

        if model_path_exists:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(model_path)

        return model, tokenizer

