def convert_and_export_predictor(
    cfg,
    pytorch_model,
    predictor_type,
    output_dir,
    data_loader,
):
    """
    Entry point for convert and export model. This involves two steps:
        - convert: converting the given `pytorch_model` to another format, currently
            mainly for quantizing the model.
        - export: exporting the converted `pytorch_model` to predictor. This step
            should not alter the behaviour of model.
    """
    pytorch_model = convert_model(cfg, pytorch_model, predictor_type, data_loader)
    return export_predictor(cfg, pytorch_model, predictor_type, output_dir, data_loader)

