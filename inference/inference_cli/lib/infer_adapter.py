def save_prediction(
    reference: Union[str, int],
    prediction: dict,
    output_location: str,
) -> None:
    target_path = prepare_target_path(
        reference=reference, output_location=output_location, extension="json"
    )
    dump_json(path=target_path, content=prediction)