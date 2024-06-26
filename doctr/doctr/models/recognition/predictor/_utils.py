def remap_preds(
    preds: List[Tuple[str, float]], crop_map: List[Union[int, Tuple[int, int]]], dilation: float
) -> List[Tuple[str, float]]:
    remapped_out = []
    for _idx in crop_map:
        # Crop hasn't been split
        if isinstance(_idx, int):
            remapped_out.append(preds[_idx])
        else:
            # unzip
            vals, probs = zip(*preds[_idx[0] : _idx[1]])
            # Merge the string values
            remapped_out.append((merge_multi_strings(vals, dilation), min(probs)))  # type: ignore[arg-type]
    return remapped_out