def len_to_mask(lengths, zeros):
    """
    :param lengths: list of ints with the lengths of the sequences
    :param zeros: bool. If false, the first lengths[i] values will be True and the rest will be false.
            If true, the first values will be False and the rest True
    :return: Boolean tensor of dimension (L, T) with L = len(lenghts) and T = lengths.max(), where with rows with lengths[i] True values followed by lengths.max()-lengths[i] False values. The True and False values are inverted if `zeros == True`
    """  # noqa
    # Clean trick from:
    # https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
    mask = torch.arange(lengths.max(), device=lengths.device)[None, :] < lengths[:, None]
    if zeros:
        mask = ~mask  # Logical not
    return mask