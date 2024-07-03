def get_rnn(cell_type: Union[Type[RNN], str]) -> Type[RNN]:
    """
    Get LSTM or GRU.

    Args:
        cell_type (Union[RNN, str]): "LSTM" or "GRU"

    Returns:
        Type[RNN]: returns GRU or LSTM RNN module
    """
    if isinstance(cell_type, RNN):
        rnn = cell_type
    elif cell_type == "LSTM":
        rnn = LSTM
    elif cell_type == "GRU":
        rnn = GRU
    else:
        raise ValueError(f"RNN type {cell_type} is not supported. supported: [LSTM, GRU]")
    return rnn