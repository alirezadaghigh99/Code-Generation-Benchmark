def one_step_forward(rnn, batch_input, recurrent_state):
    """One-step batch forward computation of a recurrent module.

    Args:
        rnn (torch.nn.Module): Recurrent module.
        batch_input (BatchData): One-step batched input.
        recurrent_state (object): Batched recurrent state.

    Returns:
        object: One-step batched output.
        object: New batched recurrent state.
    """
    pack = pack_one_step_batch_as_sequences(batch_input)
    y, recurrent_state = rnn(pack, recurrent_state)
    return unpack_sequences_as_one_step_batch(y), recurrent_state