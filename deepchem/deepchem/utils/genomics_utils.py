def seq_one_hot_encode(sequences, letters: str = 'ATCGN') -> np.ndarray:
    """One hot encodes list of genomic sequences.

    Sequences encoded have shape (N_sequences, N_letters, sequence_length, 1).
    These sequences will be processed as images with one color channel.

    Parameters
    ----------
    sequences: np.ndarray or Iterator[Bio.SeqRecord]
        Iterable object of genetic sequences
    letters: str, optional (default "ATCGN")
        String with the set of possible letters in the sequences.

    Raises
    ------
    ValueError:
        If sequences are of different lengths.

    Returns
    -------
    np.ndarray
        A numpy array of shape `(N_sequences, N_letters, sequence_length, 1)`.
    """

    # The label encoder is given characters for ACGTN
    letter_encoder = {l: i for i, l in enumerate(letters)}
    alphabet_length = len(letter_encoder)

    # Peak at the first sequence to get the length of the sequence.
    if isinstance(sequences, np.ndarray):
        first_seq = sequences[0]
        tail_seq = sequences[1:]
    else:
        first_seq = next(sequences)
        tail_seq = sequences

    sequence_length = len(first_seq)
    seqs = []
    seqs.append(
        _seq_to_encoded(first_seq, letter_encoder, alphabet_length,
                        sequence_length))

    for other_seq in tail_seq:
        if len(other_seq) != sequence_length:
            raise ValueError("The genetic sequences must have a same length")
        seqs.append(
            _seq_to_encoded(other_seq, letter_encoder, alphabet_length,
                            sequence_length))

    return np.expand_dims(np.array(seqs), -1)

def encode_bio_sequence(fname: str,
                        file_type: str = "fasta",
                        letters: str = "ATCGN") -> np.ndarray:
    """
    Loads a sequence file and returns an array of one-hot sequences.

    Parameters
    ----------
    fname: str
        Filename of fasta file.
    file_type: str, optional (default "fasta")
        The type of file encoding to process, e.g. fasta or fastq, this
        is passed to Biopython.SeqIO.parse.
    letters: str, optional (default "ATCGN")
        The set of letters that the sequences consist of, e.g. ATCG.

    Returns
    -------
    np.ndarray
        A numpy array of shape `(N_sequences, N_letters, sequence_length, 1)`.

    Notes
    -----
    This function requires BioPython to be installed.
    """
    try:
        from Bio import SeqIO
    except ModuleNotFoundError:
        raise ImportError("This function requires BioPython to be installed.")

    sequences = SeqIO.parse(fname, file_type)
    return seq_one_hot_encode(sequences, letters)

