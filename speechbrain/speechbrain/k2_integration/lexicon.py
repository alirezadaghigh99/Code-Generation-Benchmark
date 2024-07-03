def prepare_char_lexicon(
    lang_dir,
    vocab_files,
    extra_csv_files=[],
    column_text_key="wrd",
    add_word_boundary=True,
):
    """
    Read extra_csv_files to generate a $lang_dir/lexicon.txt for k2 training.
    This usually includes the csv files of the training set and the dev set in the
    output_folder. During training, we need to make sure that the lexicon.txt contains
    all (or the majority of) the words in the training set and the dev set.

    NOTE: This assumes that the csv files contain the transcription in the last column.

    Also note that in each csv_file, the first line is the header, and the remaining
    lines are in the following format:

    ID, duration, wav, spk_id, wrd (transcription)

    We only need the transcription in this function.

    Writes out $lang_dir/lexicon.txt

    Note that the lexicon.txt is a text file with the following format:
    word1 phone1 phone2 phone3 ...
    word2 phone1 phone2 phone3 ...

    In this code, we simply use the characters in the word as the phones.
    You can use other phone sets, e.g., phonemes, BPEs, to train a better model.

    Arguments
    ---------
    lang_dir: str
        The directory to store the lexicon.txt
    vocab_files: List[str]
        A list of extra vocab files. For example, for librispeech this could be the
        librispeech-vocab.txt file.
    extra_csv_files: List[str]
        A list of csv file paths
    column_text_key: str
        The column name of the transcription in the csv file. By default, it is "wrd".
    add_word_boundary: bool
        whether to add word boundary symbols <eow> at the end of each line to the
        lexicon for every word.

    Example
    -------
    >>> from speechbrain.k2_integration.lexicon import prepare_char_lexicon
    >>> # Create some dummy csv files containing only the words `hello`, `world`.
    >>> # The first line is the header, and the remaining lines are in the following
    >>> # format:
    >>> # ID, duration, wav, spk_id, wrd (transcription)
    >>> csv_file = getfixture('tmpdir').join("train.csv")
    >>> # Data to be written to the CSV file.
    >>> import csv
    >>> data = [
    ...    ["ID", "duration", "wav", "spk_id", "wrd"],
    ...    [1, 1, 1, 1, "hello world"],
    ...    [2, 0.5, 1, 1, "hello"]
    ... ]
    >>> with open(csv_file, "w", newline="") as f:
    ...    writer = csv.writer(f)
    ...    writer.writerows(data)
    >>> extra_csv_files = [csv_file]
    >>> lang_dir = getfixture('tmpdir')
    >>> vocab_files = []
    >>> prepare_char_lexicon(lang_dir, vocab_files, extra_csv_files=extra_csv_files, add_word_boundary=False)
    """
    # Read train.csv, dev-clean.csv to generate a lexicon.txt for k2 training
    lexicon = dict()
    if len(extra_csv_files) != 0:
        for file in extra_csv_files:
            with open(file, "r") as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    # Split the transcription into words
                    words = row[column_text_key].split()
                    for word in words:
                        if word not in lexicon:
                            if add_word_boundary:
                                lexicon[word] = list(word) + [EOW]
                            else:
                                lexicon[word] = list(word)

    for file in vocab_files:
        with open(file) as f:
            for line in f:
                # Split the line
                word = line.strip().split()[0]
                # Split the transcription into words
                if word not in lexicon:
                    if add_word_boundary:
                        lexicon[word] = list(word) + [EOW]
                    else:
                        lexicon[word] = list(word)
    # Write the lexicon to lang_dir/lexicon.txt
    os.makedirs(lang_dir, exist_ok=True)
    with open(os.path.join(lang_dir, "lexicon.txt"), "w") as f:
        fc = f"{UNK} {UNK_t}\n"
        for word in lexicon:
            fc += word + " " + " ".join(lexicon[word]) + "\n"
        f.write(fc)