class ErrorRateStats(MetricStats):
    """A class for tracking error rates (e.g., WER, PER).

    Arguments
    ---------
    merge_tokens : bool
        Whether to merge the successive tokens (used for e.g.,
        creating words out of character tokens).
        See ``speechbrain.dataio.dataio.merge_char``.
    split_tokens : bool
        Whether to split tokens (used for e.g. creating
        characters out of word tokens).
        See ``speechbrain.dataio.dataio.split_word``.
    space_token : str
        The character to use for boundaries. Used with ``merge_tokens``
        this represents character to split on after merge.
        Used with ``split_tokens`` the sequence is joined with
        this token in between, and then the whole sequence is split.
    keep_values : bool
        Whether to keep the values of the concepts or not.
    extract_concepts_values : bool
        Process the predict and target to keep only concepts and values.
    tag_in : str
        Start of the concept ('<' for example).
    tag_out : str
        End of the concept ('>' for example).
    equality_comparator : Callable[[str, str], bool]
        The function used to check whether two words are equal.

    Example
    -------
    >>> cer_stats = ErrorRateStats()
    >>> i2l = {0: 'a', 1: 'b'}
    >>> cer_stats.append(
    ...     ids=['utterance1'],
    ...     predict=torch.tensor([[0, 1, 1]]),
    ...     target=torch.tensor([[0, 1, 0]]),
    ...     target_len=torch.ones(1),
    ...     ind2lab=lambda batch: [[i2l[int(x)] for x in seq] for seq in batch],
    ... )
    >>> stats = cer_stats.summarize()
    >>> stats['WER']
    33.33...
    >>> stats['insertions']
    0
    >>> stats['deletions']
    0
    >>> stats['substitutions']
    1
    """

    def __init__(
        self,
        merge_tokens=False,
        split_tokens=False,
        space_token="_",
        keep_values=True,
        extract_concepts_values=False,
        tag_in="",
        tag_out="",
        equality_comparator: Callable[[str, str], bool] = _str_equals,
    ):
        self.clear()
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token
        self.extract_concepts_values = extract_concepts_values
        self.keep_values = keep_values
        self.tag_in = tag_in
        self.tag_out = tag_out
        self.equality_comparator = equality_comparator

    def append(
        self,
        ids,
        predict,
        target,
        predict_len=None,
        target_len=None,
        ind2lab=None,
    ):
        """Add stats to the relevant containers.

        * See MetricStats.append()

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : torch.tensor
            A predicted output, for comparison with the target output
        target : torch.tensor
            The correct reference output, for comparison with the prediction.
        predict_len : torch.tensor
            The predictions relative lengths, used to undo padding if
            there is padding present in the predictions.
        target_len : torch.tensor
            The target outputs' relative lengths, used to undo padding if
            there is padding present in the target.
        ind2lab : callable
            Callable that maps from indices to labels, operating on batches,
            for writing alignments.
        """
        self.ids.extend(ids)

        if predict_len is not None:
            predict = undo_padding(predict, predict_len)

        if target_len is not None:
            target = undo_padding(target, target_len)

        if ind2lab is not None:
            predict = ind2lab(predict)
            target = ind2lab(target)

        if self.merge_tokens:
            predict = merge_char(predict, space=self.space_token)
            target = merge_char(target, space=self.space_token)

        if self.split_tokens:
            predict = split_word(predict, space=self.space_token)
            target = split_word(target, space=self.space_token)

        if self.extract_concepts_values:
            predict = extract_concepts_values(
                predict,
                self.keep_values,
                self.tag_in,
                self.tag_out,
                space=self.space_token,
            )
            target = extract_concepts_values(
                target,
                self.keep_values,
                self.tag_in,
                self.tag_out,
                space=self.space_token,
            )

        scores = wer_details_for_batch(
            ids,
            target,
            predict,
            compute_alignments=True,
            equality_comparator=self.equality_comparator,
        )

        self.scores.extend(scores)

    def summarize(self, field=None):
        """Summarize the error_rate and return relevant statistics.

        * See MetricStats.summarize()
        """
        self.summary = wer_summary(self.scores)

        # Add additional, more generic key
        self.summary["error_rate"] = self.summary["WER"]

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        if not self.summary:
            self.summarize()

        print_wer_summary(self.summary, filestream)
        print_alignments(self.scores, filestream)

