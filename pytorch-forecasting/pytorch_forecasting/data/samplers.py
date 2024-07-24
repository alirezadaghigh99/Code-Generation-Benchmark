class TimeSynchronizedBatchSampler(GroupedSampler):
    """
    Samples mini-batches randomly but in a time-synchronised manner.

    Time-synchornisation means that the time index of the first decoder samples are aligned across the batch.
    This sampler does not support missing values in the dataset.
    """

    def get_groups(self, sampler: Sampler):
        data_source = sampler.data_source
        index = data_source.index
        # get groups, i.e. group all samples by first predict time
        last_time = data_source.data["time"][index["index_end"].to_numpy()].numpy()
        decoder_lengths = data_source.calculate_decoder_length(last_time, index.sequence_length)
        first_prediction_time = index.time + index.sequence_length - decoder_lengths + 1
        groups = pd.RangeIndex(0, len(index.index)).groupby(first_prediction_time)
        return groups

