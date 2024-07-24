class OnlineCLScenario(CLScenario):
    def __init__(
        self,
        original_streams: Iterable[CLStream[DatasetExperience[TCLDataset]]],
        experiences: Optional[
            Union[
                DatasetExperience[TCLDataset], Iterable[DatasetExperience[TCLDataset]]
            ]
        ] = None,
        experience_size: int = 10,
        stream_split_strategy: Literal[
            "fixed_size_split", "continuous_linear_decay"
        ] = "fixed_size_split",
        access_task_boundaries: bool = False,
        shuffle: bool = True,
        overlap_factor: int = 4,
        iters_per_virtual_epoch: int = 10,
    ):
        """Creates an online scenario from an existing CL scenario

        :param original_streams: The streams from the original CL scenario.
        :param experiences: If None, the online stream will be created
            from the `train_stream` of the original CL scenario, otherwise it
            will create an online stream from the given sequence of experiences.
        :param experience_size: The size of each online experiences, as an int.
            Ignored if `custom_split_strategy` is used.
        :param experience_split_strategy: A function that implements a custom
            splitting strategy. The function must accept an experience and
            return an experience's iterator. Defaults to None, which means
            that the standard splitting strategy will be used (which creates
            experiences of size `experience_size`).
            A good starting to understand the mechanism is to look at the
            implementation of the standard splitting function
            :func:`fixed_size_experience_split`.
        : param access_task_boundaries: If True the attributes related to task
            boundaries such as `is_first_subexp` and `is_last_subexp` become
            accessible during training.
        :param shuffle: If True, experiences will be split by first shuffling
            instances in each experience. Defaults to True.
        :param overlap_factor: The overlap factor between consecutive
            experiences. Defaults to 4.
        :param iters_per_virtual_epoch: The number of iterations per virtual epoch
            for each experience. Defaults to 10.

        """
        warnings.warn(
            "Deprecated. Use `split_online_stream` or similar methods to split"
            "single streams or experiences instead"
        )

        if stream_split_strategy == "fixed_size_split":
            split_strat = partial(
                _fixed_size_split,
                self,
                experience_size,
                access_task_boundaries,
                shuffle,
            )
        elif stream_split_strategy == "continuous_linear_decay":
            assert access_task_boundaries is False

            split_strat = partial(
                split_online_stream,
                experience_size=experience_size,
                iters_per_virtual_epoch=iters_per_virtual_epoch,
                beta=overlap_factor,
                shuffle=True,
            )
        else:
            raise ValueError("Unknown experience split strategy")

        streams_dict = {s.name: s for s in original_streams}
        if "train" not in streams_dict:
            raise ValueError("Missing train stream for `original_streams`.")
        if experiences is None:
            online_train_stream = split_strat(streams_dict["train"])
        else:
            if not isinstance(experiences, Iterable):
                experiences = [experiences]
            online_train_stream = split_strat(experiences)

        streams: List[CLStream] = [online_train_stream]
        for s in original_streams:
            s_wrapped = wrap_stream(
                new_name="original_" + s.name, new_benchmark=self, wrapped_stream=s
            )

            streams.append(s_wrapped)

        super().__init__(streams=streams)

