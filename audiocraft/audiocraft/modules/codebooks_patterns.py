class UnrolledPatternProvider(CodebooksPatternProvider):
    """Provider for unrolling codebooks pattern.
    This pattern provider enables to represent the codebook flattened completely or only to some extend
    while also specifying a given delay between the flattened codebooks representation, allowing to
    unroll the codebooks in the sequence.

    Example:
        1. Flattening of the codebooks.
        By default, the pattern provider will fully flatten the codebooks such as flattening=range(n_q),
        taking n_q = 3 and timesteps = 4:
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]]
        will result into:
        [[S, S, 1, S, S, 2, S, S, 3, S, S, 4],
         [S, 1, S, S, 2, S, S, 3, S, S, 4, S],
         [1, S, S, 2, S, S, 3, S, S, 4, S, S]]
        2. Partial flattening of the codebooks. The ``flattening`` parameter allows to specify the inner step
        for each of the codebook, allowing to define which codebook to flatten (or keep in parallel), for example
        taking n_q = 3, timesteps = 4 and flattening = [0, 1, 1]:
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]]
        will result into:
        [[S, 1, S, S, 2, S, S, 3, S, S, 4, S],
         [S, 1, S, S, 2, S, S, 3, S, S, 4, S],
         [1, S, S, 2, S, S, 3, S, S, 4, S, S]]
        3. Flattening with delay. The ``delay`` parameter allows to further unroll the sequence of codebooks
        allowing to specify the delay per codebook. Note that the delay between codebooks flattened to the
        same inner timestep should be coherent. For example, taking n_q = 3, timesteps = 4, flattening = [0, 1, 1]
        and delays = [0, 3, 3]:
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]]
        will result into:
        [[S, S, S, 1, S, 2, S, 3, S, 4],
         [S, S, S, 1, S, 2, S, 3, S, 4],
         [1, 2, 3, S, 4, S, 5, S, 6, S]]

    Args:
        n_q (int): Number of codebooks.
        flattening (list of int, optional): Flattening schema over the codebooks. If not defined,
            the codebooks will be flattened to 1 codebook per step, meaning that the sequence will
            have n_q extra steps for each timestep.
        delays (list of int, optional): Delay for each of the codebooks. If not defined,
            no delay is added and therefore will default to [0] * ``n_q``.
            Note that two codebooks that will be flattened to the same inner step
            should have the same delay, otherwise the pattern is considered as invalid.
    """
    FlattenedCodebook = namedtuple('FlattenedCodebook', ['codebooks', 'delay'])

    def __init__(self, n_q: int, flattening: tp.Optional[tp.List[int]] = None,
                 delays: tp.Optional[tp.List[int]] = None):
        super().__init__(n_q)
        if flattening is None:
            flattening = list(range(n_q))
        if delays is None:
            delays = [0] * n_q
        assert len(flattening) == n_q
        assert len(delays) == n_q
        assert sorted(flattening) == flattening
        assert sorted(delays) == delays
        self._flattened_codebooks = self._build_flattened_codebooks(delays, flattening)
        self.max_delay = max(delays)

    def _build_flattened_codebooks(self, delays: tp.List[int], flattening: tp.List[int]):
        """Build a flattened codebooks representation as a dictionary of inner step
        and the actual codebook indices corresponding to the flattened codebook. For convenience, we
        also store the delay associated to the flattened codebook to avoid maintaining an extra mapping.
        """
        flattened_codebooks: dict = {}
        for q, (inner_step, delay) in enumerate(zip(flattening, delays)):
            if inner_step not in flattened_codebooks:
                flat_codebook = UnrolledPatternProvider.FlattenedCodebook(codebooks=[q], delay=delay)
            else:
                flat_codebook = flattened_codebooks[inner_step]
                assert flat_codebook.delay == delay, (
                    "Delay and flattening between codebooks is inconsistent: ",
                    "two codebooks flattened to the same position should have the same delay."
                )
                flat_codebook.codebooks.append(q)
            flattened_codebooks[inner_step] = flat_codebook
        return flattened_codebooks

    @property
    def _num_inner_steps(self):
        """Number of inner steps to unroll between timesteps in order to flatten the codebooks.
        """
        return max([inner_step for inner_step in self._flattened_codebooks.keys()]) + 1

    def num_virtual_steps(self, timesteps: int) -> int:
        return timesteps * self._num_inner_steps + 1

    def get_pattern(self, timesteps: int) -> Pattern:
        """Builds pattern for delay across codebooks.

        Args:
            timesteps (int): Total number of timesteps.
        """
        # the PatternLayout is built as a tuple of sequence position and list of coordinates
        # so that it can be reordered properly given the required delay between codebooks of given timesteps
        indexed_out: list = [(-1, [])]
        max_timesteps = timesteps + self.max_delay
        for t in range(max_timesteps):
            # for each timestep, we unroll the flattened codebooks,
            # emitting the sequence step with the corresponding delay
            for step in range(self._num_inner_steps):
                if step in self._flattened_codebooks:
                    # we have codebooks at this virtual step to emit
                    step_codebooks = self._flattened_codebooks[step]
                    t_for_q = t + step_codebooks.delay
                    coords = [LayoutCoord(t, q) for q in step_codebooks.codebooks]
                    if t_for_q < max_timesteps and t < max_timesteps:
                        indexed_out.append((t_for_q, coords))
                else:
                    # there is no codebook in this virtual step so we emit an empty list
                    indexed_out.append((t, []))
        out = [coords for _, coords in sorted(indexed_out)]
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)

class DelayedPatternProvider(CodebooksPatternProvider):
    """Provider for delayed pattern across delayed codebooks.
    Codebooks are delayed in the sequence and sequence steps will contain codebooks
    from different timesteps.

    Example:
        Taking timesteps=4 and n_q=3, delays=None, the multi-codebook sequence:
        [[1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4]]
        The resulting sequence obtained from the returned pattern is:
        [[S, 1, 2, 3, 4],
        [S, S, 1, 2, 3],
        [S, S, S, 1, 2]]
        (with S being a special token)

    Args:
        n_q (int): Number of codebooks.
        delays (list of int, optional): Delay for each of the codebooks.
            If delays not defined, each codebook is delayed by 1 compared to the previous one.
        flatten_first (int): Flatten the first N timesteps.
        empty_initial (int): Prepend with N empty list of coordinates.
    """
    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None,
                 flatten_first: int = 0, empty_initial: int = 0):
        super().__init__(n_q)
        if delays is None:
            delays = list(range(n_q))
        self.delays = delays
        self.flatten_first = flatten_first
        self.empty_initial = empty_initial
        assert len(self.delays) == self.n_q
        assert sorted(self.delays) == self.delays

    def get_pattern(self, timesteps: int) -> Pattern:
        omit_special_token = self.empty_initial < 0
        out: PatternLayout = [] if omit_special_token else [[]]
        max_delay = max(self.delays)
        if self.empty_initial:
            out += [[] for _ in range(self.empty_initial)]
        if self.flatten_first:
            for t in range(min(timesteps, self.flatten_first)):
                for q in range(self.n_q):
                    out.append([LayoutCoord(t, q)])
        for t in range(self.flatten_first, timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= self.flatten_first:
                    v.append(LayoutCoord(t_for_q, q))
            out.append(v)
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)

