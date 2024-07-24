class Monitor(_GymMonitor):
    """`Monitor` with PFRL's `ContinuingTimeLimit` support.

    `Agent` in PFRL might reset the env even when `done=False`
    if `ContinuingTimeLimit` returns `info['needs_reset']=True`,
    which is not expected for `gym.Monitor`.

    For details, see
    https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py
    """

    def _start(
        self,
        directory,
        video_callable=None,
        force=False,
        resume=False,
        write_upon_reset=False,
        uid=None,
        mode=None,
    ):
        if self.env_semantics_autoreset:
            raise NotImplementedError(
                "Detect 'semantics.autoreset=True' in `env.metadata`, "
                "which means the env is from deprecated OpenAI Universe."
            )
        ret = super()._start(
            directory=directory,
            video_callable=video_callable,
            force=force,
            resume=resume,
            write_upon_reset=write_upon_reset,
            uid=uid,
            mode=mode,
        )
        env_id = self.stats_recorder.env_id
        self.stats_recorder = _StatsRecorder(
            directory,
            "{}.episode_batch.{}".format(self.file_prefix, self.file_infix),
            autoreset=False,
            env_id=env_id,
        )
        if mode is not None:
            self._set_mode(mode)
        return ret

