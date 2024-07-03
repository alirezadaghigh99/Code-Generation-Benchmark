class ConcatObsAndAction(Lambda):
    def __init__(self):
        return super().__init__(concat_obs_and_action)