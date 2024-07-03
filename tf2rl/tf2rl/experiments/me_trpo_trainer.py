    def get_argument(parser=None):
        parser = MPCTrainer.get_argument(parser)
        parser.add_argument("--n-collect-steps", type=int, default=100)
        parser.add_argument("--debug", action='store_true')
        return parser