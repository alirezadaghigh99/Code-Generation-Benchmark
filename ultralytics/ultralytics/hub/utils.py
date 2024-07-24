class Events:
    """
    A class for collecting anonymous event analytics. Event analytics are enabled when sync=True in settings and
    disabled when sync=False. Run 'yolo settings' to see and update settings YAML file.

    Attributes:
        url (str): The URL to send anonymous events.
        rate_limit (float): The rate limit in seconds for sending events.
        metadata (dict): A dictionary containing metadata about the environment.
        enabled (bool): A flag to enable or disable Events based on certain conditions.
    """

    url = "https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw"

    def __init__(self):
        """Initializes the Events object with default values for events, rate_limit, and metadata."""
        self.events = []  # events list
        self.rate_limit = 60.0  # rate limit (seconds)
        self.t = 0.0  # rate limit timer (seconds)
        self.metadata = {
            "cli": Path(ARGV[0]).name == "yolo",
            "install": "git" if IS_GIT_DIR else "pip" if IS_PIP_PACKAGE else "other",
            "python": ".".join(platform.python_version_tuple()[:2]),  # i.e. 3.10
            "version": __version__,
            "env": ENVIRONMENT,
            "session_id": round(random.random() * 1e15),
            "engagement_time_msec": 1000,
        }
        self.enabled = (
            SETTINGS["sync"]
            and RANK in {-1, 0}
            and not TESTS_RUNNING
            and ONLINE
            and (IS_PIP_PACKAGE or get_git_origin_url() == "https://github.com/ultralytics/ultralytics.git")
        )

    def __call__(self, cfg):
        """
        Attempts to add a new event to the events list and send events if the rate limit is reached.

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
        """
        if not self.enabled:
            # Events disabled, do nothing
            return

        # Attempt to add to events
        if len(self.events) < 25:  # Events list limited to 25 events (drop any events past this)
            params = {
                **self.metadata,
                "task": cfg.task,
                "model": cfg.model if cfg.model in GITHUB_ASSETS_NAMES else "custom",
            }
            if cfg.mode == "export":
                params["format"] = cfg.format
            self.events.append({"name": cfg.mode, "params": params})

        # Check rate limit
        t = time.time()
        if (t - self.t) < self.rate_limit:
            # Time is under rate limiter, wait to send
            return

        # Time is over rate limiter, send now
        data = {"client_id": SETTINGS["uuid"], "events": self.events}  # SHA-256 anonymized UUID hash and events list

        # POST equivalent to requests.post(self.url, json=data)
        smart_request("post", self.url, json=data, retry=0, verbose=False)

        # Reset events and rate limit timer
        self.events = []
        self.t = t

