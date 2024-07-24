class MigrationRunner(object):
    """Class for running FiftyOne migrations.

    Args:
        head: the current revision
        destination: the destination revision
    """

    def __init__(
        self,
        head,
        destination,
        _revisions=None,
        _admin_revisions=None,
    ):
        pkg_ver = Version(foc.VERSION)
        head_ver = Version(head)
        dest_ver = Version(destination)
        if head_ver > pkg_ver or dest_ver > pkg_ver:
            raise EnvironmentError(
                "You must have fiftyone>=%s installed in order to migrate "
                "from v%s to v%s, but you are currently running fiftyone==%s."
                "\n\nSee https://docs.voxel51.com/getting_started/install.html#downgrading-fiftyone "
                "for information about downgrading FiftyOne."
                % (max(head_ver, dest_ver), head_ver, dest_ver, pkg_ver)
            )

        if _revisions is None:
            _revisions = _get_all_revisions()

        if _admin_revisions is None:
            _admin_revisions = _get_all_revisions(admin=True)

        self._head = head
        self._destination = destination
        self._revisions, self._direction = _get_revisions_to_run(
            head, destination, _revisions
        )
        self._admin_revisions, _ = _get_revisions_to_run(
            head, destination, _admin_revisions
        )

    @property
    def head(self):
        """The head revision."""
        return self._head

    @property
    def destination(self):
        """The destination revision."""
        return self._destination

    @property
    def direction(self):
        """The direction of the migration runner; one of ``("up", "down").``"""
        return self._direction

    @property
    def has_revisions(self):
        """Whether there are any revisions to run."""
        return bool(self._revisions)

    @property
    def has_admin_revisions(self):
        """Whether there are any admin revisions to run."""
        return bool(self._admin_revisions)

    @property
    def revisions(self):
        """The list of revisions that will be run by :meth:`run`."""
        return [r[0] for r in self._revisions]

    @property
    def admin_revisions(self):
        """The list of admin revisions that will be run by :meth:`run_admin`."""
        return [r[0] for r in self._admin_revisions]

    def run(self, dataset_name, verbose=False):
        """Runs any required migrations on the specified dataset.

        Args:
            dataset_name: the name of the dataset to migrate
            verbose (False): whether to log incremental migrations that are run
        """
        conn = foo.get_db_conn()
        for rev, module in self._revisions:
            if verbose:
                logger.info("Running v%s %s migration", rev, self.direction)

            fcn = etau.get_function(self.direction, module)
            fcn(conn, dataset_name)

    def run_admin(self, verbose=False):
        """Runs any required admin revisions.

        Args:
            verbose (False): whether to log incremental migrations that are run
        """
        client = foo.get_db_client()
        for rev, module in self._admin_revisions:
            if verbose:
                logger.info(
                    "Running v%s %s admin migration", rev, self.direction
                )

            fcn = etau.get_function(self.direction, module)
            fcn(client)

