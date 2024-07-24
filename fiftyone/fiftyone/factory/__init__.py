class DelegatedOperationPagingParams(object):
    """Paging parameters for delegated operations."""

    def __init__(
        self,
        sort_by: SortByField = SortByField.QUEUED_AT,
        sort_direction: SortDirection = SortDirection.DESCENDING,
        skip: int = 0,
        limit: int = 10,
    ):
        self.sort_by = sort_by
        self.sort_direction = sort_direction
        self.skip = skip
        self.limit = limit

